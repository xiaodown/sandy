"""
Sandy's Discord bot — the main event loop.

Handles the full message pipeline:
  1. Receive message via discord.py on_message
  2. Vision: describe any image attachments
  3. Bouncer: decide whether to respond and whether to use a tool
  4. Tool dispatch: execute the recommended tool, if any
  5. RAG: query vector memory for semantically similar past messages
  6. Brain: generate Sandy's reply with all context injected
  7. Memory: tag, summarize, store to Recall + embed to ChromaDB (background)

Entry point is ``python -m sandy`` (see __main__.py).
"""


import asyncio
import io
import os
import logging
import re
import time
from collections.abc import Awaitable

import discord
from dotenv import load_dotenv
from PIL import Image

from .logconf import get_logger
from .recall import ChatDatabase
from .registry import Registry
from .last10 import Last10, resolve_mentions, SyntheticMessage, _SyntheticAuthor, _SyntheticGuild, _SyntheticChannel
from .memory import MemoryClient
from .llm import OllamaInterface
from .trace import TurnTrace, event_payload
from .vector_memory import VectorMemory
from . import tools

load_dotenv()

logger = get_logger("sandy.bot")

intents = discord.Intents.default()
intents.members = True
intents.message_content = True
bot = discord.Client(intents=intents)
DISCORD_API_KEY = os.getenv("DISCORD_API_KEY")

# Quiet the discord.py library — it's chatty at INFO but we still want
# WARNING and above (including event-handler tracebacks).
logging.getLogger("discord").setLevel(logging.WARNING)

registry = Registry()
cache = Last10(maxlen=10, registry=registry)
llm = OllamaInterface()
vector_memory = VectorMemory()

# Recall database — path built from DB_DIR + RECALL_DB_NAME so the test/prod
# switch is controlled by a single env var.
_db_dir = os.getenv("DB_DIR", "data/prod/")
_recall_db_name = os.getenv("RECALL_DB_NAME", "recall.db")
recall_db = ChatDatabase(os.path.join(_db_dir, _recall_db_name))
recall_db.init_db()

# Give tools.py a reference to the shared Recall DB
tools.init_recall_db(recall_db)

memory = MemoryClient(db=recall_db, llm=llm, vector_memory=vector_memory)

# Guard so cache seeding only runs once, even if on_ready fires on reconnect.
_cache_seeded = False
_memory_worker_task: asyncio.Task | None = None

# ---------------------------------------------------------------------------
# Image attachment handling
# ---------------------------------------------------------------------------

# Whitelisted image MIME types for vision analysis.
# SVG is intentionally excluded: it's an XML vector format that vision models
# can't render, and it's a documented CVE magnet. Same reasoning for PDF etc.
_VISION_CONTENT_TYPES: frozenset[str] = frozenset({
    "image/jpeg",
    "image/png",
    "image/gif",
    "image/webp",
})

# Skip files over this size — avoids downloading huge uploads on slow connections.
_MAX_IMAGE_BYTES = 20 * 1024 * 1024  # 20 MB

# Discord hard-caps message content at 2000 chars.
_DISCORD_MESSAGE_LIMIT = 2000

# ---------------------------------------------------------------------------
# Tool result framing for brain injection
# ---------------------------------------------------------------------------

_MEMORY_TOOLS: frozenset[str] = frozenset({
    "recall_recent", "recall_from_user", "recall_by_topic", "search_memories",
})


class BackgroundTaskSupervisor:
    """Track background tasks so failures are logged and shutdown is orderly."""

    def __init__(self) -> None:
        self._tasks: set[asyncio.Task] = set()

    def create_task(self, coro: Awaitable[object], *, name: str) -> asyncio.Task:
        task = asyncio.create_task(coro, name=name)
        self._tasks.add(task)
        task.add_done_callback(self._on_done)
        return task

    def _on_done(self, task: asyncio.Task) -> None:
        self._tasks.discard(task)
        try:
            task.result()
        except asyncio.CancelledError:
            logger.info("Background task cancelled: %s", task.get_name())
        except Exception:
            logger.exception("Background task failed: %s", task.get_name())

    async def shutdown(self) -> None:
        if not self._tasks:
            return

        pending = tuple(self._tasks)
        logger.info("Waiting for %d background task(s) to finish", len(pending))
        await asyncio.gather(*pending, return_exceptions=True)


class MemoryWorker:
    """Serialize deferred memory work behind a small in-process queue."""

    _SENTINEL = object()

    def __init__(self, handler) -> None:
        self._handler = handler
        self._queue: asyncio.Queue[object] = asyncio.Queue()
        self._closed = False

    async def run(self) -> None:
        logger.info("Memory worker started")
        while True:
            item = await self._queue.get()
            try:
                if item is self._SENTINEL:
                    logger.info("Memory worker stopping")
                    return

                message, image_descriptions = item
                await self._handler(message, image_descriptions=image_descriptions)
            finally:
                self._queue.task_done()

    async def enqueue(
        self,
        message: discord.Message,
        image_descriptions: list[str] | None = None,
    ) -> None:
        if self._closed:
            raise RuntimeError("Memory worker is closed")
        await self._queue.put((message, image_descriptions))

    async def shutdown(self) -> None:
        if self._closed:
            return

        self._closed = True
        await self._queue.join()
        await self._queue.put(self._SENTINEL)


background_tasks = BackgroundTaskSupervisor()
memory_worker = MemoryWorker(memory.process_and_store)


def _format_tool_context(tool_name: str, result: str) -> str:
    """Frame a tool result for injection into the brain's system prompt.

    The framing varies by tool type so Sandy talks about remembering vs.
    looking something up, matching natural personality.
    """
    if tool_name == "search_web":
        return f"## You just looked this up online\n{result}"
    elif tool_name == "get_current_time":
        return f"## You just checked the time\n{result}"
    elif tool_name == "dice_roll":
        return f"## You just rolled some dice\n{result}"
    elif tool_name in _MEMORY_TOOLS:
        return f"## You just recalled this from memory\n{result}"
    else:
        return f"## Additional context\n{result}"


def _trace_event(
    trace: TurnTrace,
    stage: str,
    *,
    status: str = "ok",
    duration_ms: int | None = None,
    **fields: object,
) -> None:
    """Emit one structured trace event into the normal logs."""
    payload = event_payload(
        trace,
        stage,
        status=status,
        duration_ms=duration_ms,
        **fields,
    )
    logger.info("TRACE %s", payload)


def _split_reply(reply: str, limit: int = _DISCORD_MESSAGE_LIMIT) -> list[str]:
    """Split a reply into Discord-sized chunks, preferring natural boundaries."""
    if len(reply) <= limit:
        return [reply]

    chunks: list[str] = []
    remaining = reply.strip()

    while remaining:
        if len(remaining) <= limit:
            chunks.append(remaining)
            break

        split_at = remaining.rfind("\n\n", 0, limit + 1)
        if split_at == -1:
            split_at = remaining.rfind("\n", 0, limit + 1)
        if split_at == -1:
            split_at = remaining.rfind(" ", 0, limit + 1)
        if split_at == -1 or split_at < limit // 2:
            split_at = limit

        chunk = remaining[:split_at].strip()
        if not chunk:
            chunk = remaining[:limit]
            split_at = limit

        chunks.append(chunk)
        remaining = remaining[split_at:].lstrip()

    return chunks


def _trim_to_last_boundary(reply: str) -> str:
    """Trim a reply back to the last strong sentence or paragraph boundary."""
    sentence_matches = list(re.finditer(r'[.!?]["\')\]]?(?:\s|$)', reply))
    if sentence_matches:
        cut = sentence_matches[-1].end()
        trimmed = reply[:cut].strip()
        if trimmed:
            return trimmed

    split_at = reply.rfind("\n\n")
    if split_at != -1:
        trimmed = reply[:split_at].strip()
        if trimmed:
            return trimmed

    split_at = reply.rfind("\n")
    if split_at != -1:
        trimmed = reply[:split_at].strip()
        if trimmed:
            return trimmed

    return reply.strip()


def _trim_truncated_reply(reply: str) -> str:
    """Trim a likely-truncated reply back to a clearly complete stopping point."""
    paragraph_matches = list(re.finditer(r"\n\s*\n", reply))
    sentence_matches = list(re.finditer(r'[.!?]["\')\]]?(?:\s|$)', reply))

    if paragraph_matches:
        paragraph_cut = paragraph_matches[-1].start()
        trimmed = reply[:paragraph_cut].strip()
        if trimmed and len(trimmed) >= max(80, len(reply) // 3):
            return trimmed

    if sentence_matches:
        sentence_cut = sentence_matches[-1].end()
        trimmed = reply[:sentence_cut].strip()
        if trimmed and len(trimmed) >= max(80, len(reply) // 3):
            return trimmed

    return _trim_to_last_boundary(reply)


def _looks_truncated(reply: str, done_reason: str | None = None) -> bool:
    """Heuristic for obvious generation cutoffs."""
    text = reply.rstrip()
    if not text:
        return False

    if done_reason == "length":
        return True

    if text[-1] in ".!?\"')]}":
        return False

    if text[-1] in ",:;/-([{":
        return True

    last_word_match = re.search(r"([A-Za-z']+)\s*$", text)
    last_word = last_word_match.group(1).lower() if last_word_match else ""
    if last_word in {
        "a", "an", "and", "are", "as", "at", "but", "for", "from",
        "i", "if", "in", "is", "it", "like", "my", "of", "on", "or",
        "so", "that", "the", "to", "was", "with", "you", "your",
    }:
        return True

    return len(last_word) <= 2


def _finalize_reply(reply: str | None, *, done_reason: str | None = None) -> str | None:
    """Normalize reply text and trim obvious model cutoffs to a clean boundary."""
    if reply is None:
        return None

    cleaned = reply.strip()
    if not cleaned:
        return None

    if not _looks_truncated(cleaned, done_reason=done_reason):
        return cleaned

    trimmed = _trim_truncated_reply(cleaned)
    if trimmed and trimmed != cleaned:
        logger.warning(
            "Brain reply looked truncated; trimmed from %d to %d chars",
            len(cleaned),
            len(trimmed),
        )
        return trimmed

    return cleaned


async def _send_reply(message: discord.Message, reply: str) -> int:
    """Send a reply, splitting into multiple Discord messages if needed."""
    parts = _split_reply(reply)
    if len(parts) > 1:
        logger.warning(
            "Reply exceeded Discord limit (%d chars) - sending %d chunks",
            len(reply),
            len(parts),
        )

    for part in parts:
        await message.channel.send(part)
    return len(parts)


async def _describe_attachments(message: discord.Message) -> list[str]:
    """Download and describe all image attachments in a Discord message.

    Returns a list of description strings, one per successfully processed
    image. Non-image attachments and oversized files are silently skipped.
    Order matches attachment order in the message.
    """
    descriptions: list[str] = []
    for attachment in message.attachments:
        # Use Discord's own content_type — it's set server-side and reliable.
        # Split on ';' to strip any charset/boundary params.
        content_type = (attachment.content_type or "").split(";")[0].strip().lower()
        if content_type not in _VISION_CONTENT_TYPES:
            logger.debug(
                "Skipping attachment %s (type %s — not a supported image format)",
                attachment.filename, content_type or "unknown",
            )
            continue
        if attachment.size > _MAX_IMAGE_BYTES:
            logger.warning(
                "Skipping oversized image %s (%d MB)",
                attachment.filename, attachment.size // (1024 * 1024),
            )
            continue
        try:
            image_bytes = await attachment.read()
        except Exception as exc:
            logger.error("Failed to download attachment %s: %s", attachment.filename, exc)
            continue
        # WebP causes a 500 from ollama's vision runner — convert to JPEG in
        # memory first. Pillow handles all our whitelisted formats so this is
        # safe to do unconditionally, but we only bother for WebP since the
        # others work fine as-is.
        if content_type == "image/webp":
            try:
                with Image.open(io.BytesIO(image_bytes)) as img:
                    buf = io.BytesIO()
                    img.convert("RGB").save(buf, format="JPEG", quality=90)
                    image_bytes = buf.getvalue()
                logger.debug("Converted WebP→JPEG for %s", attachment.filename)
            except Exception as exc:
                logger.error("WebP conversion failed for %s: %s", attachment.filename, exc)
                continue
        desc = await llm.ask_vision(image_bytes)
        if desc:
            descriptions.append(desc)
            logger.info(
                "Vision described %s: %s", attachment.filename, desc[:80] + ("…" if len(desc) > 80 else "")
            )
        else:
            logger.warning("Vision returned nothing for %s", attachment.filename)
    return descriptions


def _build_augmented_content(message: discord.Message, descriptions: list[str]) -> str:
    """Compose augmented message content with image descriptions injected.

    Format varies based on whether the message had text and how many images:
      - text + 1 image:  "<original text>\n[<name> also attached an image: <desc>]"
      - text + N images: "<original text>\n[<name> also attached N images]\n[Image 1: ...]..."
      - pure image(s):   "[<name> pasted an image/N images]\n[Image: <desc>]..."
    """
    name = message.author.display_name
    original = resolve_mentions(message.content, message.mentions).strip()
    n = len(descriptions)

    if n == 1:
        desc = descriptions[0]
        if original:
            return f"{original}\n[{name} also attached an image: {desc}]"
        else:
            return f"[{name} pasted an image into the chat]\n[Image: {desc}]"
    else:
        image_lines = "\n".join(f"[Image {i}: {d}]" for i, d in enumerate(descriptions, 1))
        if original:
            return f"{original}\n[{name} also attached {n} images]\n{image_lines}"
        else:
            return f"[{name} pasted {n} images into the chat]\n{image_lines}"


@bot.event
async def on_ready():
    """Event handler for when the bot is ready."""
    global _cache_seeded, _memory_worker_task

    logger.info("Logged in as %s (%s)", bot.user.name, bot.user.id)
    if _memory_worker_task is None or _memory_worker_task.done():
        _memory_worker_task = background_tasks.create_task(
            memory_worker.run(),
            name="memory-worker",
        )
    if not _cache_seeded:
        seeded = await memory.seed_cache(cache)
        logger.info("Cache seeded with %d message(s) from Recall", seeded)
        _cache_seeded = True
        ready_info=f"       ###   BOT READY   ###\n\n"
        ready_info+=f"      * bot logged in as {bot.user.name} ({bot.user.id})\n"
        guild_count = 0
        for guild in bot.guilds:
            ready_info+=f"      * attached to {guild.name} ({guild.id})\n"
            guild_count = guild_count + 1
        ready_info+=f"      * {bot.user.name} is on {str(guild_count)} servers\n"
        logger.warning("\n\n%s", ready_info)


@bot.event
async def on_message(message: discord.Message):
    """Event handler for incoming messages."""
    # Ignore DMs — guild context is required for registry, cache, and memory.
    # DM support can be added later if needed.
    if message.guild is None:
        return

    trace = TurnTrace.from_message(message)
    turn_started = time.perf_counter()
    _trace_event(
        trace,
        "message_received",
        content_chars=len(message.content or ""),
        attachments=len(message.attachments),
        author_is_bot=message.author.bot,
    )

    # Update the registry (server / channel / user lookup cache)
    # Note: bot's own messages are included — they're part of the conversation context
    # registry.ensure_seen() uses sqlite3 (blocking I/O), so run it in a thread.
    background_tasks.create_task(
        asyncio.to_thread(registry.ensure_seen, message),
        name=f"registry.ensure_seen:{message.id}",
    )

    logger.info(
        "[%s/%s] %s%s: %s",
        message.guild.name, message.channel.name, message.author.display_name,
        f" [{len(message.attachments)} attachment(s)]" if message.attachments else "",
        resolve_mentions(message.content, message.mentions),
    )

    if message.author.bot:
        # Store bot messages (including Sandy's own replies) in Recall and last10
        # so they appear in conversation history, but skip bouncer/brain entirely.
        logger.debug("Bot message from %s — storing and skipping pipeline", message.author.display_name)
        cache.add(message)
        await memory_worker.enqueue(message)
        _trace_event(trace, "memory_enqueued", source="bot_message")
        _trace_event(
            trace,
            "turn_completed",
            duration_ms=int((time.perf_counter() - turn_started) * 1000),
            replied=False,
            bot_message=True,
        )
        return

    # --- Image attachment processing -----------------------------------
    # Describe any image attachments before adding to cache, so the
    # bouncer and brain both see the image content in context.
    # The original message is always passed to process_and_store — Recall
    # stores what was actually said, not our augmented version.
    vision_started = time.perf_counter()
    image_descriptions = await _describe_attachments(message)
    _trace_event(
        trace,
        "vision_completed",
        duration_ms=int((time.perf_counter() - vision_started) * 1000),
        image_count=len(image_descriptions),
    )
    if image_descriptions:
        augmented_content = _build_augmented_content(message, image_descriptions)
        cache_message = SyntheticMessage(
            content=augmented_content,
            created_at=message.created_at,
            author=_SyntheticAuthor(
                id=message.author.id,
                display_name=message.author.display_name,
                bot=message.author.bot,
            ),
            guild=_SyntheticGuild(id=message.guild.id, name=message.guild.name),
            channel=_SyntheticChannel(id=message.channel.id, name=message.channel.name),
            mentions=message.mentions,
        )
        cache.add(cache_message)
        # Use augmented content for RAG so image-only messages still get
        # a meaningful semantic query (message.content would be empty).
        rag_query_text = augmented_content
    else:
        cache.add(message)
        rag_query_text = message.content

    # --- Bouncer decision (respond + tool) ----------------------------
    history = cache.get(message.guild.id, message.channel.id)
    bouncer_started = time.perf_counter()
    bouncer_result = await llm.ask_bouncer(history.format(), bot_name=bot.user.display_name)

    logger.info(
        "Bouncer → respond=%s tool=%s(%s)",
        bouncer_result.should_respond,
        bouncer_result.recommended_tool or "none",
        bouncer_result.use_tool,
    )
    _trace_event(
        trace,
        "bouncer_completed",
        duration_ms=int((time.perf_counter() - bouncer_started) * 1000),
        should_respond=bouncer_result.should_respond,
        use_tool=bouncer_result.use_tool,
        tool_name=bouncer_result.recommended_tool,
    )

    await memory_worker.enqueue(message, image_descriptions=image_descriptions)
    _trace_event(trace, "memory_enqueued", source="user_message")

    if bouncer_result.should_respond:
        # Show "Sandy is typing..." for the entire duration of tool call +
        # LLM generation.  channel.typing() is an async context manager that
        # sends the indicator and refreshes it every 5 seconds automatically.
        async with message.channel.typing():
            # --- Tool call (if bouncer recommended one) ----------------
            tool_context = None
            if bouncer_result.use_tool and bouncer_result.recommended_tool:
                if bouncer_result.recommended_tool not in tools.KNOWN_TOOLS:
                    logger.warning(
                        "Bouncer recommended unknown tool %r — ignoring",
                        bouncer_result.recommended_tool,
                    )
                    _trace_event(
                        trace,
                        "tool_completed",
                        status="ignored",
                        tool_name=bouncer_result.recommended_tool,
                    )
                else:
                    logger.info(
                        "Dispatching tool %s with params %s",
                        bouncer_result.recommended_tool,
                        bouncer_result.tool_parameters or {},
                    )
                    tool_started = time.perf_counter()
                    _trace_event(
                        trace,
                        "tool_started",
                        tool_name=bouncer_result.recommended_tool,
                    )
                    tool_result = await tools.dispatch(
                        bouncer_result.recommended_tool,
                        bouncer_result.tool_parameters or {},
                        server_id=message.guild.id,
                        server_name=message.guild.name,
                    )
                    _trace_event(
                        trace,
                        "tool_completed",
                        duration_ms=int((time.perf_counter() - tool_started) * 1000),
                        tool_name=bouncer_result.recommended_tool,
                        result_chars=len(tool_result or ""),
                    )
                    tool_context = _format_tool_context(
                        bouncer_result.recommended_tool, tool_result,
                    )

            # --- RAG query ---------------------------------------------
            ollama_history = history.to_ollama_messages(bot.user.id)
            retrieval_started = time.perf_counter()
            rag_context = await vector_memory.query(
                rag_query_text,
                server_id=message.guild.id,
            )
            _trace_event(
                trace,
                "retrieval_completed",
                duration_ms=int((time.perf_counter() - retrieval_started) * 1000),
                context_chars=len(rag_context or ""),
            )

            # --- Brain response ----------------------------------------
            brain_started = time.perf_counter()
            brain_response = await llm.ask_brain(
                ollama_history,
                bot_name=bot.user.display_name,
                server_name=message.guild.name,
                channel_name=message.channel.name,
                rag_context=rag_context,
                tool_context=tool_context,
            )
            _trace_event(
                trace,
                "brain_completed",
                duration_ms=int((time.perf_counter() - brain_started) * 1000),
                done_reason=brain_response.done_reason if brain_response else None,
                reply_chars=len((brain_response.content if brain_response else "") or ""),
            )
            reply = _finalize_reply(
                brain_response.content if brain_response else None,
                done_reason=brain_response.done_reason if brain_response else None,
            )

            if reply:
                try:
                    send_started = time.perf_counter()
                    _trace_event(trace, "reply_send_started", reply_chars=len(reply))
                    sent_parts = await _send_reply(message, reply)
                    _trace_event(
                        trace,
                        "reply_send_completed",
                        duration_ms=int((time.perf_counter() - send_started) * 1000),
                        reply_chars=len(reply),
                        message_parts=sent_parts,
                    )
                    logger.info(
                        "Brain replied in %s/%s (%d chars, %d message(s), done_reason=%s)",
                        message.guild.name,
                        message.channel.name,
                        len(reply),
                        sent_parts,
                        brain_response.done_reason if brain_response else None,
                    )
                except Exception:
                    _trace_event(trace, "reply_send_completed", status="error")
                    logger.exception(
                        "Reply send failed in %s/%s after generation",
                        message.guild.name,
                        message.channel.name,
                    )
            else:
                _trace_event(trace, "reply_skipped", status="empty_reply")
                logger.warning("Brain returned None for message in %s/%s — not sending",
                               message.guild.name, message.channel.name)

    _trace_event(
        trace,
        "turn_completed",
        duration_ms=int((time.perf_counter() - turn_started) * 1000),
        replied=bouncer_result.should_respond,
    )


async def shutdown_background_work() -> None:
    """Flush queued background work before process exit."""
    await memory_worker.shutdown()
    await background_tasks.shutdown()
