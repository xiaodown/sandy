"""
Sandy's message pipeline and supporting runtime state.

bot.py should remain thin Discord glue. This module owns the application-side
objects needed to turn a Discord message into Sandy behavior.
"""

import asyncio
import io
import os
import re
import time
from dataclasses import dataclass

import discord
from dotenv import load_dotenv
from PIL import Image

from . import tools
from .last10 import (
    Last10,
    SyntheticMessage,
    _SyntheticAuthor,
    _SyntheticChannel,
    _SyntheticGuild,
    resolve_mentions,
)
from .llm import OllamaInterface
from .logconf import emit_forensic_record, get_logger
from .memory import MemoryClient
from .paths import resolve_runtime_path
from .recall import ChatDatabase
from .registry import Registry
from .runtime_state import RuntimeState
from .trace import TurnTrace, event_payload, forensic_payload
from .vector_memory import VectorMemory

load_dotenv()

logger = get_logger("sandy.bot")

_VISION_CONTENT_TYPES: frozenset[str] = frozenset({
    "image/jpeg",
    "image/png",
    "image/gif",
    "image/webp",
})
_MAX_IMAGE_BYTES = 20 * 1024 * 1024
_DISCORD_MESSAGE_LIMIT = 2000
_MEMORY_TOOLS: frozenset[str] = frozenset({
    "recall_recent", "recall_from_user", "recall_by_topic", "search_memories",
})
_RAG_BYPASS_TOOLS: frozenset[str] = frozenset({"steam_browse"})


@dataclass(slots=True)
class AttachmentProcessingResult:
    descriptions: list[str]
    fallback_count: int = 0
    fallback_reasons: list[str] | None = None


class MemoryWorker:
    """Serialize deferred memory work behind a small in-process queue."""

    _SENTINEL = object()

    def __init__(self, handler, *, runtime_state: RuntimeState | None = None) -> None:
        self._handler = handler
        self._runtime_state = runtime_state
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
                if self._runtime_state is not None:
                    self._runtime_state.memory_processing_started(
                        message_id=getattr(message, "id", None),
                    )
                try:
                    await self._handler(message, image_descriptions=image_descriptions)
                except Exception:
                    logger.exception("Memory worker handler failed for message %s", getattr(message, "id", "?"))
                finally:
                    if self._runtime_state is not None:
                        self._runtime_state.memory_processing_finished(
                            message_id=getattr(message, "id", None),
                        )
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
        if self._runtime_state is not None:
            self._runtime_state.memory_enqueued()

    async def shutdown(self) -> None:
        if self._closed:
            return

        self._closed = True
        await self._queue.join()
        await self._queue.put(self._SENTINEL)


def _format_tool_context(tool_name: str, result: str) -> str:
    if tool_name == "search_web":
        return f"## You just looked this up online\n{result}"
    if tool_name == "steam_browse":
        return f"## You just checked Steam\n{result}"
    if tool_name == "get_current_time":
        return f"## You just checked the time\n{result}"
    if tool_name == "dice_roll":
        return f"## You just rolled some dice\n{result}"
    if tool_name in _MEMORY_TOOLS:
        return f"## You just recalled this from memory\n{result}"
    return f"## Additional context\n{result}"


def _trace_event(
    trace: TurnTrace,
    stage: str,
    *,
    status: str = "ok",
    duration_ms: int | None = None,
    **fields: object,
) -> None:
    payload = event_payload(
        trace,
        stage,
        status=status,
        duration_ms=duration_ms,
        **fields,
    )
    logger.info(
        "TRACE %s",
        payload,
        extra={"event_payload": payload, "log_to_console": False},
    )


def _forensic_event(trace: TurnTrace, artifact: str, **fields: object) -> None:
    emit_forensic_record(
        logger,
        f"FORENSIC {artifact}",
        forensic_payload(trace, artifact, **fields),
    )


def _split_reply(reply: str, limit: int = _DISCORD_MESSAGE_LIMIT) -> list[str]:
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


def _fallback_attachment_description(reason: str) -> str:
    return f"attached image could not be inspected because {reason}"


async def _describe_attachments(message: discord.Message, llm) -> AttachmentProcessingResult:
    descriptions: list[str] = []
    fallback_reasons: list[str] = []
    for attachment in message.attachments:
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
            descriptions.append(_fallback_attachment_description("the file was too large"))
            fallback_reasons.append("oversized")
            continue
        try:
            image_bytes = await attachment.read()
        except Exception as exc:
            logger.error("Failed to download attachment %s: %s", attachment.filename, exc)
            descriptions.append(_fallback_attachment_description("it could not be downloaded"))
            fallback_reasons.append("download_failed")
            continue
        if content_type == "image/webp":
            try:
                with Image.open(io.BytesIO(image_bytes)) as img:
                    buf = io.BytesIO()
                    img.convert("RGB").save(buf, format="JPEG", quality=90)
                    image_bytes = buf.getvalue()
                logger.debug("Converted WebP→JPEG for %s", attachment.filename)
            except Exception as exc:
                logger.error("WebP conversion failed for %s: %s", attachment.filename, exc)
                descriptions.append(_fallback_attachment_description("it could not be processed"))
                fallback_reasons.append("conversion_failed")
                continue
        desc = await llm.ask_vision(image_bytes)
        if desc:
            descriptions.append(desc)
            logger.info(
                "Vision described %s: %s", attachment.filename, desc[:80] + ("…" if len(desc) > 80 else "")
            )
        else:
            logger.warning("Vision returned nothing for %s", attachment.filename)
            descriptions.append(_fallback_attachment_description("no description could be generated"))
            fallback_reasons.append("empty_description")
    return AttachmentProcessingResult(
        descriptions=descriptions,
        fallback_count=len(fallback_reasons),
        fallback_reasons=fallback_reasons,
    )


def _build_augmented_content(message: discord.Message, descriptions: list[str]) -> str:
    name = message.author.display_name
    original = resolve_mentions(message.content, message.mentions).strip()
    n = len(descriptions)

    if n == 1:
        desc = descriptions[0]
        if original:
            return f"{original}\n[{name} also attached an image: {desc}]"
        return f"[{name} pasted an image into the chat]\n[Image: {desc}]"

    image_lines = "\n".join(f"[Image {i}: {d}]" for i, d in enumerate(descriptions, 1))
    if original:
        return f"{original}\n[{name} also attached {n} images]\n{image_lines}"
    return f"[{name} pasted {n} images into the chat]\n{image_lines}"


class SandyPipeline:
    """Application-side owner of Sandy's message pipeline."""

    def __init__(
        self,
        *,
        background_tasks,
        registry: Registry,
        cache: Last10,
        llm: OllamaInterface,
        vector_memory: VectorMemory,
        recall_db: ChatDatabase,
        memory: MemoryClient,
        memory_worker: MemoryWorker,
        runtime_state: RuntimeState,
        tools_module=tools,
        trace_event=_trace_event,
    ) -> None:
        self.background_tasks = background_tasks
        self.registry = registry
        self.cache = cache
        self.llm = llm
        self.vector_memory = vector_memory
        self.recall_db = recall_db
        self.memory = memory
        self.memory_worker = memory_worker
        self.runtime_state = runtime_state
        self.tools_module = tools_module
        self.trace_event = trace_event
        self._cache_seeded = False
        self._memory_worker_task: asyncio.Task | None = None

    async def describe_attachments(self, message: discord.Message) -> AttachmentProcessingResult:
        return await _describe_attachments(message, self.llm)

    async def send_reply(self, message: discord.Message, reply: str) -> int:
        return await _send_reply(message, reply)

    async def on_ready(self, bot: discord.Client) -> None:
        logger.info("Logged in as %s (%s)", bot.user.name, bot.user.id)
        if self._memory_worker_task is None or self._memory_worker_task.done():
            self._memory_worker_task = self.background_tasks.create_task(
                self.memory_worker.run(),
                name="memory-worker",
            )
        if not self._cache_seeded:
            seeded = await self.memory.seed_cache(self.cache)
            logger.info("Cache seeded with %d message(s) from Recall", seeded)
            self._cache_seeded = True
            ready_info = "       ###   BOT READY   ###\n\n"
            ready_info += f"      * bot logged in as {bot.user.name} ({bot.user.id})\n"
            guild_count = 0
            for guild in bot.guilds:
                ready_info += f"      * attached to {guild.name} ({guild.id})\n"
                guild_count += 1
            ready_info += f"      * {bot.user.name} is on {guild_count} servers\n"
            logger.warning("\n\n%s", ready_info)

    def _log_message_received(self, message: discord.Message) -> None:
        logger.info(
            "[%s/%s] %s%s: %s",
            message.guild.name,
            message.channel.name,
            message.author.display_name,
            f" [{len(message.attachments)} attachment(s)]" if message.attachments else "",
            resolve_mentions(message.content, message.mentions),
        )

    def _add_message_to_cache(
        self,
        message: discord.Message,
        image_descriptions: list[str],
    ) -> str:
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
            self.cache.add(cache_message)
            return augmented_content

        self.cache.add(message)
        return message.content

    async def _run_tool_step(self, message: discord.Message, trace: TurnTrace, bouncer_result) -> str | None:
        if not (bouncer_result.use_tool and bouncer_result.recommended_tool):
            return None

        if bouncer_result.recommended_tool not in self.tools_module.KNOWN_TOOLS:
            logger.warning(
                "Bouncer recommended unknown tool %r — ignoring",
                bouncer_result.recommended_tool,
            )
            self.trace_event(
                trace,
                "tool_completed",
                status="ignored",
                tool_name=bouncer_result.recommended_tool,
            )
            return None

        logger.debug(
            "Dispatching tool %s with params %s",
            bouncer_result.recommended_tool,
            bouncer_result.tool_parameters or {},
        )
        tool_started = time.perf_counter()
        self.runtime_state.update_turn_stage(trace, "tool_started")
        self.trace_event(
            trace,
            "tool_started",
            tool_name=bouncer_result.recommended_tool,
        )
        tool_result = await self.tools_module.dispatch(
            bouncer_result.recommended_tool,
            bouncer_result.tool_parameters or {},
            server_id=message.guild.id,
            server_name=message.guild.name,
        )
        self.trace_event(
            trace,
            "tool_completed",
            duration_ms=int((time.perf_counter() - tool_started) * 1000),
            tool_name=bouncer_result.recommended_tool,
            result_chars=len(tool_result or ""),
        )
        _forensic_event(
            trace,
            "tool_call",
            tool_name=bouncer_result.recommended_tool,
            arguments=bouncer_result.tool_parameters or {},
            result=tool_result,
            tool_context=_format_tool_context(
                bouncer_result.recommended_tool,
                tool_result,
            ),
        )
        return _format_tool_context(
            bouncer_result.recommended_tool,
            tool_result,
        )

    async def _run_reply_pipeline(
        self,
        message: discord.Message,
        *,
        bot_user,
        trace: TurnTrace,
        history,
        rag_query_text: str,
        bouncer_result,
    ) -> bool:
        # 5. Optional tool execution / tool-context construction
        tool_context = await self._run_tool_step(message, trace, bouncer_result)

        # 6. Semantic retrieval (RAG)
        ollama_history = history.to_ollama_messages(bot_user.id)
        if bouncer_result.recommended_tool in _RAG_BYPASS_TOOLS:
            self.runtime_state.update_turn_stage(trace, "retrieval_skipped")
            rag_context = ""
            self.trace_event(
                trace,
                "retrieval_completed",
                status="skipped",
                skipped_reason=f"tool:{bouncer_result.recommended_tool}",
                context_chars=0,
            )
            _forensic_event(
                trace,
                "retrieval",
                query_text=rag_query_text,
                rag_context="",
                ollama_history=ollama_history,
                skipped_reason=f"tool:{bouncer_result.recommended_tool}",
            )
        else:
            retrieval_started = time.perf_counter()
            self.runtime_state.update_turn_stage(trace, "retrieval")
            rag_context = await self.vector_memory.query(
                rag_query_text,
                server_id=message.guild.id,
            )
            self.trace_event(
                trace,
                "retrieval_completed",
                duration_ms=int((time.perf_counter() - retrieval_started) * 1000),
                context_chars=len(rag_context or ""),
            )
            _forensic_event(
                trace,
                "retrieval",
                query_text=rag_query_text,
                rag_context=rag_context,
                ollama_history=ollama_history,
            )

        # 7. Brain generation
        brain_started = time.perf_counter()
        self.runtime_state.update_turn_stage(trace, "brain")
        brain_response = await self.llm.ask_brain(
            ollama_history,
            bot_name=bot_user.display_name,
            server_name=message.guild.name,
            channel_name=message.channel.name,
            rag_context=rag_context,
            tool_context=tool_context,
            trace=trace,
        )
        self.trace_event(
            trace,
            "brain_completed",
            duration_ms=int((time.perf_counter() - brain_started) * 1000),
            done_reason=brain_response.done_reason if brain_response else None,
            reply_chars=len((brain_response.content if brain_response else "") or ""),
        )

        # 8. Reply finalization / truncation cleanup
        reply = _finalize_reply(
            brain_response.content if brain_response else None,
            done_reason=brain_response.done_reason if brain_response else None,
        )
        _forensic_event(
            trace,
            "reply_output",
            raw_reply=brain_response.content if brain_response else None,
            finalized_reply=reply,
            done_reason=brain_response.done_reason if brain_response else None,
        )

        if not reply:
            # 9. Reply skipped
            self.trace_event(trace, "reply_skipped", status="empty_reply")
            logger.warning(
                "Brain returned None for message in %s/%s — not sending",
                message.guild.name,
                message.channel.name,
            )
            return False

        try:
            # 9. Reply send
            send_started = time.perf_counter()
            self.runtime_state.update_turn_stage(trace, "reply_send")
            self.trace_event(trace, "reply_send_started", reply_chars=len(reply))
            sent_parts = await self.send_reply(message, reply)
            self.trace_event(
                trace,
                "reply_send_completed",
                duration_ms=int((time.perf_counter() - send_started) * 1000),
                reply_chars=len(reply),
                message_parts=sent_parts,
            )
            _forensic_event(
                trace,
                "reply_delivery",
                finalized_reply=reply,
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
            return True
        except Exception:
            self.trace_event(trace, "reply_send_completed", status="error")
            logger.exception(
                "Reply send failed in %s/%s after generation",
                message.guild.name,
                message.channel.name,
            )
            return False

    async def handle_message(self, message: discord.Message, *, bot_user) -> None:
        trace = TurnTrace.from_message(message)
        turn_started = time.perf_counter()
        replied = False
        self.runtime_state.begin_turn(trace, author_is_bot=message.author.bot)

        try:
            # 1. Turn intake / trace bootstrap
            self.trace_event(
                trace,
                "message_received",
                content_chars=len(message.content or ""),
                attachments=len(message.attachments),
                author_is_bot=message.author.bot,
            )
            _forensic_event(
                trace,
                "turn_input",
                raw_content=message.content,
                resolved_content=resolve_mentions(message.content, message.mentions),
                attachments=[
                    {
                        "filename": attachment.filename,
                        "content_type": attachment.content_type,
                        "size_bytes": attachment.size,
                    }
                    for attachment in message.attachments
                ],
                mentions=[mention.display_name for mention in message.mentions],
            )

            # 2. Registry refresh + human-readable ingress log
            self.background_tasks.create_task(
                asyncio.to_thread(self.registry.ensure_seen, message),
                name=f"registry.ensure_seen:{message.id}",
            )
            self._log_message_received(message)

            # 3. Bot messages short-circuit after cache + memory enqueue
            if message.author.bot:
                self.runtime_state.update_turn_stage(trace, "memory_enqueue")
                logger.debug("Bot message from %s — storing and skipping pipeline", message.author.display_name)
                self.cache.add(message)
                await self.memory_worker.enqueue(message)
                self.trace_event(trace, "memory_enqueued", source="bot_message")
                self.trace_event(
                    trace,
                    "turn_completed",
                    duration_ms=int((time.perf_counter() - turn_started) * 1000),
                    replied=False,
                    bot_message=True,
                )
                return

            # 4. Vision / attachment augmentation
            vision_started = time.perf_counter()
            self.runtime_state.update_turn_stage(trace, "vision")
            attachment_result = await self.describe_attachments(message)
            self.trace_event(
                trace,
                "vision_completed",
                duration_ms=int((time.perf_counter() - vision_started) * 1000),
                image_count=len(attachment_result.descriptions),
                fallback_images=attachment_result.fallback_count,
                fallback_reasons=attachment_result.fallback_reasons or None,
            )
            rag_query_text = self._add_message_to_cache(message, attachment_result.descriptions)
            _forensic_event(
                trace,
                "vision_artifacts",
                descriptions=attachment_result.descriptions,
                fallback_count=attachment_result.fallback_count,
                fallback_reasons=attachment_result.fallback_reasons or None,
                rag_query_text=rag_query_text,
            )

            # 5. Bouncer decision and deferred memory enqueue
            history = self.cache.get(message.guild.id, message.channel.id)
            bouncer_started = time.perf_counter()
            self.runtime_state.update_turn_stage(trace, "bouncer")
            bouncer_result = await self.llm.ask_bouncer(
                history.format(),
                bot_name=bot_user.display_name,
                trace=trace,
            )

            logger.info(
                "Bouncer → respond=%s tool=%s(%s) reason=%r",
                bouncer_result.should_respond,
                bouncer_result.recommended_tool or "none",
                bouncer_result.use_tool,
                getattr(bouncer_result, "reason", None),
            )
            self.runtime_state.set_last_bouncer_decision(
                trace_id=trace.trace_id,
                should_respond=bouncer_result.should_respond,
                use_tool=bouncer_result.use_tool,
                tool_name=bouncer_result.recommended_tool,
            )
            self.trace_event(
                trace,
                "bouncer_completed",
                duration_ms=int((time.perf_counter() - bouncer_started) * 1000),
                should_respond=bouncer_result.should_respond,
                use_tool=bouncer_result.use_tool,
                tool_name=bouncer_result.recommended_tool,
            )
            _forensic_event(
                trace,
                "bouncer_context",
                history_text=history.format(),
            )

            self.runtime_state.update_turn_stage(trace, "memory_enqueue")
            await self.memory_worker.enqueue(message, image_descriptions=attachment_result.descriptions)
            self.trace_event(trace, "memory_enqueued", source="user_message")

            # 6-9. Tool -> RAG -> brain -> reply handling - see _run_reply_pipeline()
            if bouncer_result.should_respond:
                async with message.channel.typing():
                    replied = await self._run_reply_pipeline(
                        message,
                        bot_user=bot_user,
                        trace=trace,
                        history=history,
                        rag_query_text=rag_query_text,
                        bouncer_result=bouncer_result,
                    )

            # 10. Turn completion
            self.runtime_state.update_turn_stage(trace, "turn_completed")
            self.trace_event(
                trace,
                "turn_completed",
                duration_ms=int((time.perf_counter() - turn_started) * 1000),
                replied=replied,
            )
        finally:
            self.runtime_state.end_turn(trace.trace_id)

    async def shutdown(self) -> None:
        await self.memory_worker.shutdown()


def build_pipeline(background_tasks, *, trace_event=_trace_event, runtime_state: RuntimeState | None = None) -> SandyPipeline:
    """Construct Sandy's pipeline with the default production dependencies."""
    runtime_state = runtime_state or RuntimeState()
    registry = Registry()
    cache = Last10(maxlen=10, registry=registry)
    llm = OllamaInterface()
    vector_memory = VectorMemory()

    db_dir = resolve_runtime_path(os.getenv("DB_DIR", "data/prod/"))
    recall_db_name = os.getenv("RECALL_DB_NAME", "recall.db")
    recall_db = ChatDatabase(str(db_dir / recall_db_name))
    recall_db.init_db()
    tools.init_recall_db(recall_db)

    memory = MemoryClient(
        db=recall_db,
        llm=llm,
        vector_memory=vector_memory,
    )
    memory_worker = MemoryWorker(memory.process_and_store, runtime_state=runtime_state)

    return SandyPipeline(
        background_tasks=background_tasks,
        registry=registry,
        cache=cache,
        llm=llm,
        vector_memory=vector_memory,
        recall_db=recall_db,
        memory=memory,
        memory_worker=memory_worker,
        runtime_state=runtime_state,
        tools_module=tools,
        trace_event=trace_event,
    )
