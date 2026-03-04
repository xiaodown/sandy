"""
A discord interface for my AI chatbot
It should:
1.) come online and connect to the servers
2.) be able to receive and parse chat messages
3.) be able to send these messages elsewhere for decisions to be made about what to do with them
4.) be able to send meta info to trigger a "typing..." rich presence
5.) be able to send messages to a channel
6.) gracefully handle disconnects and reconnects
7.) all the above are required to be async

Probably more to come.

"""


import asyncio
import io
import os
import logging

import discord
from dotenv import load_dotenv
from PIL import Image

from logconf import get_logger
from recall import ChatDatabase
from registry import Registry
from last10 import Last10, resolve_mentions, SyntheticMessage, _SyntheticAuthor, _SyntheticGuild, _SyntheticChannel
from memory import MemoryClient
from ollama_interface import OllamaInterface
from vector_memory import VectorMemory
import tools

load_dotenv()

logger = get_logger("discord_handler")

intents = discord.Intents.default()
intents.members = True
intents.message_content = True
bot = discord.Client(intents=intents)
DISCORD_API_KEY = os.getenv("DISCORD_API_KEY")

# Suppress default discord logging
logging.getLogger("discord").propagate = False

registry = Registry()
cache = Last10(maxlen=10, registry=registry)
llm = OllamaInterface()
vector_memory = VectorMemory()

# Recall database — path built from DB_DIR + RECALL_DB_NAME so the test/prod
# switch is controlled by a single env var.
_db_dir = os.getenv("DB_DIR", "data/")
_recall_db_name = os.getenv("RECALL_DB_NAME", "recall.db")
recall_db = ChatDatabase(os.path.join(_db_dir, _recall_db_name))
recall_db.init_db()

# Give tools.py a reference to the shared Recall DB
tools.init_recall_db(recall_db)

memory = MemoryClient(db=recall_db, llm=llm, vector_memory=vector_memory)

# Guard so cache seeding only runs once, even if on_ready fires on reconnect.
_cache_seeded = False

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

# ---------------------------------------------------------------------------
# Tool result framing for brain injection
# ---------------------------------------------------------------------------

_MEMORY_TOOLS: frozenset[str] = frozenset({
    "recall_recent", "recall_from_user", "recall_by_topic", "search_memories",
})


def _format_tool_context(tool_name: str, result: str) -> str:
    """Frame a tool result for injection into the brain's system prompt.

    The framing varies by tool type so Sandy talks about remembering vs.
    looking something up, matching natural personality.
    """
    if tool_name == "search_web":
        return f"## You just looked this up online\n{result}"
    elif tool_name == "get_current_time":
        return f"## You just checked the time\n{result}"
    elif tool_name in _MEMORY_TOOLS:
        return f"## You just recalled this from memory\n{result}"
    else:
        return f"## Additional context\n{result}"


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
    global _cache_seeded
    PREWARM_MODEL = os.getenv("PREWARM_MODEL")
    PREWARM_MODEL_NAME = os.getenv("PREWARM_MODEL_NAME")

    logger.info("Logged in as %s (%s)", bot.user.name, bot.user.id)
    if not _cache_seeded:
        seeded = await memory.seed_cache(cache)
        logger.info("Cache seeded with %d message(s) from Recall", seeded)
        _cache_seeded = True
        if PREWARM_MODEL == "True":
            logger.info("Beginning pre-warming of model...")
            if await llm.warm_model(PREWARM_MODEL_NAME):
                logger.info("Pre-warming model %s complete", PREWARM_MODEL_NAME)
            else:
                logger.warning("Pre-warming of model %s failed", PREWARM_MODEL_NAME)
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

    # Update the registry (server / channel / user lookup cache)
    # Note: bot's own messages are included — they're part of the conversation context
    # registry.ensure_seen() uses sqlite3 (blocking I/O), so run it in a thread.
    asyncio.create_task(asyncio.to_thread(registry.ensure_seen, message))

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
        asyncio.create_task(memory.process_and_store(message))
        return

    # --- Image attachment processing -----------------------------------
    # Describe any image attachments before adding to cache, so the
    # bouncer and brain both see the image content in context.
    # The original message is always passed to process_and_store — Recall
    # stores what was actually said, not our augmented version.
    image_descriptions = await _describe_attachments(message)
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
    bouncer_result = await llm.ask_bouncer(history.format(), bot_name=bot.user.display_name)

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
                else:
                    tool_result = await tools.dispatch(
                        bouncer_result.recommended_tool,
                        bouncer_result.tool_parameters or {},
                        server_id=message.guild.id,
                        server_name=message.guild.name,
                    )
                    tool_context = _format_tool_context(
                        bouncer_result.recommended_tool, tool_result,
                    )

            # --- RAG query ---------------------------------------------
            ollama_history = history.to_ollama_messages(bot.user.id)
            rag_context = await vector_memory.query(
                rag_query_text,
                server_id=message.guild.id,
            )

            # --- Brain response ----------------------------------------
            reply = await llm.ask_brain(
                ollama_history,
                bot_name=bot.user.display_name,
                server_name=message.guild.name,
                channel_name=message.channel.name,
                rag_context=rag_context,
                tool_context=tool_context,
            )

            if reply:
                await message.channel.send(reply)
                logger.info("Brain replied in %s/%s (%d chars)",
                            message.guild.name, message.channel.name, len(reply))
            else:
                logger.warning("Brain returned None for message in %s/%s — not sending",
                               message.guild.name, message.channel.name)

    # --- Memory (fire-and-forget after pipeline) -----------------------
    # Now that the LLM is free, tag and store the message in the background.
    # process_and_store will await the lock, so if another message's pipeline
    # fires concurrently the tagger/summarizer will queue behind it cleanly.
    # Pass image_descriptions so Recall and RAG store the description text
    # rather than an empty content field.
    asyncio.create_task(memory.process_and_store(message, image_descriptions=image_descriptions))


if __name__ == "__main__":
    async def main():
        try:
            await bot.start(DISCORD_API_KEY)
        except discord.LoginFailure as e:
            print(f"Failed to log in: {e}")
        except discord.DiscordException as e:
            print(f"A Discord-related error occurred: {e}")
        finally:
            await bot.close()

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("^c caught - stand by, shutting down cleanly...")
        pass  # Ctrl+C — clean shutdown already handled in main()'s finally block