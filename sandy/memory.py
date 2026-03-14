"""
Memory client — persists Discord messages to Recall (SQLite) for long-term storage.

Also handles the tag+summarize pipeline via the LLM, and stores
embeddings in ChromaDB for semantic RAG retrieval.

Public API
----------
    from sandy.recall import ChatDatabase

    db = ChatDatabase("data/prod/recall.db")
    db.init_db()
    client = MemoryClient(db=db, llm=llm, vector_memory=vector_memory)

    # Full pipeline: tag → summarize → store in Recall + RAG
    await client.process_and_store(message)
"""

import asyncio
import logging
import os
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

import discord
from dotenv import load_dotenv

from .recall import ChatDatabase, ChatMessageCreate

if TYPE_CHECKING:
    from .llm import OllamaInterface
    from .vector_memory import VectorMemory
    from .last10 import Last10

load_dotenv()

logger = logging.getLogger(__name__)


class MemoryClient:
    """
    Stores Discord messages in Recall (SQLite) and ChromaDB (vector).

    Create one instance and reuse it for the lifetime of the bot.

    Usage:
        db = ChatDatabase("data/prod/recall.db")
        db.init_db()
        client = MemoryClient(db=db, llm=llm, vector_memory=vector_memory)
        await client.process_and_store(message)
    """

    #: Messages longer than this get passed through the summarizer before storage.
    SUMMARIZE_THRESHOLD = int(os.getenv('SUMMARIZE_THRESHOLD', 144))

    def __init__(
        self,
        db: ChatDatabase,
        llm: "Optional[OllamaInterface]" = None,
        vector_memory: "Optional[VectorMemory]" = None,
    ):
        self._db = db
        self._llm = llm
        self._vector_memory = vector_memory

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def process_and_store(self, message: discord.Message, image_descriptions: Optional[list[str]] = None) -> None:
        """Tag, optionally summarise, then persist one message.

        Intended to be run as a fire-and-forget background task after the
        bouncer/brain pipeline has finished for this message:

            asyncio.create_task(memory.process_and_store(message))

        The LLM lock in OllamaInterface ensures tagger/summarizer calls
        serialise correctly if two messages are processed concurrently.
        Both calls are optional — if either fails or llm is not set, the
        message is still stored without tags or a summary.

        image_descriptions — if the message had image attachments, pass the
        list of description strings here.  The descriptions are appended to
        the stored content so Recall and RAG embed the image context rather
        than empty string.  The real message object is still used for all
        metadata (author, channel, guild, id).
        """
        tags: list[str] = []
        summary: Optional[str] = None
        content_for_storage = self._build_content_for_storage(
            message,
            image_descriptions=image_descriptions,
        )

        if self._llm is not None and content_for_storage:
            try:
                tags = await self._llm.ask_tagger(content_for_storage)
                # Strip leading hyphen or emdash from each tag
                tags = [t.lstrip("-—") for t in tags]
            except Exception as exc:
                logger.error(
                    "Tagger failed for message %s in %s/%s: %s",
                    message.id,
                    message.guild.name,
                    message.channel.name,
                    exc,
                )
                tags = []

            if len(content_for_storage) > self.SUMMARIZE_THRESHOLD:
                try:
                    summary = await self._llm.ask_summarizer(content_for_storage)
                except Exception as exc:
                    logger.error(
                        "Summarizer failed for message %s in %s/%s: %s",
                        message.id,
                        message.guild.name,
                        message.channel.name,
                        exc,
                    )
                    summary = None

        recall_stored = False
        vector_stored = False
        vector_error: Exception | None = None

        try:
            vector_stored = await self._store_vector(
                message,
                content=content_for_storage,
            )
        except Exception as exc:
            vector_error = exc
            logger.error(
                "Vector store failed for message %s in %s/%s: %s",
                message.id,
                message.guild.name,
                message.channel.name,
                exc,
            )

        try:
            recall_stored = self._store_recall(
                message,
                tags=tags or None,
                summary=summary,
                content_override=content_for_storage or None,
            )
        except Exception as exc:
            logger.error(
                "Recall store failed for message %s in %s/%s: %s",
                message.id,
                message.guild.name,
                message.channel.name,
                exc,
            )

        logger.info(
            "Stored message from %s in %s/%s — tags=%r summary=%s vector=%s recall=%s",
            message.author.display_name,
            message.guild.name,
            message.channel.name,
            tags,
            "yes" if summary else "no",
            vector_stored,
            recall_stored,
        )

        if vector_error is not None:
            raise vector_error

    async def store_message(self, message: discord.Message) -> bool:
        """Store a Discord message in Recall without any tags or LLM processing.

        Use process_and_store() for the full tag+summarize+store pipeline.
        Returns True on success. Raises on any store error.
        """
        return self._store_recall(message, tags=None, summary=None)

    async def store_message_with_tags(
        self,
        message: discord.Message,
        tags: list[str],
        summary: Optional[str] = None,
    ) -> bool:
        """Store a Discord message in Recall with LLM-generated tags.

        tags     — list of normalized lowercase strings, e.g. ["game", "tarkov"]
        summary  — optional one-line LLM summary of the message
        """
        return self._store_recall(message, tags=tags, summary=summary)

    async def seed_cache(self, cache: "Last10", hours: int = 24) -> int:
        """Seed the in-memory rolling cache from Recall on bot startup.

        Queries Recall for the last ``hours`` hours of messages, groups them
        by (server_id, channel_id), and adds the most recent ``cache.maxlen``
        from each channel as SyntheticMessage objects.

        Safe to call once from on_ready.  Returns the total number of messages
        seeded (0 on any error, so a cold Recall is not fatal).
        """
        from .last10 import SyntheticMessage, _SyntheticAuthor, _SyntheticGuild, _SyntheticChannel

        try:
            # ChatDatabase methods are synchronous (sqlite3); run in a thread
            # to avoid blocking the event loop.
            rows = await asyncio.to_thread(
                self._db.get_messages,
                hours_ago=hours,
                limit=1000,
            )
        except Exception as exc:
            logger.error("seed_cache: failed to fetch from Recall — %s", exc)
            return 0

        # Recall returns newest-first (ORDER BY timestamp DESC).
        # Group by channel, preserving that order so we can slice cheaply.
        groups: dict[tuple[int, int], list] = {}
        for m in rows:
            key = (m.server_id, m.channel_id)
            groups.setdefault(key, []).append(m)

        maxlen = cache.maxlen
        count = 0
        for msgs in groups.values():
            # Take at most maxlen (already newest-first), then reverse so we
            # add them oldest-first — matching the deque's expected append order.
            recent = list(reversed(msgs[:maxlen]))
            for m in recent:
                created_at = m.timestamp
                if created_at.tzinfo is None:
                    created_at = created_at.replace(tzinfo=timezone.utc)
                sm = SyntheticMessage(
                    content=m.content,
                    created_at=created_at,
                    author=_SyntheticAuthor(
                        id=m.author_id,
                        display_name=m.author_name,
                    ),
                    guild=_SyntheticGuild(
                        id=m.server_id,
                        name=m.server_name,
                    ),
                    channel=_SyntheticChannel(
                        id=m.channel_id,
                        name=m.channel_name,
                    ),
                )
                cache.add(sm)
                count += 1

        logger.info(
            "seed_cache: seeded %d messages across %d channel(s) from the last %dh",
            count, len(groups), hours,
        )
        return count

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_content_for_storage(
        self,
        message: discord.Message,
        *,
        image_descriptions: Optional[list[str]] = None,
    ) -> str:
        """Build the content string used for both Recall storage and RAG."""
        content_for_storage = message.content or ""
        if image_descriptions:
            img_block = "  ".join(
                f"[Image {i}: {d}]" if len(image_descriptions) > 1 else f"[Image: {d}]"
                for i, d in enumerate(image_descriptions, 1)
            )
            content_for_storage = (
                f"{content_for_storage}  {img_block}".strip()
                if content_for_storage else img_block
            )
        return content_for_storage

    def _store_recall(
        self,
        message: discord.Message,
        tags: Optional[list[str]],
        summary: Optional[str],
        content_override: Optional[str] = None,
    ) -> bool:
        """Create a ChatMessageCreate and insert directly into Recall.

        content_override — when set, stored as the message content instead of
        message.content.  Used for image messages where we want the description,
        not an empty string, in Recall.

        Returns True on success. Raises on failure so the caller can decide policy.
        """
        msg = ChatMessageCreate(
            discord_message_id=message.id,
            author_id=message.author.id,
            author_name=message.author.display_name,
            channel_id=message.channel.id,
            channel_name=message.channel.name,
            server_id=message.guild.id,
            server_name=message.guild.name,
            content=content_override or message.content or "(no text content)",
            timestamp=message.created_at,
            tags=tags,
            summary=summary,
        )
        self._db.create_message(msg)
        return True

    async def _store_vector(
        self,
        message: discord.Message,
        *,
        content: str,
    ) -> bool:
        """Store one message in vector memory. Returns True when a write occurred."""
        if self._vector_memory is None or not content:
            return False

        await self._vector_memory.add_message(
            message_id=str(message.id),
            content=content,
            author_name=message.author.display_name,
            server_id=message.guild.id,
            timestamp=message.created_at,
        )
        logger.info(
            "Sent message to RAG from %s in %s/%s",
            message.author.display_name,
            message.guild.name,
            message.channel.name,
        )
        return True
