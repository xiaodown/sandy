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
import os
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import TYPE_CHECKING

import discord
from dotenv import load_dotenv

from .logconf import get_logger
from .recall import (
    ChatDatabase,
    ChatMessageCreate,
    DeferredMessageCreate,
    DeferredMessageResponse,
)

if TYPE_CHECKING:
    from .llm import OllamaInterface
    from .vector_memory import VectorMemory
    from .last10 import Last10
    from .pipeline.attachments import AttachmentProcessingResult

load_dotenv()

logger = get_logger(__name__)


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
        llm: "OllamaInterface | None" = None,
        vector_memory: "VectorMemory | None" = None,
        summarize_threshold: int | None = None,
    ):
        self._db = db
        self._llm = llm
        self._vector_memory = vector_memory
        self._deferred_drain_lock = asyncio.Lock()
        if summarize_threshold is not None:
            self.SUMMARIZE_THRESHOLD = summarize_threshold

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def process_and_store(self, message: discord.Message, image_descriptions: list[str] | None = None) -> None:
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
        result = await self._process_payload(
            discord_message_id=message.id,
            author_id=message.author.id,
            author_name=message.author.display_name,
            channel_id=message.channel.id,
            channel_name=message.channel.name,
            server_id=message.guild.id,
            server_name=message.guild.name,
            base_content=message.content or "",
            timestamp=message.created_at,
            image_descriptions=image_descriptions,
        )
        logger.info(
            "Stored message from %s in %s/%s — tags=%r summary=%s vector=%s recall=%s",
            message.author.display_name,
            message.guild.name,
            message.channel.name,
            result["tags"],
            "yes" if result["summary"] else "no",
            result["vector_stored"],
            result["recall_stored"],
        )
        if result["vector_error"] is not None:
            raise result["vector_error"]

    async def enqueue_deferred_message(self, message: discord.Message) -> int:
        """Store a text message in the deferred queue while VC is active."""
        payload = DeferredMessageCreate(
            discord_message_id=message.id,
            author_id=message.author.id,
            author_name=message.author.display_name,
            channel_id=message.channel.id,
            channel_name=message.channel.name,
            server_id=message.guild.id,
            server_name=message.guild.name,
            content=message.content or "",
            timestamp=message.created_at,
            attachment_payload=self._build_attachment_payload(message),
        )
        queue_id = await asyncio.to_thread(self._db.enqueue_deferred_message, payload)
        logger.info(
            "Deferred text memory for message %s in %s/%s while VC is active",
            message.id,
            message.guild.name,
            message.channel.name,
        )
        return queue_id

    async def drain_deferred_messages(self, *, limit: int = 100) -> int:
        """Drain queued text messages into Recall and vector memory."""
        if self._llm is None:
            logger.warning("Deferred-message drain skipped: no LLM configured")
            return 0
        async with self._deferred_drain_lock:
            processed = 0
            attempted_ids: set[int] = set()
            while True:
                rows = await asyncio.to_thread(self._db.get_deferred_messages, limit=limit)
                rows = [row for row in rows if row.id not in attempted_ids]
                if not rows:
                    break
                for row in rows:
                    attempted_ids.add(row.id)
                    try:
                        image_descriptions = await self._describe_deferred_attachments(row)
                        if not (row.content or "").strip() and not image_descriptions:
                            await asyncio.to_thread(self._db.delete_deferred_message, row.id)
                            logger.info(
                                "Dropped deferred message %s because image content was no longer retrievable",
                                row.discord_message_id,
                            )
                            continue
                        result = await self._process_payload(
                            discord_message_id=row.discord_message_id,
                            author_id=row.author_id,
                            author_name=row.author_name,
                            channel_id=row.channel_id,
                            channel_name=row.channel_name,
                            server_id=row.server_id,
                            server_name=row.server_name,
                            base_content=row.content,
                            timestamp=row.timestamp,
                            image_descriptions=image_descriptions,
                            allow_existing_recall=True,
                        )
                        if result["vector_error"] is not None:
                            raise result["vector_error"]
                        await asyncio.to_thread(self._db.delete_deferred_message, row.id)
                        processed += 1
                    except Exception as exc:
                        await asyncio.to_thread(
                            self._db.record_deferred_message_failure,
                            row.id,
                            str(exc),
                        )
                        logger.error(
                            "Deferred message drain failed for discord_message_id=%s in %s/%s: %s",
                            row.discord_message_id,
                            row.server_name,
                            row.channel_name,
                            exc,
                        )
            if processed:
                logger.info("Deferred-message drain completed: processed=%d", processed)
            return processed

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
        summary: str | None = None,
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
        content: str,
        *,
        image_descriptions: list[str] | None = None,
    ) -> str:
        """Build the content string used for both Recall storage and RAG."""
        content_for_storage = content or ""
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

    def _build_attachment_payload(self, message: discord.Message) -> list[dict] | None:
        payload: list[dict] = []
        for attachment in message.attachments:
            payload.append({
                "filename": attachment.filename,
                "content_type": attachment.content_type,
                "size_bytes": attachment.size,
                "url": getattr(attachment, "url", None),
                "proxy_url": getattr(attachment, "proxy_url", None),
                "width": getattr(attachment, "width", None),
                "height": getattr(attachment, "height", None),
            })
        return payload or None

    async def _describe_deferred_attachments(
        self,
        row: DeferredMessageResponse,
    ) -> list[str] | None:
        if not row.attachment_payload or self._llm is None:
            return None
        from .pipeline.attachments import describe_deferred_attachment_payload

        result = await describe_deferred_attachment_payload(row.attachment_payload, self._llm)
        return result.descriptions or None

    async def _process_payload(
        self,
        *,
        discord_message_id: int,
        author_id: int,
        author_name: str,
        channel_id: int,
        channel_name: str,
        server_id: int,
        server_name: str,
        base_content: str,
        timestamp: datetime,
        image_descriptions: list[str] | None = None,
        allow_existing_recall: bool = False,
    ) -> dict[str, object]:
        tags: list[str] = []
        summary: str | None = None
        content_for_storage = self._build_content_for_storage(
            base_content,
            image_descriptions=image_descriptions,
        )

        if self._llm is not None and content_for_storage:
            tags, summary = await self._generate_tags_and_summary(
                discord_message_id=discord_message_id,
                server_name=server_name,
                channel_name=channel_name,
                content_for_storage=content_for_storage,
            )

        recall_stored = False
        vector_stored = False
        vector_error: Exception | None = None

        payload = SimpleNamespace(
            id=discord_message_id,
            created_at=timestamp,
            author=SimpleNamespace(id=author_id, display_name=author_name),
            channel=SimpleNamespace(id=channel_id, name=channel_name),
            guild=SimpleNamespace(id=server_id, name=server_name),
            content=base_content,
        )

        try:
            vector_stored = await self._store_vector(
                payload,
                content=content_for_storage,
            )
        except Exception as exc:
            vector_error = exc
            logger.error(
                "Vector store failed for message %s in %s/%s: %s",
                discord_message_id,
                server_name,
                channel_name,
                exc,
            )

        try:
            existing = None
            if allow_existing_recall:
                existing = self._db.get_message_by_discord_id(discord_message_id)
            if existing is None:
                recall_stored = self._store_recall(
                    payload,
                    tags=tags or None,
                    summary=summary,
                    content_override=content_for_storage or None,
                )
            else:
                recall_stored = True
        except Exception as exc:
            logger.error(
                "Recall store failed for message %s in %s/%s: %s",
                discord_message_id,
                server_name,
                channel_name,
                exc,
            )

        return {
            "tags": tags,
            "summary": summary,
            "vector_stored": vector_stored,
            "recall_stored": recall_stored,
            "vector_error": vector_error,
        }

    async def _generate_tags_and_summary(
        self,
        *,
        discord_message_id: int,
        server_name: str,
        channel_name: str,
        content_for_storage: str,
    ) -> tuple[list[str], str | None]:
        tags: list[str] = []
        summary: str | None = None
        try:
            tags = await self._llm.ask_tagger(content_for_storage)
            tags = [t.lstrip("-—") for t in tags]
        except Exception as exc:
            logger.error(
                "Tagger failed for message %s in %s/%s: %s",
                discord_message_id,
                server_name,
                channel_name,
                exc,
            )
            tags = []

        if len(content_for_storage) > self.SUMMARIZE_THRESHOLD:
            try:
                summary = await self._llm.ask_summarizer(content_for_storage)
            except Exception as exc:
                logger.error(
                    "Summarizer failed for message %s in %s/%s: %s",
                    discord_message_id,
                    server_name,
                    channel_name,
                    exc,
                )
                summary = None
        return tags, summary

    def _store_recall(
        self,
        message: discord.Message,
        tags: list[str] | None,
        summary: str | None,
        content_override: str | None = None,
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

        stored = await self._vector_memory.add_message(
            message_id=str(message.id),
            content=content,
            author_name=message.author.display_name,
            server_id=message.guild.id,
            timestamp=message.created_at,
        )
        if stored:
            logger.info(
                "Sent message to RAG from %s in %s/%s",
                message.author.display_name,
                message.guild.name,
                message.channel.name,
            )
        return bool(stored)
