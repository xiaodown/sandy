"""
Recall client — sends Discord messages to the Recall API for long-term storage.

Recall host/port are read from the root .env (RECALL_HOST / RECALL_PORT)
so the bot and the API can run on different machines if
needed.

Public API
----------
    client = MemoryClient()

    # Store without tags (use this until the LLM tagger is wired up)
    await client.store_message(message)

    # Store with tags + optional summary (call this from the LLM tagger later)
    await client.store_message_with_tags(message, tags=["game", "tarkov"], summary="...")
"""

import logging
import os
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

import httpx
import discord
from dotenv import load_dotenv

if TYPE_CHECKING:
    from ollama_interface import OllamaInterface
    from vector_memory import VectorMemory
    from last10 import Last10

load_dotenv()

logger = logging.getLogger(__name__)

RECALL_BASE_URL = (
    f"http://{os.getenv('RECALL_HOST', '127.0.0.1')}"
    f":{os.getenv('RECALL_PORT', '8000')}"
)


class MemoryClient:
    """
    Async client for storing Discord messages in the Recall API.

    Uses a persistent httpx.AsyncClient for connection pooling. Create one
    instance and reuse it for the lifetime of the bot.

    Usage:
        client = MemoryClient()
        await client.store_message(message)
        # later, from the LLM tagger:
        await client.store_message_with_tags(message, tags=["fun"], summary="...")
    """

    #: Messages longer than this get passed through the summarizer before storage.
    # Cast to int: os.getenv() returns str when the var is set, which breaks the
    # len() > threshold comparison at runtime.
    SUMMARIZE_THRESHOLD = int(os.getenv('SUMMARIZE_THRESHOLD', 144))

    def __init__(
        self,
        base_url: str = RECALL_BASE_URL,
        timeout: float = 10.0,
        llm: "Optional[OllamaInterface]" = None,
        vector_memory: "Optional[VectorMemory]" = None,
    ):
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._client = httpx.AsyncClient(timeout=self._timeout)
        self._llm = llm
        self._vector_memory = vector_memory

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def process_and_store(self, message: discord.Message) -> None:
        """Tag, optionally summarise, then persist one message.

        Intended to be run as a fire-and-forget background task after the
        bouncer/brain pipeline has finished for this message:

            asyncio.create_task(memory.process_and_store(message))

        The LLM lock in OllamaInterface ensures tagger/summarizer calls
        serialise correctly if two messages are processed concurrently.
        Both calls are optional — if either fails or llm is not set, the
        message is still stored without tags or a summary.
        """
        tags: list[str] = []
        summary: Optional[str] = None

        if self._llm is not None and message.content:
            tags = await self._llm.ask_tagger(message.content)
            # Strip leading hyphen or emdash from each tag
            tags = [t.lstrip("-—") for t in tags]

            if len(message.content) > self.SUMMARIZE_THRESHOLD:
                summary = await self._llm.ask_summarizer(message.content)

        stored = await self._post(message, tags=tags or None, summary=summary)

        # Embed and store in the vector memory for semantic RAG retrieval.
        # message.id is always a real Discord snowflake here — process_and_store
        # is only ever called with genuine discord.Message objects.
        if self._vector_memory is not None and message.content:
            await self._vector_memory.add_message(
                message_id  = str(message.id),
                content     = message.content,
                author_name = message.author.display_name,
                server_id   = message.guild.id,
                timestamp   = message.created_at,
            )

        logger.info(
            "Stored message from %s in %s/%s — tags=%r summary=%s stored=%s",
            message.author.display_name,
            message.guild.name,
            message.channel.name,
            tags,
            "yes" if summary else "no",
            stored,
        )

    async def store_message(self, message: discord.Message) -> bool:
        """Store a Discord message in Recall without any tags or LLM processing.

        Use process_and_store() for the full tag+summarize+store pipeline.
        Returns True on success, False on any error.
        """
        return await self._post(message, tags=None, summary=None)

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
        return await self._post(message, tags=tags, summary=summary)

    async def seed_cache(self, cache: "Last10", hours: int = 24) -> int:
        """Seed the in-memory rolling cache from Recall on bot startup.

        Queries Recall for the last ``hours`` hours of messages, groups them
        by (server_id, channel_id), and adds the most recent ``cache.maxlen``
        from each channel as SyntheticMessage objects.

        Safe to call once from on_ready.  Returns the total number of messages
        seeded (0 on any error, so a cold Recall is not fatal).
        """
        from last10 import SyntheticMessage, _SyntheticAuthor, _SyntheticGuild, _SyntheticChannel

        try:
            response = await self._client.get(
                f"{self._base_url}/messages/",
                params={"hours_ago": hours, "limit": 1000},
            )
            response.raise_for_status()
            raw: list[dict] = response.json()
        except Exception as exc:
            logger.error("seed_cache: failed to fetch from Recall — %s", exc)
            return 0

        # Recall returns newest-first (ORDER BY timestamp DESC).
        # Group by channel, preserving that order so we can slice cheaply.
        groups: dict[tuple[int, int], list[dict]] = {}
        for m in raw:
            key = (m["server_id"], m["channel_id"])
            groups.setdefault(key, []).append(m)

        maxlen = cache.maxlen
        count = 0
        for msgs in groups.values():
            # Take at most maxlen (already newest-first), then reverse so we
            # add them oldest-first — matching the deque's expected append order.
            recent = list(reversed(msgs[:maxlen]))
            for m in recent:
                ts = m["timestamp"]
                created_at = datetime.fromisoformat(ts)
                if created_at.tzinfo is None:
                    created_at = created_at.replace(tzinfo=timezone.utc)
                sm = SyntheticMessage(
                    content=m["content"],
                    created_at=created_at,
                    author=_SyntheticAuthor(
                        id=m["author_id"],
                        display_name=m["author_name"],
                    ),
                    guild=_SyntheticGuild(
                        id=m["server_id"],
                        name=m["server_name"],
                    ),
                    channel=_SyntheticChannel(
                        id=m["channel_id"],
                        name=m["channel_name"],
                    ),
                )
                cache.add(sm)
                count += 1

        logger.info(
            "seed_cache: seeded %d messages across %d channel(s) from the last %dh",
            count, len(groups), hours,
        )
        return count

    async def close(self) -> None:
        """Cleanly close the underlying HTTP client. Call on bot shutdown."""
        await self._client.aclose()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _post(
        self,
        message: discord.Message,
        tags: Optional[list[str]],
        summary: Optional[str],
    ) -> bool:
        """Build the payload and POST to /messages/. Returns True on 200/201."""
        payload = {
            "author_id":    message.author.id,
            "author_name":  message.author.display_name,
            "channel_id":   message.channel.id,
            "channel_name": message.channel.name,
            "server_id":    message.guild.id,
            "server_name":  message.guild.name,
            "content":      message.content or "(no text content)",
            "timestamp":    message.created_at.isoformat(),
        }
        if tags is not None:
            payload["tags"] = tags
        if summary is not None:
            payload["summary"] = summary

        try:
            response = await self._client.post(
                f"{self._base_url}/messages/",
                json=payload,
            )
            response.raise_for_status()
            return True
        except httpx.ConnectError:
            logger.warning(
                "Recall server unreachable at %s — message not stored (channel %s)",
                self._base_url,
                message.channel.id,
            )
        except httpx.TimeoutException:
            logger.warning(
                "Recall server timed out — message not stored (channel %s)",
                message.channel.id,
            )
        except httpx.HTTPStatusError as e:
            logger.error(
                "Recall returned %s for message in channel %s: %s",
                e.response.status_code,
                message.channel.id,
                e.response.text,
            )
        except Exception as e:
            logger.exception("Unexpected error storing message in Recall: %s", e)

        return False
