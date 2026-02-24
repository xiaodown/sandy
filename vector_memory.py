"""
Semantic vector memory backed by ChromaDB + ollama embeddings.

Every message Sandy sees is embedded and stored here alongside its server_id.
Before each brain call, the triggering message is embedded and the most
semantically similar past messages are retrieved and injected into Sandy's
system prompt as "background awareness" — fuzzy, passive memory that
complements the precise on-demand Recall tools.

Server isolation
----------------
Every document is stored with {"server_id": <int>} in its metadata.
All queries include a where={"server_id": ...} filter — results are always
scoped to the current guild and cannot leak across servers.

Storage path
------------
Reads DB_DIR from .env (default "data/") and stores Chroma files in
<DB_DIR>/chroma/.  No separate server or daemon required — ChromaDB runs
embedded in-process, writes to disk on mutation, exactly like SQLite.

Embedding model
---------------
Reads EMBED_MODEL from .env (default "mxbai-embed-large").  The model must
be pulled in ollama before the bot starts.  Embeddings are generated via the
async ollama Python client.
"""

import logging
import os
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import chromadb
import ollama
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

_PACIFIC      = ZoneInfo("America/Los_Angeles")
_EMBED_MODEL  = os.getenv("EMBED_MODEL", "mxbai-embed-large")
_CHROMA_PATH  = os.path.join(os.getenv("DB_DIR", "data").rstrip("/\\"), "chroma")
_COLLECTION   = "sandy_messages"

# Cosine distance threshold — results above this value are discarded as
# too dissimilar.  Cosine distance ranges 0 (identical) → 2 (opposite);
# 0.6 is a fairly generous cutoff.  Tune downward if irrelevant memories
# appear too often.
_MAX_DISTANCE = float(os.getenv("VECTOR_MAX_DISTANCE", "0.6"))


class VectorMemory:
    """
    Persistent semantic memory backed by ChromaDB + ollama embeddings.

    Create one instance and reuse it for the lifetime of the bot:

        vm = VectorMemory()

        # Store a message (called from process_and_store):
        await vm.add_message(
            message_id="123456789",
            content="let's play Tarkov tonight",
            author_name="Dave",
            server_id=987654321,
            timestamp=datetime.utcnow(),
        )

        # Retrieve relevant memories (called before ask_brain):
        block = await vm.query("video games", server_id=987654321)
        # → "[2026-02-20 14:32 PST] <Dave>: let's play Tarkov tonight"
    """

    def __init__(self) -> None:
        os.makedirs(_CHROMA_PATH, exist_ok=True)
        self._chroma = chromadb.PersistentClient(path=_CHROMA_PATH)
        self._collection = self._chroma.get_or_create_collection(
            name=_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
        self._embed_client = ollama.AsyncClient()
        logger.info(
            "VectorMemory ready (path=%r, collection=%r, docs=%d)",
            _CHROMA_PATH, _COLLECTION, self._collection.count(),
        )

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    async def add_message(
        self,
        message_id: str,
        content: str,
        author_name: str,
        server_id: int,
        timestamp: datetime,
    ) -> None:
        """Embed and upsert one message into the vector store.

        message_id  — unique string key (Discord snowflake as str recommended)
        content     — raw message text; empty/whitespace-only messages are skipped
        author_name — display name at time of storage
        server_id   — Discord guild ID; stored in metadata for isolation filtering
        timestamp   — message creation time (tz-aware UTC preferred)
        """
        if not content or not content.strip():
            return
        # Skip pure placeholder text stored by the Recall server for attachments.
        if content.strip() == "(no text content)":
            return
        try:
            resp = await self._embed_client.embed(model=_EMBED_MODEL, input=content)
            embedding = resp.embeddings[0]
            ts_str = timestamp.isoformat() if timestamp else ""
            self._collection.upsert(
                ids=[message_id],
                embeddings=[embedding],
                documents=[content],
                metadatas=[{
                    "author_name": author_name,
                    "server_id":   server_id,
                    "timestamp":   ts_str,
                }],
            )
            logger.debug(
                "VectorMemory.add_message stored id=%s server=%d author=%r",
                message_id, server_id, author_name,
            )
        except Exception as exc:
            logger.error("VectorMemory.add_message failed (id=%s): %s", message_id, exc)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    async def query(
        self,
        text: str,
        server_id: int,
        n_results: int = 5,
    ) -> str:
        """Return a formatted block of semantically similar past messages.

        text      — query text (typically the most recent user message)
        server_id — only messages from this guild are returned
        n_results — maximum number of results to include

        Returns a newline-joined block ready for injection into a system
        prompt, or an empty string if nothing relevant is found or on error.
        """
        if not text or not text.strip():
            return ""
        try:
            total = self._collection.count()
            if total == 0:
                return ""
            # Cap n_results at total doc count to avoid ChromaDB errors when
            # the collection is smaller than the requested result count.
            n = min(n_results, total)
            resp = await self._embed_client.embed(model=_EMBED_MODEL, input=text)
            embedding = resp.embeddings[0]
            results = self._collection.query(
                query_embeddings=[embedding],
                n_results=n,
                where={"server_id": server_id},
                include=["documents", "metadatas", "distances"],
            )
            docs      = results.get("documents",  [[]])[0]
            metas     = results.get("metadatas",  [[]])[0]
            distances = results.get("distances",  [[]])[0]

            lines = []
            for doc, meta, dist in zip(docs, metas, distances):
                if dist > _MAX_DISTANCE:
                    continue
                author = meta.get("author_name", "?")
                ts_raw = meta.get("timestamp", "")
                try:
                    dt = datetime.fromisoformat(ts_raw)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    ts = dt.astimezone(_PACIFIC).strftime("%Y-%m-%d %H:%M %Z")
                except Exception:
                    ts = ts_raw or "?"
                lines.append(f"[{ts}] <{author}>: {doc}")

            if lines:
                logger.debug("VectorMemory.query → %d result(s) for server %d", len(lines), server_id)
            else:
                logger.debug(
                    "VectorMemory.query → 0 result(s) within threshold (%.2f) for server %d",
                    _MAX_DISTANCE, server_id,
                )
            return "\n".join(lines)
        except Exception as exc:
            logger.error("VectorMemory.query failed: %s", exc)
            return ""
