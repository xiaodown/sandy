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
Reads DB_DIR from .env (default "data/prod/") and stores Chroma files in
<DB_DIR>/chroma/.  No separate server or daemon required — ChromaDB runs
embedded in-process, writes to disk on mutation, exactly like SQLite.

Embedding model
---------------
Reads EMBED_MODEL from .env (default "mxbai-embed-large").  The model must
be pulled in ollama before the bot starts.  Embeddings are generated via the
async ollama Python client.
"""

from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import chromadb
import ollama

from .logconf import get_logger
from .paths import resolve_runtime_path

logger = get_logger(__name__)

_PACIFIC      = ZoneInfo("America/Los_Angeles")
_COLLECTION   = "sandy_messages"


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

    def __init__(
        self,
        *,
        db_dir: str | None = None,
        embed_model: str | None = None,
        max_distance: float | None = None,
        rag_n_results: int | None = None,
        rag_max_chars: int | None = None,
        rag_max_doc_chars: int | None = None,
        rag_scope: str | None = None,
    ) -> None:
        if (
            db_dir is None
            or embed_model is None
            or max_distance is None
            or rag_n_results is None
            or rag_max_chars is None
            or rag_max_doc_chars is None
            or rag_scope is None
        ):
            from .config import SandyConfig

            storage_cfg = SandyConfig.from_env().storage
            if db_dir is None:
                db_dir = storage_cfg.db_dir
            if embed_model is None:
                embed_model = storage_cfg.embed_model
            if max_distance is None:
                max_distance = storage_cfg.vector_max_distance
            if rag_n_results is None:
                rag_n_results = storage_cfg.rag_n_results
            if rag_max_chars is None:
                rag_max_chars = storage_cfg.rag_max_chars
            if rag_max_doc_chars is None:
                rag_max_doc_chars = storage_cfg.rag_max_doc_chars
            if rag_scope is None:
                rag_scope = storage_cfg.rag_scope

        _db_dir = resolve_runtime_path(db_dir)
        chroma_path = _db_dir / "chroma"
        chroma_path.mkdir(parents=True, exist_ok=True)
        try:
            self._chroma = chromadb.PersistentClient(path=str(chroma_path))
        except Exception as exc:
            logger.error("VectorMemory startup failed for path %r: %s", str(chroma_path), exc)
            raise
        self._collection = self._chroma.get_or_create_collection(
            name=_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
        self._embed_model = embed_model
        self._max_distance = max_distance
        self._rag_n_results = rag_n_results
        self._rag_max_chars = rag_max_chars
        self._rag_max_doc_chars = rag_max_doc_chars
        self._rag_scope = rag_scope
        self._embed_client = ollama.AsyncClient()
        logger.info(
            "VectorMemory ready (path=%r, collection=%r, docs=%d)",
            str(chroma_path), _COLLECTION, self._collection.count(),
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
        channel_id: int | None = None,
    ) -> bool:
        """Embed and upsert one message into the vector store.

        message_id  — unique string key (Discord snowflake as str recommended)
        content     — raw message text; empty/whitespace-only messages are skipped
        author_name — display name at time of storage
        server_id   — Discord guild ID; stored in metadata for isolation filtering
        timestamp   — message creation time (tz-aware UTC preferred)
        channel_id  — optional Discord channel ID for channel-scoped retrieval
        """
        if not content or not content.strip():
            return False
        # Skip pure placeholder text stored by the Recall server for attachments.
        if content.strip() == "(no text content)":
            return False
        resp = await self._embed_client.embed(model=self._embed_model, input=content)
        embedding = resp.embeddings[0]
        ts_str = timestamp.isoformat() if timestamp else ""
        metadata = {
            "author_name": author_name,
            "server_id":   server_id,
            "timestamp":   ts_str,
        }
        if channel_id is not None:
            metadata["channel_id"] = channel_id
        self._collection.upsert(
            ids=[message_id],
            embeddings=[embedding],
            documents=[content],
            metadatas=[metadata],
        )
        logger.debug(
            "VectorMemory.add_message stored id=%s server=%d author=%r",
            message_id, server_id, author_name,
        )
        return True

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    async def query(
        self,
        text: str,
        server_id: int,
        n_results: int | None = None,
        channel_id: int | None = None,
        scope: str | None = None,
        max_chars: int | None = None,
        max_doc_chars: int | None = None,
    ) -> str:
        """Return a formatted block of semantically similar past messages.

        text      — query text (typically the most recent user message)
        server_id — only messages from this guild are returned
        n_results — maximum number of results to include
        channel_id — channel to filter to when scope="channel"
        scope — "server" or "channel"; server scope is the default
        max_chars — maximum size of the returned formatted block
        max_doc_chars — maximum characters from any single stored document

        Returns a newline-joined block ready for injection into a system
        prompt, or an empty string if nothing relevant is found or on error.
        """
        if not text or not text.strip():
            return ""
        try:
            if n_results is None:
                n_results = getattr(self, "_rag_n_results", 8)
            if scope is None:
                scope = getattr(self, "_rag_scope", "server")
            if scope not in {"server", "channel"}:
                scope = "server"
            if max_chars is None:
                max_chars = getattr(self, "_rag_max_chars", 4000)
            if max_doc_chars is None:
                max_doc_chars = getattr(self, "_rag_max_doc_chars", 800)
            total = self._collection.count()
            if total == 0:
                return ""
            # Cap n_results at total doc count to avoid ChromaDB errors when
            # the collection is smaller than the requested result count.
            n = min(n_results, total)
            resp = await self._embed_client.embed(model=self._embed_model, input=text)
            embedding = resp.embeddings[0]
            where: dict = {"server_id": server_id}
            if scope == "channel" and channel_id is not None:
                where = {"$and": [{"server_id": server_id}, {"channel_id": channel_id}]}
            results = self._collection.query(
                query_embeddings=[embedding],
                n_results=n,
                where=where,
                include=["documents", "metadatas", "distances"],
            )
            docs      = results.get("documents",  [[]])[0]
            metas     = results.get("metadatas",  [[]])[0]
            distances = results.get("distances",  [[]])[0]

            lines = []
            for doc, meta, dist in zip(docs, metas, distances):
                if dist > self._max_distance:
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
                doc_text = str(doc)
                if max_doc_chars is not None and max_doc_chars > 0 and len(doc_text) > max_doc_chars:
                    doc_text = doc_text[:max_doc_chars].rstrip() + "..."
                line = f"[{ts}] <{author}>: {doc_text}"
                if max_chars is not None and max_chars > 0:
                    current_chars = sum(len(existing) for existing in lines) + max(0, len(lines) - 1)
                    separator_chars = 1 if lines else 0
                    if current_chars + separator_chars + len(line) > max_chars:
                        remaining = max_chars - current_chars - separator_chars
                        if remaining >= 80:
                            lines.append(line[: max(0, remaining - 3)].rstrip() + "...")
                        break
                lines.append(line)

            if lines:
                logger.debug("VectorMemory.query → %d result(s) for server %d", len(lines), server_id)
            else:
                logger.debug(
                    "VectorMemory.query → 0 result(s) within threshold (%.2f) for server %d",
                    self._max_distance, server_id,
                )
            return "\n".join(lines)
        except Exception as exc:
            logger.error("VectorMemory.query failed: %s", exc)
            return ""

    def delete_message(self, message_id: str) -> bool:
        """Delete one vector-memory document by its Discord message snowflake."""
        try:
            existing = self._collection.get(ids=[message_id], include=[])
            if not existing.get("ids"):
                return False
            self._collection.delete(ids=[message_id])
            logger.info("VectorMemory.delete_message removed id=%s", message_id)
            return True
        except Exception as exc:
            logger.error("VectorMemory.delete_message failed (id=%s): %s", message_id, exc)
            raise
