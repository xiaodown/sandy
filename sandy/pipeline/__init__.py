"""
Sandy's message pipeline package.

Public API:
    build_pipeline(...)  — construct the pipeline with production dependencies
    SandyPipeline        — the pipeline owner class
    MemoryWorker         — deferred memory queue worker
    AttachmentProcessingResult — used by tests
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from ..last10 import Last10
from ..llm import OllamaInterface
from ..memory import MemoryClient
from ..paths import resolve_runtime_path
from ..recall import ChatDatabase
from ..registry import Registry
from ..runtime_state import RuntimeState
from ..vector_memory import VectorMemory
from ..voice import VoiceManager, configure_voice
from .. import tools

from .attachments import AttachmentProcessingResult
from .memory_worker import MemoryWorker
from .orchestrator import SandyPipeline
from .tracing import trace_event as _default_trace_event

if TYPE_CHECKING:
    from ..config import SandyConfig


def build_pipeline(
    background_tasks,
    *,
    trace_event=_default_trace_event,
    runtime_state: RuntimeState | None = None,
    config: SandyConfig | None = None,
) -> SandyPipeline:
    """Construct Sandy's pipeline with the default production dependencies.

    If *config* is provided, it's threaded through to all sub-components.
    Otherwise, each component falls back to reading env vars directly
    (backward-compatible during migration / in tests).
    """
    runtime_state = runtime_state or RuntimeState()

    if config is not None:
        registry = Registry(db_path=str(config.storage.server_db_path))
        llm = OllamaInterface(config=config.llm)
        vector_memory = VectorMemory(
            db_dir=config.storage.db_dir,
            embed_model=config.storage.embed_model,
            max_distance=config.storage.vector_max_distance,
        )
        recall_db = ChatDatabase(str(config.storage.recall_db_path))
        recall_db.init_db()
        tools.init_recall_db(recall_db)
        tools.init_tools_config(
            searxng_base_url=config.search.searxng_base_url,
            steam_cache_ttl=config.search.steam_cache_ttl_seconds,
            registry=registry,
        )
        configure_voice(
            capture_dir=config.voice.capture_dir,
            preroll_ms=config.voice.preroll_ms,
            stitch_gap_seconds=config.voice.stitch_gap_seconds,
            stitch_release_seconds=config.voice.stitch_release_seconds,
            history_maxlen=config.voice.history_maxlen,
            idle_auto_leave_seconds=config.voice.idle_auto_leave_seconds,
            force_release_seconds=config.voice.force_release_seconds,
            reply_max_words=config.voice.reply_max_words,
            reply_max_chars=config.voice.reply_max_chars,
            reply_max_sentences=config.voice.reply_max_sentences,
        )
        memory = MemoryClient(
            db=recall_db,
            llm=llm,
            vector_memory=vector_memory,
            summarize_threshold=config.storage.summarize_threshold,
        )
        voice = VoiceManager(
            registry=registry,
            runtime_state=runtime_state,
            llm=llm,
            vector_memory=vector_memory,
            background_tasks=background_tasks,
            voice_config=config.voice,
        )
    else:
        # Legacy path — no config object, read env vars directly.
        registry = Registry()
        llm = OllamaInterface()
        vector_memory = VectorMemory()
        db_dir = resolve_runtime_path(os.getenv("DB_DIR", "data/prod/"))
        recall_db_name = os.getenv("RECALL_DB_NAME", "recall.db")
        recall_db = ChatDatabase(str(db_dir / recall_db_name))
        recall_db.init_db()
        tools.init_recall_db(recall_db)
        tools.init_tools_config(registry=registry)
        memory = MemoryClient(
            db=recall_db,
            llm=llm,
            vector_memory=vector_memory,
        )
        voice = VoiceManager(
            registry=registry,
            runtime_state=runtime_state,
            llm=llm,
            vector_memory=vector_memory,
            background_tasks=background_tasks,
        )

    cache = Last10(maxlen=10, registry=registry)
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
        voice=voice,
        tools_module=tools,
        trace_event=trace_event,
    )


__all__ = [
    "AttachmentProcessingResult",
    "MemoryWorker",
    "SandyPipeline",
    "build_pipeline",
]
