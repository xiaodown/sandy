"""
Sandy's message pipeline package.

Public API:
    build_pipeline(...)  — construct the pipeline with production dependencies
    SandyPipeline        — the pipeline owner class
    MemoryWorker         — deferred memory queue worker
    AttachmentProcessingResult — used by tests
"""

import os

from ..last10 import Last10
from ..llm import OllamaInterface
from ..memory import MemoryClient
from ..paths import resolve_runtime_path
from ..recall import ChatDatabase
from ..registry import Registry
from ..runtime_state import RuntimeState
from ..vector_memory import VectorMemory
from ..voice import VoiceManager
from .. import tools

from .attachments import AttachmentProcessingResult
from .memory_worker import MemoryWorker
from .orchestrator import SandyPipeline
from .tracing import trace_event as _default_trace_event


def build_pipeline(
    background_tasks,
    *,
    trace_event=_default_trace_event,
    runtime_state: RuntimeState | None = None,
) -> SandyPipeline:
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
    voice = VoiceManager(
        registry=registry,
        runtime_state=runtime_state,
        llm=llm,
        vector_memory=vector_memory,
        background_tasks=background_tasks,
    )

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
