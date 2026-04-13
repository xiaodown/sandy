"""RAG retrieval step: vector memory query with bypass rules."""

import time

from ..logconf import get_logger
from ..trace import TurnTrace
from .tracing import trace_event, forensic_event

logger = get_logger("sandy.bot")

_RAG_BYPASS_TOOLS: frozenset[str] = frozenset({"steam_browse"})


async def run_retrieval(
    vector_memory,
    *,
    rag_query_text: str,
    server_id: int,
    ollama_history: list[dict],
    recommended_tool: str | None,
    trace: TurnTrace,
    runtime_state,
) -> str:
    """Run RAG retrieval and return the context string (may be empty)."""
    if recommended_tool in _RAG_BYPASS_TOOLS:
        runtime_state.update_turn_stage(trace, "retrieval_skipped")
        trace_event(
            trace,
            "retrieval_completed",
            status="skipped",
            skipped_reason=f"tool:{recommended_tool}",
            context_chars=0,
        )
        forensic_event(
            trace,
            "retrieval",
            query_text=rag_query_text,
            rag_context="",
            ollama_history=ollama_history,
            skipped_reason=f"tool:{recommended_tool}",
        )
        return ""

    retrieval_started = time.perf_counter()
    runtime_state.update_turn_stage(trace, "retrieval")
    rag_context = await vector_memory.query(
        rag_query_text,
        server_id=server_id,
    )
    trace_event(
        trace,
        "retrieval_completed",
        duration_ms=int((time.perf_counter() - retrieval_started) * 1000),
        context_chars=len(rag_context or ""),
    )
    forensic_event(
        trace,
        "retrieval",
        query_text=rag_query_text,
        rag_context=rag_context,
        ollama_history=ollama_history,
    )
    return rag_context
