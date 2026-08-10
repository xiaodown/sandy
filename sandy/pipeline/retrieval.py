"""RAG retrieval step: vector memory query with bypass rules."""

import time
from dataclasses import dataclass

from ..logconf import get_logger
from ..trace import TurnTrace
from .tracing import trace_event, forensic_event

logger = get_logger("sandy.bot")

_RAG_BYPASS_TOOLS: frozenset[str] = frozenset({"steam_browse"})


@dataclass(frozen=True, slots=True)
class RagConfig:
    enabled: bool = True
    n_results: int = 8
    max_chars: int = 4000
    max_doc_chars: int = 800
    scope: str = "server"


async def run_retrieval(
    vector_memory,
    *,
    rag_query_text: str,
    server_id: int,
    channel_id: int,
    ollama_history: list[dict],
    recommended_tool: str | None,
    rag_config: RagConfig,
    trace: TurnTrace,
    runtime_state,
) -> str:
    """Run RAG retrieval and return the context string (may be empty)."""
    if not rag_config.enabled:
        runtime_state.update_turn_stage(trace, "retrieval_skipped")
        trace_event(
            trace,
            "retrieval_completed",
            status="skipped",
            skipped_reason="rag_disabled",
            context_chars=0,
        )
        forensic_event(
            trace,
            "retrieval",
            query_text=rag_query_text,
            rag_context="",
            ollama_history=ollama_history,
            skipped_reason="rag_disabled",
        )
        return ""

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
        channel_id=channel_id,
        n_results=rag_config.n_results,
        scope=rag_config.scope,
        max_chars=rag_config.max_chars,
        max_doc_chars=rag_config.max_doc_chars,
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
