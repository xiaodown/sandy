"""Bouncer context assembly and decision call."""

import time

import discord

from ..last10 import ChannelHistory
from ..logconf import get_logger
from ..trace import TurnTrace
from .attachments import build_cache_message
from .tracing import trace_event, forensic_event

logger = get_logger("sandy.bot")


def build_bouncer_context(
    history: ChannelHistory,
    message: discord.Message,
    descriptions: list[str],
) -> str:
    synthetic_latest = build_cache_message(message, descriptions)
    latest_line = ChannelHistory(
        [synthetic_latest],
        getattr(history, "registry", None),
    ).format()
    existing = history.format()
    if existing == "(no recent messages)":
        return latest_line
    return f"{existing}\n{latest_line}"


async def run_bouncer(
    llm,
    *,
    bouncer_context: str,
    bot_user,
    trace: TurnTrace,
    runtime_state,
):
    """Run the bouncer and return its result."""
    bouncer_started = time.perf_counter()
    runtime_state.update_turn_stage(trace, "bouncer")
    bouncer_result = await llm.ask_bouncer(
        bouncer_context,
        trace=trace,
    )

    logger.info(
        "Bouncer → respond=%s tool=%s(%s) reason=%r",
        bouncer_result.should_respond,
        bouncer_result.recommended_tool or "none",
        bouncer_result.use_tool,
        getattr(bouncer_result, "reason", None),
    )
    runtime_state.set_last_bouncer_decision(
        trace_id=trace.trace_id,
        should_respond=bouncer_result.should_respond,
        use_tool=bouncer_result.use_tool,
        tool_name=bouncer_result.recommended_tool,
    )
    trace_event(
        trace,
        "bouncer_completed",
        duration_ms=int((time.perf_counter() - bouncer_started) * 1000),
        should_respond=bouncer_result.should_respond,
        use_tool=bouncer_result.use_tool,
        tool_name=bouncer_result.recommended_tool,
    )
    forensic_event(
        trace,
        "bouncer_context",
        history_text=bouncer_context,
    )

    return bouncer_result
