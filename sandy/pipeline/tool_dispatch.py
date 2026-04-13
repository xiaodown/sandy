"""Tool dispatch step: execute bouncer-recommended tools and frame results."""

import time

import discord

from ..logconf import get_logger
from ..trace import TurnTrace
from .tracing import trace_event, forensic_event

logger = get_logger("sandy.bot")

_MEMORY_TOOLS: frozenset[str] = frozenset({
    "recall_recent", "recall_from_user", "recall_by_topic", "search_memories",
})


def format_tool_context(tool_name: str, result: str) -> str:
    if tool_name == "search_web":
        return f"## You just looked this up online\n{result}"
    if tool_name == "steam_browse":
        return f"## You just checked Steam\n{result}"
    if tool_name == "get_current_time":
        return f"## You just checked the time\n{result}"
    if tool_name == "dice_roll":
        return f"## You just rolled some dice\n{result}"
    if tool_name in _MEMORY_TOOLS:
        return f"## You just recalled this from memory\n{result}"
    return f"## Additional context\n{result}"


async def run_tool_dispatch(
    tools_module,
    *,
    message: discord.Message,
    bouncer_result,
    trace: TurnTrace,
    runtime_state,
) -> str | None:
    """Execute the bouncer-recommended tool and return formatted context (or None)."""
    if not (bouncer_result.use_tool and bouncer_result.recommended_tool):
        return None

    if bouncer_result.recommended_tool not in tools_module.KNOWN_TOOLS:
        logger.warning(
            "Bouncer recommended unknown tool %r — ignoring",
            bouncer_result.recommended_tool,
        )
        trace_event(
            trace,
            "tool_completed",
            status="ignored",
            tool_name=bouncer_result.recommended_tool,
        )
        return None

    logger.debug(
        "Dispatching tool %s with params %s",
        bouncer_result.recommended_tool,
        bouncer_result.tool_parameters or {},
    )
    tool_started = time.perf_counter()
    runtime_state.update_turn_stage(trace, "tool_started")
    trace_event(
        trace,
        "tool_started",
        tool_name=bouncer_result.recommended_tool,
    )
    tool_result = await tools_module.dispatch(
        bouncer_result.recommended_tool,
        bouncer_result.tool_parameters or {},
        server_id=message.guild.id,
        server_name=message.guild.name,
    )
    trace_event(
        trace,
        "tool_completed",
        duration_ms=int((time.perf_counter() - tool_started) * 1000),
        tool_name=bouncer_result.recommended_tool,
        result_chars=len(tool_result or ""),
    )
    forensic_event(
        trace,
        "tool_call",
        tool_name=bouncer_result.recommended_tool,
        arguments=bouncer_result.tool_parameters or {},
        result=tool_result,
        tool_context=format_tool_context(
            bouncer_result.recommended_tool,
            tool_result,
        ),
    )
    return format_tool_context(
        bouncer_result.recommended_tool,
        tool_result,
    )
