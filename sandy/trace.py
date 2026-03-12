"""Lightweight turn-tracing helpers for Sandy's message pipeline."""

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import discord


@dataclass(slots=True, frozen=True)
class TurnTrace:
    """Metadata that identifies one Discord message turn through the pipeline."""

    trace_id: str
    message_id: int
    guild_id: int
    guild_name: str
    channel_id: int
    channel_name: str
    author_id: int
    author_name: str
    started_at: float

    @classmethod
    def from_message(cls, message: discord.Message) -> "TurnTrace":
        return cls(
            trace_id=str(message.id),
            message_id=message.id,
            guild_id=message.guild.id,
            guild_name=message.guild.name,
            channel_id=message.channel.id,
            channel_name=message.channel.name,
            author_id=message.author.id,
            author_name=message.author.display_name,
            started_at=perf_counter(),
        )


def now_ms(started_at: float) -> int:
    """Return elapsed milliseconds from a perf_counter start timestamp."""
    return int((perf_counter() - started_at) * 1000)


def event_payload(
    trace: TurnTrace,
    stage: str,
    *,
    status: str = "ok",
    duration_ms: int | None = None,
    **fields: Any,
) -> dict[str, Any]:
    """Build a compact structured event payload for logs."""
    payload: dict[str, Any] = {
        "trace_id": trace.trace_id,
        "stage": stage,
        "status": status,
        "message_id": trace.message_id,
        "guild_id": trace.guild_id,
        "channel_id": trace.channel_id,
        "author_id": trace.author_id,
    }
    if duration_ms is not None:
        payload["duration_ms"] = duration_ms
    payload.update(fields)
    return payload
