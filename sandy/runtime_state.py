from dataclasses import dataclass
from threading import Lock
from time import time
from typing import Any

from .trace import TurnTrace


@dataclass(slots=True)
class ActiveTurn:
    trace_id: str
    message_id: int
    guild_name: str
    channel_name: str
    author_name: str
    stage: str
    status: str
    author_is_bot: bool
    started_at: float
    updated_at: float


@dataclass(slots=True)
class BouncerDecisionSnapshot:
    trace_id: str
    should_respond: bool
    use_tool: bool
    tool_name: str | None
    recorded_at: float


class RuntimeState:
    """Thread-safe snapshot state for the observability API."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._discord_connected = False
        self._discord_user: str | None = None
        self._discord_servers: list[str] = []
        self._active_turns: dict[str, ActiveTurn] = {}
        self._memory_queue_depth = 0
        self._memory_processing_message_id: int | None = None
        self._last_bouncer_decision: BouncerDecisionSnapshot | None = None

    def set_discord_connected(self, connected: bool, *, user_name: str | None = None) -> None:
        with self._lock:
            self._discord_connected = connected
            if user_name is not None:
                self._discord_user = user_name

    def set_discord_servers(self, server_names: list[str]) -> None:
        with self._lock:
            self._discord_servers = list(server_names)

    def begin_turn(self, trace: TurnTrace, *, author_is_bot: bool) -> None:
        now = time()
        with self._lock:
            self._active_turns[trace.trace_id] = ActiveTurn(
                trace_id=trace.trace_id,
                message_id=trace.message_id,
                guild_name=trace.guild_name,
                channel_name=trace.channel_name,
                author_name=trace.author_name,
                stage="message_received",
                status="ok",
                author_is_bot=author_is_bot,
                started_at=now,
                updated_at=now,
            )

    def update_turn_stage(self, trace: TurnTrace, stage: str, *, status: str = "ok") -> None:
        now = time()
        with self._lock:
            active = self._active_turns.get(trace.trace_id)
            if active is None:
                self._active_turns[trace.trace_id] = ActiveTurn(
                    trace_id=trace.trace_id,
                    message_id=trace.message_id,
                    guild_name=trace.guild_name,
                    channel_name=trace.channel_name,
                    author_name=trace.author_name,
                    stage=stage,
                    status=status,
                    author_is_bot=False,
                    started_at=now,
                    updated_at=now,
                )
                return
            active.stage = stage
            active.status = status
            active.updated_at = now

    def end_turn(self, trace_id: str) -> None:
        with self._lock:
            self._active_turns.pop(trace_id, None)

    def memory_enqueued(self) -> None:
        with self._lock:
            self._memory_queue_depth += 1

    def memory_processing_started(self, *, message_id: int | None) -> None:
        with self._lock:
            if self._memory_queue_depth > 0:
                self._memory_queue_depth -= 1
            self._memory_processing_message_id = message_id

    def memory_processing_finished(self, *, message_id: int | None) -> None:
        with self._lock:
            if self._memory_processing_message_id == message_id:
                self._memory_processing_message_id = None

    def set_last_bouncer_decision(
        self,
        *,
        trace_id: str,
        should_respond: bool,
        use_tool: bool,
        tool_name: str | None,
    ) -> None:
        with self._lock:
            self._last_bouncer_decision = BouncerDecisionSnapshot(
                trace_id=trace_id,
                should_respond=should_respond,
                use_tool=use_tool,
                tool_name=tool_name,
                recorded_at=time(),
            )

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            active_turns = sorted(
                (
                    {
                        "trace_id": turn.trace_id,
                        "message_id": turn.message_id,
                        "guild_name": turn.guild_name,
                        "channel_name": turn.channel_name,
                        "author_name": turn.author_name,
                        "stage": turn.stage,
                        "status": turn.status,
                        "author_is_bot": turn.author_is_bot,
                        "started_at": turn.started_at,
                        "updated_at": turn.updated_at,
                    }
                    for turn in self._active_turns.values()
                ),
                key=lambda turn: turn["updated_at"],
                reverse=True,
            )
            last_bouncer = None
            if self._last_bouncer_decision is not None:
                last_bouncer = {
                    "trace_id": self._last_bouncer_decision.trace_id,
                    "should_respond": self._last_bouncer_decision.should_respond,
                    "use_tool": self._last_bouncer_decision.use_tool,
                    "tool_name": self._last_bouncer_decision.tool_name,
                    "recorded_at": self._last_bouncer_decision.recorded_at,
                }
            return {
                "discord": {
                    "connected": self._discord_connected,
                    "user_name": self._discord_user,
                    "server_count": len(self._discord_servers),
                    "server_names": list(self._discord_servers),
                },
                "active_turns": active_turns,
                "memory_worker": {
                    "queue_depth": self._memory_queue_depth,
                    "processing_message_id": self._memory_processing_message_id,
                    "busy": self._memory_processing_message_id is not None,
                },
                "last_bouncer_decision": last_bouncer,
            }
