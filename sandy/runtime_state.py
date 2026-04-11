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


@dataclass(slots=True)
class VoiceSnapshot:
    active: bool
    status: str
    stage: str | None
    session_id: str | None
    guild_id: int | None
    guild_name: str | None
    channel_id: int | None
    channel_name: str | None
    participant_names: list[str]
    session_started_at: float | None
    current_trace_id: str | None
    last_transcript: str | None
    last_reply: str | None
    last_error: str | None
    updated_at: float


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
        self._voice = VoiceSnapshot(
            active=False,
            status="idle",
            stage=None,
            session_id=None,
            guild_id=None,
            guild_name=None,
            channel_id=None,
            channel_name=None,
            participant_names=[],
            session_started_at=None,
            current_trace_id=None,
            last_transcript=None,
            last_reply=None,
            last_error=None,
            updated_at=time(),
        )

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

    def set_voice_state(
        self,
        *,
        active: bool,
        status: str,
        stage: str | None = None,
        session_id: str | None = None,
        guild_id: int | None = None,
        guild_name: str | None = None,
        channel_id: int | None = None,
        channel_name: str | None = None,
        participant_names: list[str] | None = None,
        session_started_at: float | None = None,
        current_trace_id: str | None = None,
        last_transcript: str | None = None,
        last_reply: str | None = None,
        last_error: str | None = None,
    ) -> None:
        with self._lock:
            previous = self._voice
            if not active:
                self._voice = VoiceSnapshot(
                    active=False,
                    status=status,
                    stage=None,
                    session_id=None,
                    guild_id=None,
                    guild_name=None,
                    channel_id=None,
                    channel_name=None,
                    participant_names=[],
                    session_started_at=None,
                    current_trace_id=None,
                    last_transcript=None,
                    last_reply=None,
                    last_error=None,
                    updated_at=time(),
                )
                return
            self._voice = VoiceSnapshot(
                active=active,
                status=status,
                stage=stage if stage is not None else previous.stage,
                session_id=session_id if session_id is not None else previous.session_id,
                guild_id=guild_id if guild_id is not None else previous.guild_id,
                guild_name=guild_name if guild_name is not None else previous.guild_name,
                channel_id=channel_id if channel_id is not None else previous.channel_id,
                channel_name=channel_name if channel_name is not None else previous.channel_name,
                participant_names=list(participant_names) if participant_names is not None else list(previous.participant_names),
                session_started_at=session_started_at if session_started_at is not None else previous.session_started_at,
                current_trace_id=current_trace_id if current_trace_id is not None else previous.current_trace_id,
                last_transcript=last_transcript if last_transcript is not None else previous.last_transcript,
                last_reply=last_reply if last_reply is not None else previous.last_reply,
                last_error=last_error if last_error is not None else previous.last_error,
                updated_at=time(),
            )

    def update_voice_stage(
        self,
        *,
        stage: str,
        status: str | None = None,
        current_trace_id: str | None = None,
        last_transcript: str | None = None,
        last_reply: str | None = None,
        last_error: str | None = None,
    ) -> None:
        with self._lock:
            previous = self._voice
            self._voice = VoiceSnapshot(
                active=previous.active,
                status=status if status is not None else previous.status,
                stage=stage,
                session_id=previous.session_id,
                guild_id=previous.guild_id,
                guild_name=previous.guild_name,
                channel_id=previous.channel_id,
                channel_name=previous.channel_name,
                participant_names=list(previous.participant_names),
                session_started_at=previous.session_started_at,
                current_trace_id=current_trace_id if current_trace_id is not None else previous.current_trace_id,
                last_transcript=last_transcript if last_transcript is not None else previous.last_transcript,
                last_reply=last_reply if last_reply is not None else previous.last_reply,
                last_error=last_error if last_error is not None else previous.last_error,
                updated_at=time(),
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
                "voice": {
                    "active": self._voice.active,
                    "status": self._voice.status,
                    "stage": self._voice.stage,
                    "session_id": self._voice.session_id,
                    "guild_id": self._voice.guild_id,
                    "guild_name": self._voice.guild_name,
                    "channel_id": self._voice.channel_id,
                    "channel_name": self._voice.channel_name,
                    "participant_names": list(self._voice.participant_names),
                    "session_started_at": self._voice.session_started_at,
                    "current_trace_id": self._voice.current_trace_id,
                    "last_transcript": self._voice.last_transcript,
                    "last_reply": self._voice.last_reply,
                    "last_error": self._voice.last_error,
                    "updated_at": self._voice.updated_at,
                },
                "active_turns": active_turns,
                "memory_worker": {
                    "queue_depth": self._memory_queue_depth,
                    "processing_message_id": self._memory_processing_message_id,
                    "busy": self._memory_processing_message_id is not None,
                },
                "last_bouncer_decision": last_bouncer,
            }
