from __future__ import annotations

from time import perf_counter

from ..logconf import emit_forensic_record, get_logger
from ..trace import VoiceTurnTrace, event_payload, forensic_payload
from .models import CompletedVoiceTurn, VoiceSession

logger = get_logger("sandy.voice")


def trace_event(
    trace: VoiceTurnTrace,
    stage: str,
    *,
    status: str = "ok",
    duration_ms: int | None = None,
    **fields: object,
) -> None:
    payload = event_payload(
        trace,
        stage,
        status=status,
        duration_ms=duration_ms,
        **fields,
    )
    logger.info(
        "TRACE %s",
        payload,
        extra={"event_payload": payload, "log_to_console": False},
    )


def forensic_event(trace: VoiceTurnTrace, artifact: str, **fields: object) -> None:
    emit_forensic_record(
        logger,
        f"FORENSIC {artifact}",
        forensic_payload(trace, artifact, **fields),
    )


def build_voice_trace(
    session: VoiceSession,
    *,
    completed_turns: list[CompletedVoiceTurn],
) -> VoiceTurnTrace:
    if len({turn.speaker_id for turn in completed_turns}) == 1:
        author_id = completed_turns[-1].speaker_id
        author_name = completed_turns[-1].speaker_name
    else:
        author_id = 0
        author_name = "multiple speakers"

    return VoiceTurnTrace(
        trace_id=f"voice:{session.session_id}:{session.response_counter}",
        message_id=None,
        guild_id=session.guild_id,
        guild_name=session.guild_name,
        channel_id=session.channel_id,
        channel_name=session.channel_name,
        author_id=author_id,
        author_name=author_name,
        started_at=perf_counter(),
        session_id=session.session_id,
    )
