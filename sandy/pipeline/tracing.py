"""Shared trace and forensic helpers for pipeline stages."""

from ..logconf import emit_forensic_record, get_logger
from ..trace import TurnTrace, event_payload, forensic_payload

logger = get_logger("sandy.bot")


def trace_event(
    trace: TurnTrace,
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


def forensic_event(trace: TurnTrace, artifact: str, **fields: object) -> None:
    emit_forensic_record(
        logger,
        f"FORENSIC {artifact}",
        forensic_payload(trace, artifact, **fields),
    )
