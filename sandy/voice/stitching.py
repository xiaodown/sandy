from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from time import time
from typing import TYPE_CHECKING

from ..logconf import get_logger
from .capture import CaptureJob
from .history import VoiceHistoryEntry
from .models import (
    CompletedVoiceTurn,
    PendingSpeakerTurn,
    VoiceSession,
    _VOICE_FORCE_RELEASE_SECONDS,
    _VOICE_STITCH_GAP_SECONDS,
    _VOICE_STITCH_RELEASE_SECONDS,
)
from .stt import TranscriptResult

if TYPE_CHECKING:
    from .manager import VoiceManager

logger = get_logger("sandy.voice")


async def handle_transcript(manager: VoiceManager, job: CaptureJob, result: TranscriptResult) -> None:
    session = manager._session
    if session is None or session.guild_id != job.guild_id:
        return
    text = result.text.strip()
    speaker_id = job.speaker_id
    if speaker_id is not None:
        session.pending_stt_counts[speaker_id] = max(session.pending_stt_counts.get(speaker_id, 1) - 1, 0)
    if not text:
        if speaker_id is not None:
            maybe_schedule_release(manager, session, speaker_id)
        return

    session.last_activity_at = time()
    manager._arm_idle_timer(session)
    if speaker_id is None:
        # No stable speaker id; treat as immediate one-off turn.
        await emit_completed_turn(
            manager,
            session,
            completed_turn=CompletedVoiceTurn(
                speaker_id=0,
                speaker_name=job.speaker_label,
                text=text,
                started_at=job.started_at,
                ended_at=job.ended_at,
                fragment_count=1,
                total_audio_seconds=job.duration_seconds,
                total_stt_elapsed_seconds=result.elapsed_seconds,
                transcripts=[text],
            ),
        )
        return

    pending = session.pending_by_speaker.get(speaker_id)
    if pending is not None and (job.started_at - pending.ended_at) <= _VOICE_STITCH_GAP_SECONDS:
        pending.text = f"{pending.text} {text}".strip()
        pending.ended_at = job.ended_at
        pending.fragment_count += 1
        pending.total_audio_seconds += job.duration_seconds
        pending.total_stt_elapsed_seconds += result.elapsed_seconds
        pending.transcripts.append(text)
        if pending.release_task is not None:
            pending.release_task.cancel()
        if pending.force_release_task is not None:
            pending.force_release_task.cancel()
    else:
        if pending is not None:
            await release_pending_turn(manager, session, speaker_id)
        pending = PendingSpeakerTurn(
            speaker_id=speaker_id,
            speaker_name=job.speaker_label,
            text=text,
            started_at=job.started_at,
            ended_at=job.ended_at,
            fragment_count=1,
            total_audio_seconds=job.duration_seconds,
            total_stt_elapsed_seconds=result.elapsed_seconds,
            transcripts=[text],
        )
        session.pending_by_speaker[speaker_id] = pending

    maybe_schedule_release(manager, session, speaker_id)
    arm_force_release(manager, session, speaker_id)


def maybe_schedule_release(manager: VoiceManager, session: VoiceSession, speaker_id: int) -> None:
    pending = session.pending_by_speaker.get(speaker_id)
    if pending is None:
        return
    if session.pending_stt_counts.get(speaker_id, 0) > 0:
        return
    if speaker_id in session.active_speakers:
        return
    if pending.release_task is not None and not pending.release_task.done():
        pending.release_task.cancel()
    logger.info(
        "Voice release scheduled: speaker_id=%s in %.2fs",
        speaker_id,
        _VOICE_STITCH_RELEASE_SECONDS,
    )
    pending.release_task = asyncio.create_task(
        release_after_delay(manager, session.session_id, speaker_id),
        name=f"voice-release:{speaker_id}",
    )


def arm_force_release(manager: VoiceManager, session: VoiceSession, speaker_id: int) -> None:
    pending = session.pending_by_speaker.get(speaker_id)
    if pending is None:
        return
    if pending.force_release_task is not None and not pending.force_release_task.done():
        pending.force_release_task.cancel()
    pending.force_release_task = asyncio.create_task(
        force_release_after_delay(manager, session.session_id, speaker_id),
        name=f"voice-force-release:{speaker_id}",
    )


async def release_after_delay(manager: VoiceManager, session_id: str, speaker_id: int) -> None:
    await asyncio.sleep(_VOICE_STITCH_RELEASE_SECONDS)
    session = manager._session
    if session is None or session.session_id != session_id:
        return
    if session.pending_stt_counts.get(speaker_id, 0) > 0 or speaker_id in session.active_speakers:
        maybe_schedule_release(manager, session, speaker_id)
        return
    await release_pending_turn(manager, session, speaker_id)


async def force_release_after_delay(manager: VoiceManager, session_id: str, speaker_id: int) -> None:
    await asyncio.sleep(_VOICE_FORCE_RELEASE_SECONDS)
    session = manager._session
    if session is None or session.session_id != session_id:
        return
    pending = session.pending_by_speaker.get(speaker_id)
    if pending is None:
        return
    logger.warning(
        "Voice force-release triggered: speaker_id=%s active=%s pending_stt=%s",
        speaker_id,
        speaker_id in session.active_speakers,
        session.pending_stt_counts.get(speaker_id, 0),
    )
    session.active_speakers.discard(speaker_id)
    await release_pending_turn(manager, session, speaker_id)


async def release_pending_turn(manager: VoiceManager, session: VoiceSession, speaker_id: int) -> None:
    pending = session.pending_by_speaker.pop(speaker_id, None)
    if pending is None:
        return
    if pending.release_task is not None and not pending.release_task.done():
        pending.release_task.cancel()
    if pending.force_release_task is not None and not pending.force_release_task.done():
        pending.force_release_task.cancel()
    logger.info(
        "Voice turn released: speaker=%s text=%r",
        pending.speaker_name,
        pending.text,
    )
    await emit_completed_turn(
        manager,
        session,
        completed_turn=CompletedVoiceTurn(
            speaker_id=pending.speaker_id,
            speaker_name=pending.speaker_name,
            text=pending.text,
            started_at=pending.started_at,
            ended_at=pending.ended_at,
            fragment_count=max(1, pending.fragment_count),
            total_audio_seconds=pending.total_audio_seconds,
            total_stt_elapsed_seconds=pending.total_stt_elapsed_seconds,
            transcripts=list(pending.transcripts),
        ),
    )


async def emit_completed_turn(
    manager: VoiceManager,
    session: VoiceSession,
    *,
    completed_turn: CompletedVoiceTurn,
) -> None:
    entry = VoiceHistoryEntry(
        speaker_id=completed_turn.speaker_id,
        speaker_name=completed_turn.speaker_name,
        text=completed_turn.text,
        created_at=datetime.now(UTC),
        is_bot=False,
    )
    session.history.add(entry)
    session.pending_response_turns.append(completed_turn)
    session.pending_response_needed = True
    logger.info(
        "Voice turn appended: speaker=%s text=%r",
        completed_turn.speaker_name,
        completed_turn.text,
    )
    if session.response_task is None or session.response_task.done():
        logger.info(
            "Voice response task create: session_id=%s existing_done=%s pending_response_needed=%s",
            session.session_id,
            session.response_task.done() if session.response_task is not None else None,
            session.pending_response_needed,
        )
        session.response_task = manager._create_task(
            manager._respond_to_session(session.session_id),
            name=f"voice-respond:{session.session_id}",
        )
    manager._create_task(
        manager._store_voice_memory(
            session,
            message_id=(
                f"voice-human:{session.session_id}:{session.response_counter}:"
                f"{completed_turn.speaker_id}:{int(time() * 1000)}"
            ),
            author_name=completed_turn.speaker_name,
            text=completed_turn.text,
        ),
        name=f"voice-store-human:{session.session_id}",
    )
