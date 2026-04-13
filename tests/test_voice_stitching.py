"""Tests for sandy.voice.stitching — fragment merging and release scheduling."""

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from sandy.runtime_state import RuntimeState
from sandy.voice import VoiceManager
from sandy.voice.capture import CaptureJob
from sandy.voice.models import PendingSpeakerTurn, _VOICE_STITCH_GAP_SECONDS
from sandy.voice.stt import TranscriptResult
from sandy.voice.stitching import (
    emit_completed_turn,
    handle_transcript,
    maybe_schedule_release,
    release_pending_turn,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_manager(monkeypatch) -> VoiceManager:
    """Create a VoiceManager with silenced logger and mock deps."""
    import sandy.voice.manager as vm
    import sandy.voice.stitching as st

    for mod in (vm, st):
        for method_name in ("info", "warning", "error", "exception", "debug"):
            monkeypatch.setattr(mod.logger, method_name, lambda *a, **kw: None)

    manager = VoiceManager(
        registry=SimpleNamespace(is_voice_admin=lambda **_: True),
        runtime_state=RuntimeState(),
        llm=SimpleNamespace(ask_brain=AsyncMock(return_value=SimpleNamespace(content="ok", done_reason="stop"))),
        vector_memory=SimpleNamespace(query=AsyncMock(return_value=""), add_message=AsyncMock(return_value=True)),
    )
    manager._warm_voice_models = AsyncMock()
    manager._respond_to_session = AsyncMock()
    manager._store_voice_memory = AsyncMock()
    return manager


async def _join_manager(manager: VoiceManager, monkeypatch) -> None:
    """Join a voice channel so manager has an active session."""
    from tests.test_voice_manager import DummyAuthor, DummyGuild, DummyMessage, DummyVoiceChannel

    voice_channel = DummyVoiceChannel(
        99, "ops war room",
        members=[SimpleNamespace(id=10, display_name="alice")],
    )
    guild = DummyGuild(1, "Guild", [voice_channel])
    msg = DummyMessage(
        guild=guild,
        author=DummyAuthor(user_id=10, display_name="alice", voice_channel=voice_channel),
        content="!join",
    )
    await manager.handle_text_command(msg, bot_user=SimpleNamespace(id=999, display_name="Sandy"))


def _capture_job(*, speaker_id=10, speaker_label="alice", started_at=1.0, ended_at=1.5, duration=0.5) -> CaptureJob:
    return CaptureJob(
        guild_id=1,
        channel_id=99,
        path=Path("/tmp/test.wav"),
        speaker_id=speaker_id,
        speaker_label=speaker_label,
        ssrc=1234,
        started_at=started_at,
        ended_at=ended_at,
        duration_seconds=duration,
        packet_count=12,
        saved_at=ended_at + 0.1,
    )


def _transcript(text="hello", elapsed=0.2) -> TranscriptResult:
    return TranscriptResult(
        text=text,
        language="en",
        language_probability=0.99,
        elapsed_seconds=elapsed,
        device="cpu",
        compute_type="int8",
    )


# ── handle_transcript: fragment stitching ────────────────────────────────────

@pytest.mark.asyncio
async def test_stitch_merges_fragments_within_gap(monkeypatch) -> None:
    """Two transcripts within STITCH_GAP_SECONDS get merged into one pending turn."""
    manager = _make_manager(monkeypatch)
    await _join_manager(manager, monkeypatch)
    session = manager.active_session

    t0 = 1.0
    gap = _VOICE_STITCH_GAP_SECONDS * 0.5  # well within gap

    job1 = _capture_job(started_at=t0, ended_at=t0 + 0.4, duration=0.4)
    job2 = _capture_job(started_at=t0 + 0.4, ended_at=t0 + 0.4 + gap, duration=gap)

    await handle_transcript(manager, job1, _transcript("first part"))
    await handle_transcript(manager, job2, _transcript("second part"))

    pending = session.pending_by_speaker.get(10)
    assert pending is not None
    assert pending.text == "first part second part"
    assert pending.fragment_count == 2
    assert pending.transcripts == ["first part", "second part"]


@pytest.mark.asyncio
async def test_stitch_releases_old_when_gap_exceeded(monkeypatch) -> None:
    """When gap between fragments exceeds threshold, old turn is released."""
    manager = _make_manager(monkeypatch)
    await _join_manager(manager, monkeypatch)
    session = manager.active_session

    t0 = 1.0
    gap = _VOICE_STITCH_GAP_SECONDS + 0.5  # exceeds gap

    job1 = _capture_job(started_at=t0, ended_at=t0 + 0.4, duration=0.4)
    job2 = _capture_job(started_at=t0 + 0.4 + gap, ended_at=t0 + 0.4 + gap + 0.3, duration=0.3)

    await handle_transcript(manager, job1, _transcript("old turn"))
    await handle_transcript(manager, job2, _transcript("new turn"))

    pending = session.pending_by_speaker.get(10)
    assert pending is not None
    assert pending.text == "new turn"
    assert pending.fragment_count == 1


@pytest.mark.asyncio
async def test_empty_transcript_triggers_schedule_release(monkeypatch) -> None:
    """Empty transcript text with a speaker_id should not create a pending turn."""
    manager = _make_manager(monkeypatch)
    await _join_manager(manager, monkeypatch)
    session = manager.active_session

    job = _capture_job()
    await handle_transcript(manager, job, _transcript(""))

    assert 10 not in session.pending_by_speaker


@pytest.mark.asyncio
async def test_no_session_is_noop(monkeypatch) -> None:
    """handle_transcript does nothing when there's no active session."""
    manager = _make_manager(monkeypatch)
    # No join — no session

    job = _capture_job()
    # Should not raise
    await handle_transcript(manager, job, _transcript("hello"))


@pytest.mark.asyncio
async def test_wrong_guild_is_noop(monkeypatch) -> None:
    """handle_transcript ignores transcripts from a different guild."""
    manager = _make_manager(monkeypatch)
    await _join_manager(manager, monkeypatch)

    job = CaptureJob(
        guild_id=999,  # different guild
        channel_id=99,
        path=Path("/tmp/test.wav"),
        speaker_id=10,
        speaker_label="alice",
        ssrc=1234,
        started_at=1.0,
        ended_at=1.5,
        duration_seconds=0.5,
        packet_count=12,
        saved_at=1.6,
    )
    await handle_transcript(manager, job, _transcript("hello"))
    assert manager.active_session.pending_by_speaker == {}


# ── maybe_schedule_release ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_schedule_release_skips_when_speaker_active(monkeypatch) -> None:
    """Release is not scheduled if speaker is still talking."""
    manager = _make_manager(monkeypatch)
    await _join_manager(manager, monkeypatch)
    session = manager.active_session

    session.pending_by_speaker[10] = PendingSpeakerTurn(
        speaker_id=10, speaker_name="alice", text="hi",
        started_at=1.0, ended_at=1.5, fragment_count=1,
        transcripts=["hi"],
    )
    session.active_speakers.add(10)

    maybe_schedule_release(manager, session, 10)
    assert session.pending_by_speaker[10].release_task is None


@pytest.mark.asyncio
async def test_schedule_release_skips_when_pending_stt(monkeypatch) -> None:
    """Release is not scheduled if there are pending STT jobs."""
    manager = _make_manager(monkeypatch)
    await _join_manager(manager, monkeypatch)
    session = manager.active_session

    session.pending_by_speaker[10] = PendingSpeakerTurn(
        speaker_id=10, speaker_name="alice", text="hi",
        started_at=1.0, ended_at=1.5, fragment_count=1,
        transcripts=["hi"],
    )
    session.pending_stt_counts[10] = 2

    maybe_schedule_release(manager, session, 10)
    assert session.pending_by_speaker[10].release_task is None


@pytest.mark.asyncio
async def test_schedule_release_creates_task(monkeypatch) -> None:
    """Release task is created when conditions are met."""
    manager = _make_manager(monkeypatch)
    await _join_manager(manager, monkeypatch)
    session = manager.active_session

    session.pending_by_speaker[10] = PendingSpeakerTurn(
        speaker_id=10, speaker_name="alice", text="hi",
        started_at=1.0, ended_at=1.5, fragment_count=1,
        transcripts=["hi"],
    )

    maybe_schedule_release(manager, session, 10)
    task = session.pending_by_speaker[10].release_task
    assert task is not None
    # Clean up
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


# ── release_pending_turn ─────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_release_pending_turn_removes_from_speaker_map(monkeypatch) -> None:
    manager = _make_manager(monkeypatch)
    await _join_manager(manager, monkeypatch)
    session = manager.active_session

    session.pending_by_speaker[10] = PendingSpeakerTurn(
        speaker_id=10, speaker_name="alice", text="hello there",
        started_at=1.0, ended_at=1.5, fragment_count=1,
        total_audio_seconds=0.5, total_stt_elapsed_seconds=0.2,
        transcripts=["hello there"],
    )

    await release_pending_turn(manager, session, 10)
    assert 10 not in session.pending_by_speaker


@pytest.mark.asyncio
async def test_release_pending_turn_idempotent(monkeypatch) -> None:
    """Releasing a speaker_id that has no pending turn is a no-op."""
    manager = _make_manager(monkeypatch)
    await _join_manager(manager, monkeypatch)
    session = manager.active_session

    # Should not raise
    await release_pending_turn(manager, session, 42)


@pytest.mark.asyncio
async def test_release_cancels_existing_tasks(monkeypatch) -> None:
    """Release cancels both release_task and force_release_task."""
    manager = _make_manager(monkeypatch)
    await _join_manager(manager, monkeypatch)
    session = manager.active_session

    release_task = asyncio.create_task(asyncio.sleep(30))
    force_task = asyncio.create_task(asyncio.sleep(30))
    session.pending_by_speaker[10] = PendingSpeakerTurn(
        speaker_id=10, speaker_name="alice", text="hello",
        started_at=1.0, ended_at=1.5, fragment_count=1,
        total_audio_seconds=0.5, total_stt_elapsed_seconds=0.2,
        transcripts=["hello"],
        release_task=release_task,
        force_release_task=force_task,
    )

    await release_pending_turn(manager, session, 10)
    await asyncio.sleep(0)

    assert release_task.cancelled()
    assert force_task.cancelled()


# ── emit_completed_turn ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_emit_completed_turn_adds_to_history(monkeypatch) -> None:
    manager = _make_manager(monkeypatch)
    await _join_manager(manager, monkeypatch)
    session = manager.active_session

    turn = SimpleNamespace(
        speaker_id=10, speaker_name="alice", text="hello",
        started_at=1.0, ended_at=1.5, fragment_count=1,
        total_audio_seconds=0.5, total_stt_elapsed_seconds=0.2,
        transcripts=["hello"],
    )

    await emit_completed_turn(manager, session, completed_turn=turn)

    entries = session.history.entries()
    assert len(entries) == 1
    assert entries[0].text == "hello"
    assert entries[0].speaker_name == "alice"
    assert entries[0].is_bot is False


@pytest.mark.asyncio
async def test_emit_completed_turn_sets_response_needed(monkeypatch) -> None:
    manager = _make_manager(monkeypatch)
    await _join_manager(manager, monkeypatch)
    session = manager.active_session

    turn = SimpleNamespace(
        speaker_id=10, speaker_name="alice", text="hello",
        started_at=1.0, ended_at=1.5, fragment_count=1,
        total_audio_seconds=0.5, total_stt_elapsed_seconds=0.2,
        transcripts=["hello"],
    )

    await emit_completed_turn(manager, session, completed_turn=turn)

    assert session.pending_response_needed is True
    assert len(session.pending_response_turns) == 1
