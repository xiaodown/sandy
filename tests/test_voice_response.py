"""Tests for sandy.voice.response — brain call, TTS, playback error paths."""

import asyncio
import io
from types import SimpleNamespace
from unittest.mock import AsyncMock
import wave

import pytest

from sandy.runtime_state import RuntimeState
from sandy.voice import VoiceManager
from sandy.voice.response import respond_to_session, store_voice_memory, warm_voice_models


# ── Helpers ──────────────────────────────────────────────────────────────────

def _silence_loggers(monkeypatch):
    import sandy.voice.manager as vm
    import sandy.voice.response as resp
    import sandy.voice.stitching as st

    for mod in (vm, resp, st):
        for method_name in ("info", "warning", "error", "exception", "debug"):
            monkeypatch.setattr(mod.logger, method_name, lambda *a, **kw: None)


def _wav_bytes() -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(24_000)
        wav_file.writeframes(b"\x00\x00" * 64)
    return buf.getvalue()


async def _make_active_manager(monkeypatch, *, brain_content="ok fine", brain_raises=False):
    from tests.test_voice_manager import DummyAuthor, DummyGuild, DummyMessage, DummyVoiceChannel

    _silence_loggers(monkeypatch)

    if brain_raises:
        llm = SimpleNamespace(ask_brain=AsyncMock(side_effect=RuntimeError("brain exploded")))
    else:
        llm = SimpleNamespace(
            ask_brain=AsyncMock(
                return_value=SimpleNamespace(content=brain_content, done_reason="stop"),
            ),
        )

    manager = VoiceManager(
        registry=SimpleNamespace(is_voice_admin=lambda **_: True),
        runtime_state=RuntimeState(),
        llm=llm,
        vector_memory=SimpleNamespace(
            query=AsyncMock(return_value=""),
            add_message=AsyncMock(return_value=True),
        ),
    )
    manager._warm_voice_models = AsyncMock()
    manager._tts = SimpleNamespace(synthesize_bytes=AsyncMock(return_value=_wav_bytes()))
    manager._play_source = AsyncMock()

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
    return manager


def _queue_turn(session, *, text="hello from voice"):
    session.pending_response_turns.append(
        SimpleNamespace(
            speaker_id=10, speaker_name="alice", text=text,
            started_at=1.0, ended_at=1.5, fragment_count=1,
            total_audio_seconds=0.5, total_stt_elapsed_seconds=0.2,
            transcripts=[text],
        ),
    )
    session.pending_response_needed = True


# ── Brain failure path ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_brain_exception_continues_loop(monkeypatch) -> None:
    """When brain raises, the response loop logs error and continues."""
    manager = await _make_active_manager(monkeypatch, brain_raises=True)
    session = manager.active_session

    _queue_turn(session)
    await respond_to_session(manager, session.session_id)

    # Brain was called
    manager.llm.ask_brain.assert_awaited_once()
    # TTS should NOT have been called
    manager._tts.synthesize_bytes.assert_not_awaited()
    # Runtime state should show error
    voice_state = manager.runtime_state.snapshot()["voice"]
    assert voice_state["last_error"] == "Voice brain call failed"


# ── Empty reply path ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_empty_brain_reply_skips_tts(monkeypatch) -> None:
    """When brain returns empty/whitespace, TTS is not called."""
    manager = await _make_active_manager(monkeypatch, brain_content="   ")
    session = manager.active_session

    _queue_turn(session)
    await respond_to_session(manager, session.session_id)

    manager.llm.ask_brain.assert_awaited_once()
    manager._tts.synthesize_bytes.assert_not_awaited()
    manager._play_source.assert_not_awaited()


# ── TTS fallback on failure ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_tts_fallback_retries_shorter_text(monkeypatch) -> None:
    """When primary TTS fails, retry with truncated text."""
    manager = await _make_active_manager(
        monkeypatch,
        brain_content="this is a moderately long reply that should definitely get truncated on the fallback path because it has way more than sixteen words in total",
    )
    session = manager.active_session

    call_count = 0

    async def flaky_tts(text):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("TTS failed")
        return _wav_bytes()

    manager._tts = SimpleNamespace(synthesize_bytes=flaky_tts)

    _queue_turn(session)
    await respond_to_session(manager, session.session_id)

    assert call_count == 2  # first failed, second succeeded
    manager._play_source.assert_awaited_once()


@pytest.mark.asyncio
async def test_tts_total_failure_resets_playback_active(monkeypatch) -> None:
    """When TTS fails entirely, playback_active is reset to False."""
    manager = await _make_active_manager(monkeypatch, brain_content="short")
    session = manager.active_session

    manager._tts = SimpleNamespace(
        synthesize_bytes=AsyncMock(side_effect=RuntimeError("TTS service down")),
    )

    _queue_turn(session)
    await respond_to_session(manager, session.session_id)

    assert session.playback_active is False
    manager._play_source.assert_not_awaited()


# ── Playback active guard ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_exits_early_if_playback_active(monkeypatch) -> None:
    """respond_to_session exits immediately if playback is already active."""
    manager = await _make_active_manager(monkeypatch)
    session = manager.active_session
    session.playback_active = True

    _queue_turn(session)
    await respond_to_session(manager, session.session_id)

    manager.llm.ask_brain.assert_not_awaited()


# ── Missing session ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_exits_early_if_session_gone(monkeypatch) -> None:
    """respond_to_session exits if the session was replaced."""
    manager = await _make_active_manager(monkeypatch)
    session = manager.active_session
    _queue_turn(session)

    # Respond with wrong session_id
    await respond_to_session(manager, "nonexistent-session-id")

    manager.llm.ask_brain.assert_not_awaited()


# ── store_voice_memory failure ───────────────────────────────────────────────

@pytest.mark.asyncio
async def test_store_voice_memory_swallows_exception(monkeypatch) -> None:
    """store_voice_memory catches and logs exceptions without propagating."""
    _silence_loggers(monkeypatch)
    manager = VoiceManager(
        registry=SimpleNamespace(is_voice_admin=lambda **_: True),
        runtime_state=RuntimeState(),
        llm=SimpleNamespace(),
        vector_memory=SimpleNamespace(
            add_message=AsyncMock(side_effect=RuntimeError("chroma down")),
        ),
    )

    session = SimpleNamespace(guild_id=1)
    # Should not raise
    await store_voice_memory(manager, session, message_id="test-1", author_name="alice", text="hello")


# ── warm_voice_models ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_warm_voice_models_suppresses_exceptions(monkeypatch) -> None:
    """warm_voice_models catches exceptions from both transcriber and TTS."""
    _silence_loggers(monkeypatch)
    manager = VoiceManager(
        registry=SimpleNamespace(is_voice_admin=lambda **_: True),
        runtime_state=RuntimeState(),
        llm=SimpleNamespace(),
        vector_memory=SimpleNamespace(),
    )
    manager._transcriber = SimpleNamespace(warmup=AsyncMock(side_effect=RuntimeError("no GPU")))
    manager._tts = SimpleNamespace(warmup=AsyncMock(side_effect=RuntimeError("no service")))

    # Should not raise
    await warm_voice_models(manager)
