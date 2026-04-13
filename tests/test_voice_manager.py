import asyncio
import io
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock
import wave

import pytest

from sandy.runtime_state import RuntimeState
from sandy.voice import VoiceManager
from sandy.voice.capture import CaptureJob
from sandy.voice.models import PendingSpeakerTurn
from sandy.voice.stt import TranscriptResult


class DummyVoiceClient:
    def __init__(self) -> None:
        self.disconnect = AsyncMock()
        self._connected = True

    def is_connected(self) -> bool:
        return self._connected

    def is_playing(self) -> bool:
        return False


class DummyVoiceChannel:
    def __init__(self, channel_id: int, name: str, *, members=None) -> None:
        self.id = channel_id
        self.name = name
        self.members = list(members or [])
        self._voice_client = DummyVoiceClient()
        self.connect = AsyncMock(return_value=self._voice_client)


class DummyGuild:
    def __init__(self, guild_id: int, name: str, voice_channels: list[DummyVoiceChannel]) -> None:
        self.id = guild_id
        self.name = name
        self.voice_channels = voice_channels


class DummyAuthor:
    def __init__(self, *, user_id: int, display_name: str, voice_channel=None) -> None:
        self.id = user_id
        self.display_name = display_name
        self.voice = SimpleNamespace(channel=voice_channel)


class DummyChannel:
    def __init__(self) -> None:
        self.sent_messages: list[str] = []

    async def send(self, content: str) -> None:
        self.sent_messages.append(content)


class DummyMessage:
    def __init__(self, *, guild, author, content: str = "", channel=None) -> None:
        self.guild = guild
        self.author = author
        self.content = content
        self.channel = channel or DummyChannel()


@pytest.mark.asyncio
async def test_voice_manager_joins_and_leaves_channel() -> None:
    voice_channel = DummyVoiceChannel(
        99,
        "ops war room",
        members=[SimpleNamespace(id=10, display_name="alice"), SimpleNamespace(id=20, display_name="bob")],
    )
    guild = DummyGuild(1, "Guild", [voice_channel])
    registry = SimpleNamespace(is_voice_admin=lambda **_: True)
    state = RuntimeState()
    manager = VoiceManager(
        registry=registry,
        runtime_state=state,
        llm=SimpleNamespace(),
        vector_memory=SimpleNamespace(),
    )
    message = DummyMessage(
        guild=guild,
        author=DummyAuthor(user_id=10, display_name="alice", voice_channel=voice_channel),
        content="!join",
    )

    join = await manager.handle_text_command(message, bot_user=SimpleNamespace(id=999))

    assert join.handled is True
    assert join.ok is True
    assert state.snapshot()["voice"]["active"] is True
    assert state.snapshot()["voice"]["channel_name"] == "ops war room"
    assert state.snapshot()["voice"]["participant_names"] == ["alice", "bob"]

    leave = await manager.handle_text_command(
        DummyMessage(guild=guild, author=message.author, content="!leave"),
        bot_user=SimpleNamespace(id=999),
    )

    assert leave.handled is True
    assert leave.ok is True
    voice_channel._voice_client.disconnect.assert_awaited_once()
    assert state.snapshot()["voice"]["active"] is False


@pytest.mark.asyncio
async def test_voice_manager_denies_non_admin() -> None:
    voice_channel = DummyVoiceChannel(99, "ops war room")
    guild = DummyGuild(1, "Guild", [voice_channel])
    registry = SimpleNamespace(is_voice_admin=lambda **_: False)
    manager = VoiceManager(
        registry=registry,
        runtime_state=RuntimeState(),
        llm=SimpleNamespace(),
        vector_memory=SimpleNamespace(),
    )
    message = DummyMessage(
        guild=guild,
        author=DummyAuthor(user_id=10, display_name="alice", voice_channel=voice_channel),
        content="!join",
    )

    result = await manager.handle_text_command(message, bot_user=SimpleNamespace(id=999))

    assert result.handled is True
    assert result.ok is False
    voice_channel.connect.assert_not_awaited()


@pytest.mark.asyncio
async def test_voice_manager_generates_reply_from_completed_turn(monkeypatch) -> None:
    import sandy.voice.manager as voice_manager_module

    for method_name in ("info", "warning", "error", "exception", "debug"):
        monkeypatch.setattr(voice_manager_module.logger, method_name, lambda *args, **kwargs: None)

    voice_channel = DummyVoiceChannel(
        99,
        "ops war room",
        members=[SimpleNamespace(id=10, display_name="alice")],
    )
    guild = DummyGuild(1, "Guild", [voice_channel])
    llm = SimpleNamespace(
        ask_brain=AsyncMock(return_value=SimpleNamespace(content="yeah, fair enough", done_reason="stop")),
    )
    vector_memory = SimpleNamespace(
        query=AsyncMock(return_value=""),
        add_message=AsyncMock(return_value=True),
    )
    manager = VoiceManager(
        registry=SimpleNamespace(is_voice_admin=lambda **_: True),
        runtime_state=RuntimeState(),
        llm=llm,
        vector_memory=vector_memory,
    )
    manager._warm_voice_models = AsyncMock()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(24_000)
        wav_file.writeframes(b"\x00\x00" * 64)
    manager._tts = SimpleNamespace(synthesize_bytes=AsyncMock(return_value=buf.getvalue()))
    manager._play_source = AsyncMock()

    join_message = DummyMessage(
        guild=guild,
        author=DummyAuthor(user_id=10, display_name="alice", voice_channel=voice_channel),
        content="!join",
    )
    await manager.handle_text_command(join_message, bot_user=SimpleNamespace(id=999, display_name="Sandy"))

    session = manager.active_session
    assert session is not None

    await manager._emit_completed_turn(
        session,
        completed_turn=SimpleNamespace(
            speaker_id=10,
            speaker_name="alice",
            text="hello from voice",
            started_at=1.0,
            ended_at=1.5,
            fragment_count=1,
            total_audio_seconds=0.5,
            total_stt_elapsed_seconds=0.2,
            transcripts=["hello from voice"],
        ),
    )
    await session.response_task
    await asyncio.sleep(0)

    llm.ask_brain.assert_awaited_once()
    assert llm.ask_brain.await_args.kwargs["mode"] == "voice"
    manager._tts.synthesize_bytes.assert_awaited_once_with("yeah, fair enough.")
    manager._play_source.assert_awaited_once()
    assert manager.runtime_state.snapshot()["voice"]["current_trace_id"].startswith("voice:")
    assert manager.runtime_state.snapshot()["voice"]["last_transcript"] == "alice: hello from voice"
    assert manager.runtime_state.snapshot()["voice"]["last_reply"] == "yeah, fair enough."
    assert [entry.text for entry in session.history.entries()] == [
        "hello from voice",
        "yeah, fair enough.",
    ]


@pytest.mark.asyncio
async def test_voice_manager_handles_transcript_without_stable_speaker_id(monkeypatch) -> None:
    import sandy.voice.manager as voice_manager_module

    for method_name in ("info", "warning", "error", "exception", "debug"):
        monkeypatch.setattr(voice_manager_module.logger, method_name, lambda *args, **kwargs: None)

    voice_channel = DummyVoiceChannel(
        99,
        "ops war room",
        members=[SimpleNamespace(id=10, display_name="alice")],
    )
    guild = DummyGuild(1, "Guild", [voice_channel])
    llm = SimpleNamespace(
        ask_brain=AsyncMock(return_value=SimpleNamespace(content="heard you loud and clear", done_reason="stop")),
    )
    vector_memory = SimpleNamespace(
        query=AsyncMock(return_value=""),
        add_message=AsyncMock(return_value=True),
    )
    manager = VoiceManager(
        registry=SimpleNamespace(is_voice_admin=lambda **_: True),
        runtime_state=RuntimeState(),
        llm=llm,
        vector_memory=vector_memory,
    )
    manager._warm_voice_models = AsyncMock()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(24_000)
        wav_file.writeframes(b"\x00\x00" * 64)
    manager._tts = SimpleNamespace(synthesize_bytes=AsyncMock(return_value=buf.getvalue()))
    manager._play_source = AsyncMock()

    join_message = DummyMessage(
        guild=guild,
        author=DummyAuthor(user_id=10, display_name="alice", voice_channel=voice_channel),
        content="!join",
    )
    await manager.handle_text_command(join_message, bot_user=SimpleNamespace(id=999, display_name="Sandy"))

    session = manager.active_session
    assert session is not None

    await manager._handle_transcript(
        CaptureJob(
            guild_id=guild.id,
            channel_id=voice_channel.id,
            path=Path("/tmp/unknown-speaker.wav"),
            speaker_id=None,
            speaker_label="mystery",
            ssrc=1234,
            started_at=1.0,
            ended_at=1.4,
            duration_seconds=0.4,
            packet_count=12,
            saved_at=1.5,
        ),
        TranscriptResult(
            text="hello from the void",
            language="en",
            language_probability=0.99,
            elapsed_seconds=0.2,
            device="cpu",
            compute_type="int8",
        ),
    )
    await session.response_task
    await asyncio.sleep(0)

    llm.ask_brain.assert_awaited_once()
    manager._tts.synthesize_bytes.assert_awaited_once_with("heard you loud and clear.")
    assert manager.runtime_state.snapshot()["voice"]["last_transcript"] == "mystery: hello from the void"
    assert [entry.text for entry in session.history.entries()] == [
        "hello from the void",
        "heard you loud and clear.",
    ]


@pytest.mark.asyncio
async def test_voice_manager_disconnect_cancels_all_pending_release_tasks() -> None:
    voice_channel = DummyVoiceChannel(
        99,
        "ops war room",
        members=[SimpleNamespace(id=10, display_name="alice")],
    )
    guild = DummyGuild(1, "Guild", [voice_channel])
    manager = VoiceManager(
        registry=SimpleNamespace(is_voice_admin=lambda **_: True),
        runtime_state=RuntimeState(),
        llm=SimpleNamespace(),
        vector_memory=SimpleNamespace(),
    )

    join_message = DummyMessage(
        guild=guild,
        author=DummyAuthor(user_id=10, display_name="alice", voice_channel=voice_channel),
        content="!join",
    )
    await manager.handle_text_command(join_message, bot_user=SimpleNamespace(id=999, display_name="Sandy"))

    session = manager.active_session
    assert session is not None

    release_task = asyncio.create_task(asyncio.sleep(30))
    force_release_task = asyncio.create_task(asyncio.sleep(30))
    session.pending_by_speaker[10] = PendingSpeakerTurn(
        speaker_id=10,
        speaker_name="alice",
        text="still talking",
        started_at=1.0,
        ended_at=2.0,
        fragment_count=1,
        total_audio_seconds=1.0,
        total_stt_elapsed_seconds=0.3,
        transcripts=["still talking"],
        release_task=release_task,
        force_release_task=force_release_task,
    )
    session.pending_stt_counts[10] = 1
    session.active_speakers.add(10)

    await manager._disconnect_active_voice_client()
    await asyncio.sleep(0)

    assert release_task.cancelled() is True
    assert force_release_task.cancelled() is True
    assert session.pending_by_speaker == {}
    assert session.pending_stt_counts == {}
    assert session.active_speakers == set()


# ── Command edge cases ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_unrecognized_command_returns_not_handled() -> None:
    manager = VoiceManager(
        registry=SimpleNamespace(is_voice_admin=lambda **_: True),
        runtime_state=RuntimeState(),
        llm=SimpleNamespace(),
        vector_memory=SimpleNamespace(),
    )
    guild = DummyGuild(1, "Guild", [])
    msg = DummyMessage(
        guild=guild,
        author=DummyAuthor(user_id=10, display_name="alice"),
        content="!dance",
    )

    result = await manager.handle_text_command(msg, bot_user=SimpleNamespace(id=999))
    assert result.handled is False


@pytest.mark.asyncio
async def test_non_command_message_returns_not_handled() -> None:
    manager = VoiceManager(
        registry=SimpleNamespace(is_voice_admin=lambda **_: True),
        runtime_state=RuntimeState(),
        llm=SimpleNamespace(),
        vector_memory=SimpleNamespace(),
    )
    guild = DummyGuild(1, "Guild", [])
    msg = DummyMessage(
        guild=guild,
        author=DummyAuthor(user_id=10, display_name="alice"),
        content="hey everyone",
    )

    result = await manager.handle_text_command(msg, bot_user=SimpleNamespace(id=999))
    assert result.handled is False


@pytest.mark.asyncio
async def test_double_join_returns_error() -> None:
    voice_channel = DummyVoiceChannel(
        99, "ops war room",
        members=[SimpleNamespace(id=10, display_name="alice")],
    )
    guild = DummyGuild(1, "Guild", [voice_channel])
    manager = VoiceManager(
        registry=SimpleNamespace(is_voice_admin=lambda **_: True),
        runtime_state=RuntimeState(),
        llm=SimpleNamespace(),
        vector_memory=SimpleNamespace(),
    )
    manager._warm_voice_models = AsyncMock()
    author = DummyAuthor(user_id=10, display_name="alice", voice_channel=voice_channel)
    bot = SimpleNamespace(id=999, display_name="Sandy")

    first = await manager.handle_text_command(
        DummyMessage(guild=guild, author=author, content="!join"), bot_user=bot,
    )
    assert first.ok is True

    second = await manager.handle_text_command(
        DummyMessage(guild=guild, author=author, content="!join"), bot_user=bot,
    )
    assert second.ok is False
    assert "already" in second.reply


@pytest.mark.asyncio
async def test_leave_without_session_returns_error() -> None:
    guild = DummyGuild(1, "Guild", [])
    manager = VoiceManager(
        registry=SimpleNamespace(is_voice_admin=lambda **_: True),
        runtime_state=RuntimeState(),
        llm=SimpleNamespace(),
        vector_memory=SimpleNamespace(),
    )
    result = await manager.handle_text_command(
        DummyMessage(
            guild=guild,
            author=DummyAuthor(user_id=10, display_name="alice"),
            content="!leave",
        ),
        bot_user=SimpleNamespace(id=999),
    )
    assert result.ok is False
    assert "not in" in result.reply


@pytest.mark.asyncio
async def test_leave_denied_for_non_admin() -> None:
    guild = DummyGuild(1, "Guild", [])
    manager = VoiceManager(
        registry=SimpleNamespace(is_voice_admin=lambda **_: False),
        runtime_state=RuntimeState(),
        llm=SimpleNamespace(),
        vector_memory=SimpleNamespace(),
    )
    result = await manager.handle_text_command(
        DummyMessage(
            guild=guild,
            author=DummyAuthor(user_id=10, display_name="alice"),
            content="!leave",
        ),
        bot_user=SimpleNamespace(id=999),
    )
    assert result.ok is False
    assert "not allowed" in result.reply


@pytest.mark.asyncio
async def test_join_no_guild_returns_error() -> None:
    manager = VoiceManager(
        registry=SimpleNamespace(is_voice_admin=lambda **_: True),
        runtime_state=RuntimeState(),
        llm=SimpleNamespace(),
        vector_memory=SimpleNamespace(),
    )
    result = await manager.handle_text_command(
        DummyMessage(
            guild=None,
            author=DummyAuthor(user_id=10, display_name="alice"),
            content="!join",
        ),
        bot_user=SimpleNamespace(id=999),
    )
    assert result.ok is False
    assert "servers" in result.reply


@pytest.mark.asyncio
async def test_leave_no_guild_returns_error() -> None:
    manager = VoiceManager(
        registry=SimpleNamespace(is_voice_admin=lambda **_: True),
        runtime_state=RuntimeState(),
        llm=SimpleNamespace(),
        vector_memory=SimpleNamespace(),
    )
    result = await manager.handle_text_command(
        DummyMessage(
            guild=None,
            author=DummyAuthor(user_id=10, display_name="alice"),
            content="!leave",
        ),
        bot_user=SimpleNamespace(id=999),
    )
    assert result.ok is False
    assert "servers" in result.reply


@pytest.mark.asyncio
async def test_join_unresolvable_channel_returns_error() -> None:
    guild = DummyGuild(1, "Guild", [DummyVoiceChannel(99, "music")])
    manager = VoiceManager(
        registry=SimpleNamespace(is_voice_admin=lambda **_: True),
        runtime_state=RuntimeState(),
        llm=SimpleNamespace(),
        vector_memory=SimpleNamespace(),
    )
    result = await manager.handle_text_command(
        DummyMessage(
            guild=guild,
            author=DummyAuthor(user_id=10, display_name="alice", voice_channel=None),
            content="!join nonexistent channel",
        ),
        bot_user=SimpleNamespace(id=999),
    )
    assert result.ok is False
    assert "resolve" in result.reply


# ── text_replies_paused ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_text_replies_paused_reflects_session_state() -> None:
    voice_channel = DummyVoiceChannel(
        99, "ops war room",
        members=[SimpleNamespace(id=10, display_name="alice")],
    )
    guild = DummyGuild(1, "Guild", [voice_channel])
    manager = VoiceManager(
        registry=SimpleNamespace(is_voice_admin=lambda **_: True),
        runtime_state=RuntimeState(),
        llm=SimpleNamespace(),
        vector_memory=SimpleNamespace(),
    )
    manager._warm_voice_models = AsyncMock()

    assert manager.text_replies_paused() is False

    await manager.handle_text_command(
        DummyMessage(
            guild=guild,
            author=DummyAuthor(user_id=10, display_name="alice", voice_channel=voice_channel),
            content="!join",
        ),
        bot_user=SimpleNamespace(id=999, display_name="Sandy"),
    )
    assert manager.text_replies_paused() is True

    await manager.handle_text_command(
        DummyMessage(
            guild=guild,
            author=DummyAuthor(user_id=10, display_name="alice"),
            content="!leave",
        ),
        bot_user=SimpleNamespace(id=999),
    )
    assert manager.text_replies_paused() is False


# ── handle_voice_state_update ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_voice_state_update_updates_participant_names() -> None:
    voice_channel = DummyVoiceChannel(
        99, "ops war room",
        members=[SimpleNamespace(id=10, display_name="alice")],
    )
    guild = DummyGuild(1, "Guild", [voice_channel])
    manager = VoiceManager(
        registry=SimpleNamespace(is_voice_admin=lambda **_: True),
        runtime_state=RuntimeState(),
        llm=SimpleNamespace(),
        vector_memory=SimpleNamespace(),
    )
    manager._warm_voice_models = AsyncMock()
    bot_user = SimpleNamespace(id=999, display_name="Sandy")

    await manager.handle_text_command(
        DummyMessage(
            guild=guild,
            author=DummyAuthor(user_id=10, display_name="alice", voice_channel=voice_channel),
            content="!join",
        ),
        bot_user=bot_user,
    )

    # Simulate bob joining: update channel members, then fire voice state update
    voice_channel.members.append(SimpleNamespace(id=20, display_name="bob"))
    member = SimpleNamespace(id=20, guild=SimpleNamespace(id=1))
    before = SimpleNamespace(channel=None)
    after = SimpleNamespace(channel=voice_channel)

    manager.handle_voice_state_update(member, before, after, bot_user=bot_user)

    session = manager.active_session
    assert "bob" in session.participant_names


@pytest.mark.asyncio
async def test_voice_state_update_ignores_other_guilds() -> None:
    voice_channel = DummyVoiceChannel(
        99, "ops war room",
        members=[SimpleNamespace(id=10, display_name="alice")],
    )
    guild = DummyGuild(1, "Guild", [voice_channel])
    manager = VoiceManager(
        registry=SimpleNamespace(is_voice_admin=lambda **_: True),
        runtime_state=RuntimeState(),
        llm=SimpleNamespace(),
        vector_memory=SimpleNamespace(),
    )
    manager._warm_voice_models = AsyncMock()
    bot_user = SimpleNamespace(id=999, display_name="Sandy")

    await manager.handle_text_command(
        DummyMessage(
            guild=guild,
            author=DummyAuthor(user_id=10, display_name="alice", voice_channel=voice_channel),
            content="!join",
        ),
        bot_user=bot_user,
    )

    original_names = list(manager.active_session.participant_names)

    # Event from a different guild — should be ignored
    member = SimpleNamespace(id=20, guild=SimpleNamespace(id=999))
    before = SimpleNamespace(channel=None)
    after = SimpleNamespace(channel=SimpleNamespace(id=99))
    manager.handle_voice_state_update(member, before, after, bot_user=bot_user)

    assert manager.active_session.participant_names == original_names


@pytest.mark.asyncio
async def test_voice_state_update_noop_without_session() -> None:
    manager = VoiceManager(
        registry=SimpleNamespace(is_voice_admin=lambda **_: True),
        runtime_state=RuntimeState(),
        llm=SimpleNamespace(),
        vector_memory=SimpleNamespace(),
    )
    # Should not raise
    member = SimpleNamespace(id=10, guild=SimpleNamespace(id=1))
    before = SimpleNamespace(channel=None)
    after = SimpleNamespace(channel=SimpleNamespace(id=99))
    manager.handle_voice_state_update(member, before, after, bot_user=SimpleNamespace(id=999))


# ── shutdown ─────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_shutdown_disconnects_and_stops_stt() -> None:
    voice_channel = DummyVoiceChannel(
        99, "ops war room",
        members=[SimpleNamespace(id=10, display_name="alice")],
    )
    guild = DummyGuild(1, "Guild", [voice_channel])
    manager = VoiceManager(
        registry=SimpleNamespace(is_voice_admin=lambda **_: True),
        runtime_state=RuntimeState(),
        llm=SimpleNamespace(),
        vector_memory=SimpleNamespace(),
    )
    manager._warm_voice_models = AsyncMock()

    await manager.handle_text_command(
        DummyMessage(
            guild=guild,
            author=DummyAuthor(user_id=10, display_name="alice", voice_channel=voice_channel),
            content="!join",
        ),
        bot_user=SimpleNamespace(id=999, display_name="Sandy"),
    )
    assert manager.active_session is not None

    await manager.shutdown()
    assert manager.active_session is None
