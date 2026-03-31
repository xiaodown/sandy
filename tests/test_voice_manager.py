from types import SimpleNamespace
from unittest.mock import AsyncMock
import io
import wave

import pytest

from sandy.runtime_state import RuntimeState
from sandy.voice import VoiceManager


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
async def test_voice_manager_generates_reply_from_completed_turn() -> None:
    voice_channel = DummyVoiceChannel(
        99,
        "ops war room",
        members=[SimpleNamespace(id=10, display_name="alice")],
    )
    guild = DummyGuild(1, "Guild", [voice_channel])
    llm = SimpleNamespace(
        ask_brain=AsyncMock(return_value=SimpleNamespace(content="yeah, fair enough")),
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
        speaker_id=10,
        speaker_name="alice",
        text="hello from voice",
    )
    await session.response_task

    llm.ask_brain.assert_awaited_once()
    assert llm.ask_brain.await_args.kwargs["mode"] == "voice"
    manager._tts.synthesize_bytes.assert_awaited_once_with("yeah, fair enough.")
    manager._play_source.assert_awaited_once()
    assert [entry.text for entry in session.history.entries()] == [
        "hello from voice",
        "yeah, fair enough.",
    ]
