from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from sandy.llm import BrainResponse
from sandy.pipeline import AttachmentProcessingResult


@dataclass
class DummyAuthor:
    id: int
    display_name: str
    bot: bool = False
    name: str = "user"
    nick: str | None = None


@dataclass
class DummyGuild:
    id: int
    name: str


class DummyTyping:
    async def __aenter__(self) -> None:
        return None

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False


class DummyChannel:
    def __init__(self, channel_id: int = 456, name: str = "general") -> None:
        self.id = channel_id
        self.name = name
        self.sent_messages: list[str] = []

    def typing(self) -> DummyTyping:
        return DummyTyping()

    async def send(self, content: str) -> None:
        self.sent_messages.append(content)


@dataclass
class DummyMessage:
    id: int
    author: DummyAuthor
    guild: DummyGuild
    channel: DummyChannel
    content: str = ""
    attachments: list | None = None
    mentions: list | None = None
    created_at: datetime = datetime.now(UTC)

    def __post_init__(self) -> None:
        if self.attachments is None:
            self.attachments = []
        if self.mentions is None:
            self.mentions = []


class FakeHistory:
    def __init__(self) -> None:
        self.format_value = "[1m ago] [friend] hey sandy"
        self.ollama_messages = [{"role": "user", "content": "hey sandy"}]

    def format(self) -> str:
        return self.format_value

    def to_ollama_messages(self, _bot_id: int) -> list[dict]:
        return list(self.ollama_messages)


class FakeCache:
    def __init__(self, history: FakeHistory | None = None) -> None:
        self.history = history or FakeHistory()
        self.added: list[object] = []

    def add(self, message: object) -> None:
        self.added.append(message)

    def get(self, _server_id: int, _channel_id: int) -> FakeHistory:
        return self.history


class FakeBackgroundTasks:
    def __init__(self) -> None:
        self.names: list[str] = []

    def create_task(self, coro, *, name: str):
        self.names.append(name)
        coro.close()
        return None


class FakeMemoryWorker:
    def __init__(self, steps: list[str] | None = None) -> None:
        self.steps = steps if steps is not None else []
        self.calls: list[tuple[DummyMessage, list[str] | None]] = []

    async def enqueue(self, message, image_descriptions=None) -> None:
        self.steps.append("enqueue")
        self.calls.append((message, image_descriptions))


def make_message(*, author_bot: bool = False, content: str = "hey sandy") -> DummyMessage:
    author = DummyAuthor(
        id=111 if not author_bot else 999,
        display_name="friend" if not author_bot else "Sandy",
        bot=author_bot,
        name="friend" if not author_bot else "Sandy",
    )
    return DummyMessage(
        id=12345,
        author=author,
        guild=DummyGuild(id=789, name="Test Guild"),
        channel=DummyChannel(),
        content=content,
        created_at=datetime.now(UTC),
    )


@pytest.fixture
def bot_module(monkeypatch):
    import sandy.bot as bot_module

    monkeypatch.setattr(bot_module.pipeline, "trace_event", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        bot_module,
        "bot",
        SimpleNamespace(user=SimpleNamespace(id=999, display_name="Sandy")),
    )
    monkeypatch.setattr(bot_module.pipeline, "registry", SimpleNamespace(ensure_seen=lambda message: None))
    monkeypatch.setattr(bot_module, "background_tasks", FakeBackgroundTasks())
    monkeypatch.setattr(bot_module.pipeline, "background_tasks", bot_module.background_tasks)
    return bot_module


def test_split_reply_respects_limit_and_boundaries(bot_module):
    reply = "first paragraph\n\nsecond paragraph with extra words"

    chunks = bot_module._split_reply(reply, limit=25)

    assert len(chunks) == 3
    assert all(len(chunk) <= 25 for chunk in chunks)
    assert chunks == ["first paragraph", "second paragraph with", "extra words"]


def test_finalize_reply_trims_obvious_truncation(bot_module):
    reply = "This is a complete sentence. This one got cut off in the"

    finalized = bot_module._finalize_reply(reply, done_reason="length")

    assert finalized == "This is a complete sentence."


@pytest.mark.asyncio
async def test_bot_messages_skip_bouncer_and_are_still_enqueued(bot_module, monkeypatch):
    message = make_message(author_bot=True, content="self chatter")
    cache = FakeCache()
    memory_worker = FakeMemoryWorker()
    llm = SimpleNamespace(ask_bouncer=AsyncMock())

    monkeypatch.setattr(bot_module.pipeline, "cache", cache)
    monkeypatch.setattr(bot_module.pipeline, "memory_worker", memory_worker)
    monkeypatch.setattr(bot_module.pipeline, "llm", llm)

    await bot_module.on_message(message)

    assert cache.added == [message]
    assert memory_worker.calls == [(message, None)]
    llm.ask_bouncer.assert_not_awaited()


@pytest.mark.asyncio
async def test_memory_is_enqueued_before_reply_send_failure(bot_module, monkeypatch):
    message = make_message()
    steps: list[str] = []
    trace_events: list[tuple[str, dict]] = []
    cache = FakeCache()
    memory_worker = FakeMemoryWorker(steps)
    llm = SimpleNamespace(
        ask_bouncer=AsyncMock(
            return_value=SimpleNamespace(
                should_respond=True,
                use_tool=False,
                recommended_tool=None,
                tool_parameters=None,
            )
        ),
        ask_brain=AsyncMock(return_value=BrainResponse(content="hello there", done_reason="stop")),
    )
    vector_memory = SimpleNamespace(query=AsyncMock(return_value=""))

    async def fake_send_reply(_message, _reply):
        steps.append("send")
        raise RuntimeError("discord exploded")

    monkeypatch.setattr(bot_module.pipeline, "cache", cache)
    monkeypatch.setattr(bot_module.pipeline, "memory_worker", memory_worker)
    monkeypatch.setattr(bot_module.pipeline, "llm", llm)
    monkeypatch.setattr(bot_module.pipeline, "vector_memory", vector_memory)
    monkeypatch.setattr(
        bot_module.pipeline,
        "trace_event",
        lambda _trace, stage, **kwargs: trace_events.append((stage, kwargs)),
    )
    monkeypatch.setattr(
        bot_module.pipeline,
        "describe_attachments",
        AsyncMock(return_value=AttachmentProcessingResult(descriptions=[])),
    )
    monkeypatch.setattr(bot_module.pipeline, "send_reply", fake_send_reply)

    await bot_module.on_message(message)

    assert steps == ["enqueue", "send"]
    assert memory_worker.calls == [(message, [])]
    assert ("reply_send_completed", {"status": "error"}) in trace_events
    assert trace_events[-1] == ("turn_completed", {"duration_ms": trace_events[-1][1]["duration_ms"], "replied": False})


@pytest.mark.asyncio
async def test_unknown_tool_is_ignored_without_dispatch(bot_module, monkeypatch):
    message = make_message()
    cache = FakeCache()
    memory_worker = FakeMemoryWorker()
    llm = SimpleNamespace(
        ask_bouncer=AsyncMock(
            return_value=SimpleNamespace(
                should_respond=True,
                use_tool=True,
                recommended_tool="fake_tool",
                tool_parameters={"query": "whatever"},
            )
        ),
        ask_brain=AsyncMock(return_value=BrainResponse(content="tool-free reply", done_reason="stop")),
    )
    vector_memory = SimpleNamespace(query=AsyncMock(return_value=""))
    tools = SimpleNamespace(
        KNOWN_TOOLS=frozenset({"search_web"}),
        dispatch=AsyncMock(),
    )
    send_reply = AsyncMock(return_value=1)

    monkeypatch.setattr(bot_module.pipeline, "cache", cache)
    monkeypatch.setattr(bot_module.pipeline, "memory_worker", memory_worker)
    monkeypatch.setattr(bot_module.pipeline, "llm", llm)
    monkeypatch.setattr(bot_module.pipeline, "vector_memory", vector_memory)
    monkeypatch.setattr(bot_module.pipeline, "tools_module", tools)
    monkeypatch.setattr(
        bot_module.pipeline,
        "describe_attachments",
        AsyncMock(return_value=AttachmentProcessingResult(descriptions=[])),
    )
    monkeypatch.setattr(bot_module.pipeline, "send_reply", send_reply)

    await bot_module.on_message(message)

    tools.dispatch.assert_not_awaited()
    assert llm.ask_brain.await_args.kwargs["tool_context"] is None
    send_reply.assert_awaited_once_with(message, "tool-free reply")


@pytest.mark.asyncio
async def test_attachment_descriptions_feed_rag_query_and_memory_enqueue(bot_module, monkeypatch):
    message = make_message(content="")
    cache = FakeCache()
    memory_worker = FakeMemoryWorker()
    llm = SimpleNamespace(
        ask_bouncer=AsyncMock(
            return_value=SimpleNamespace(
                should_respond=True,
                use_tool=False,
                recommended_tool=None,
                tool_parameters=None,
            )
        ),
        ask_brain=AsyncMock(return_value=BrainResponse(content="nice cat", done_reason="stop")),
    )
    vector_memory = SimpleNamespace(query=AsyncMock(return_value=""))
    send_reply = AsyncMock(return_value=1)
    image_descriptions = ["a cat sleeping on a couch"]

    monkeypatch.setattr(bot_module.pipeline, "cache", cache)
    monkeypatch.setattr(bot_module.pipeline, "memory_worker", memory_worker)
    monkeypatch.setattr(bot_module.pipeline, "llm", llm)
    monkeypatch.setattr(bot_module.pipeline, "vector_memory", vector_memory)
    monkeypatch.setattr(
        bot_module.pipeline,
        "describe_attachments",
        AsyncMock(return_value=AttachmentProcessingResult(descriptions=image_descriptions)),
    )
    monkeypatch.setattr(bot_module.pipeline, "send_reply", send_reply)

    await bot_module.on_message(message)

    assert len(cache.added) == 1
    cached_message = cache.added[0]
    assert cached_message.content == "[friend pasted an image into the chat]\n[Image: a cat sleeping on a couch]"
    vector_memory.query.assert_awaited_once_with(
        "[friend pasted an image into the chat]\n[Image: a cat sleeping on a couch]",
        server_id=message.guild.id,
    )
    assert memory_worker.calls == [(message, image_descriptions)]


@pytest.mark.asyncio
async def test_attachment_fallbacks_are_injected_when_image_cannot_be_inspected(bot_module, monkeypatch):
    message = make_message(content="")
    cache = FakeCache()
    memory_worker = FakeMemoryWorker()
    fallback = "attached image could not be inspected because the file was too large"
    llm = SimpleNamespace(
        ask_bouncer=AsyncMock(
            return_value=SimpleNamespace(
                should_respond=True,
                use_tool=False,
                recommended_tool=None,
                tool_parameters=None,
            )
        ),
        ask_brain=AsyncMock(return_value=BrainResponse(content="can't see it", done_reason="stop")),
    )
    vector_memory = SimpleNamespace(query=AsyncMock(return_value=""))
    send_reply = AsyncMock(return_value=1)

    monkeypatch.setattr(bot_module.pipeline, "cache", cache)
    monkeypatch.setattr(bot_module.pipeline, "memory_worker", memory_worker)
    monkeypatch.setattr(bot_module.pipeline, "llm", llm)
    monkeypatch.setattr(bot_module.pipeline, "vector_memory", vector_memory)
    monkeypatch.setattr(
        bot_module.pipeline,
        "describe_attachments",
        AsyncMock(
            return_value=AttachmentProcessingResult(
                descriptions=[fallback],
                fallback_count=1,
                fallback_reasons=["oversized"],
            )
        ),
    )
    monkeypatch.setattr(bot_module.pipeline, "send_reply", send_reply)

    await bot_module.on_message(message)

    cached_message = cache.added[0]
    assert cached_message.content == (
        "[friend pasted an image into the chat]\n"
        "[Image: attached image could not be inspected because the file was too large]"
    )
    assert memory_worker.calls == [(message, [fallback])]


@pytest.mark.asyncio
async def test_steam_tool_skips_rag_query(bot_module, monkeypatch):
    message = make_message(content="what's good on steam?")
    cache = FakeCache()
    memory_worker = FakeMemoryWorker()
    llm = SimpleNamespace(
        ask_bouncer=AsyncMock(
            return_value=SimpleNamespace(
                should_respond=True,
                use_tool=True,
                recommended_tool="steam_browse",
                tool_parameters={"category": "top_sellers"},
                reason="steam question",
            )
        ),
        ask_brain=AsyncMock(return_value=BrainResponse(content="steam answer", done_reason="stop")),
    )
    vector_memory = SimpleNamespace(query=AsyncMock(return_value="stale rag"))
    tools = SimpleNamespace(
        KNOWN_TOOLS=frozenset({"steam_browse"}),
        dispatch=AsyncMock(return_value="Steam Top Sellers:\n\n1. Hit Game"),
    )
    send_reply = AsyncMock(return_value=1)

    monkeypatch.setattr(bot_module.pipeline, "cache", cache)
    monkeypatch.setattr(bot_module.pipeline, "memory_worker", memory_worker)
    monkeypatch.setattr(bot_module.pipeline, "llm", llm)
    monkeypatch.setattr(bot_module.pipeline, "vector_memory", vector_memory)
    monkeypatch.setattr(bot_module.pipeline, "tools_module", tools)
    monkeypatch.setattr(
        bot_module.pipeline,
        "describe_attachments",
        AsyncMock(return_value=AttachmentProcessingResult(descriptions=[])),
    )
    monkeypatch.setattr(bot_module.pipeline, "send_reply", send_reply)

    await bot_module.on_message(message)

    vector_memory.query.assert_not_awaited()
    assert llm.ask_brain.await_args.kwargs["rag_context"] == ""
    assert "Steam Top Sellers" in llm.ask_brain.await_args.kwargs["tool_context"]
