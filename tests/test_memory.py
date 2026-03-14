from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from sandy.last10 import Last10
from sandy.memory import MemoryClient


@dataclass
class DummyAuthor:
    id: int = 111
    display_name: str = "friend"


@dataclass
class DummyGuild:
    id: int = 42
    name: str = "Test Guild"


@dataclass
class DummyChannel:
    id: int = 99
    name: str = "general"


@dataclass
class DummyMessage:
    id: int = 555
    content: str = "hello world"
    created_at: datetime = datetime.now(UTC)
    author: DummyAuthor = field(default_factory=DummyAuthor)
    guild: DummyGuild = field(default_factory=DummyGuild)
    channel: DummyChannel = field(default_factory=DummyChannel)


def make_message(*, content: str = "hello world") -> DummyMessage:
    return DummyMessage(content=content, created_at=datetime.now(UTC))


@pytest.mark.asyncio
async def test_process_and_store_tags_summarizes_and_embeds_image_context():
    db = SimpleNamespace(create_message=lambda message: None)
    llm = SimpleNamespace(
        ask_tagger=AsyncMock(return_value=["-cats", "—sleeping"]),
        ask_summarizer=AsyncMock(return_value="cat summary"),
    )
    vector_memory = SimpleNamespace(add_message=AsyncMock())
    client = MemoryClient(db=db, llm=llm, vector_memory=vector_memory)
    client.SUMMARIZE_THRESHOLD = 10
    message = make_message(content="")

    await client.process_and_store(message, image_descriptions=["a sleepy cat"])

    llm.ask_tagger.assert_awaited_once_with("[Image: a sleepy cat]")
    llm.ask_summarizer.assert_awaited_once_with("[Image: a sleepy cat]")
    vector_memory.add_message.assert_awaited_once_with(
        message_id=str(message.id),
        content="[Image: a sleepy cat]",
        author_name=message.author.display_name,
        server_id=message.guild.id,
        timestamp=message.created_at,
    )


@pytest.mark.asyncio
async def test_process_and_store_skips_llm_and_vector_for_empty_content():
    created_messages = []
    db = SimpleNamespace(create_message=lambda message: created_messages.append(message))
    llm = SimpleNamespace(
        ask_tagger=AsyncMock(),
        ask_summarizer=AsyncMock(),
    )
    vector_memory = SimpleNamespace(add_message=AsyncMock())
    client = MemoryClient(db=db, llm=llm, vector_memory=vector_memory)
    message = make_message(content="")

    await client.process_and_store(message)

    assert created_messages[0].content == "(no text content)"
    llm.ask_tagger.assert_not_awaited()
    llm.ask_summarizer.assert_not_awaited()
    vector_memory.add_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_store_message_returns_false_when_db_insert_fails():
    db = SimpleNamespace(create_message=lambda message: (_ for _ in ()).throw(RuntimeError("db down")))
    client = MemoryClient(db=db)

    result = await client.store_message(make_message())

    assert result is False


@pytest.mark.asyncio
async def test_seed_cache_adds_messages_oldest_first_per_channel():
    now = datetime.now(UTC)
    rows = [
        SimpleNamespace(
            author_id=1,
            author_name="friend",
            channel_id=100,
            channel_name="general",
            server_id=42,
            server_name="Guild",
            content="newest",
            timestamp=now,
        ),
        SimpleNamespace(
            author_id=1,
            author_name="friend",
            channel_id=100,
            channel_name="general",
            server_id=42,
            server_name="Guild",
            content="older",
            timestamp=now - timedelta(minutes=5),
        ),
        SimpleNamespace(
            author_id=2,
            author_name="other",
            channel_id=200,
            channel_name="random",
            server_id=42,
            server_name="Guild",
            content="side-channel",
            timestamp=now - timedelta(minutes=1),
        ),
    ]
    db = SimpleNamespace(get_messages=lambda **kwargs: rows)
    cache = Last10(maxlen=10)
    client = MemoryClient(db=db)

    seeded = await client.seed_cache(cache, hours=24)

    assert seeded == 3
    general = cache.get(42, 100)
    assert [msg.content for msg in general] == ["older", "newest"]
    random = cache.get(42, 200)
    assert [msg.content for msg in random] == ["side-channel"]
