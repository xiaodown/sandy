from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from sandy.last10 import Last10
from sandy.memory import MemoryClient
from sandy.vector_memory import VectorMemory


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
    assert created_messages[0].discord_message_id == message.id
    llm.ask_tagger.assert_not_awaited()
    llm.ask_summarizer.assert_not_awaited()
    vector_memory.add_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_store_message_returns_false_when_db_insert_fails():
    db = SimpleNamespace(create_message=lambda message: (_ for _ in ()).throw(RuntimeError("db down")))
    client = MemoryClient(db=db)

    with pytest.raises(RuntimeError, match="db down"):
        await client.store_message(make_message())


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


@pytest.mark.asyncio
async def test_process_and_store_keeps_recall_and_vector_calls_server_scoped():
    created_messages = []
    db = SimpleNamespace(create_message=lambda message: created_messages.append(message))
    vector_memory = SimpleNamespace(add_message=AsyncMock())
    client = MemoryClient(db=db, vector_memory=vector_memory)
    message = DummyMessage(
        content="server local",
        guild=DummyGuild(id=777, name="Private Guild"),
        channel=DummyChannel(id=888, name="secret"),
    )

    await client.process_and_store(message)

    assert created_messages[0].server_id == 777
    assert created_messages[0].discord_message_id == message.id
    assert created_messages[0].server_name == "Private Guild"
    assert created_messages[0].channel_id == 888
    assert created_messages[0].channel_name == "secret"
    vector_memory.add_message.assert_awaited_once_with(
        message_id=str(message.id),
        content="server local",
        author_name=message.author.display_name,
        server_id=777,
        timestamp=message.created_at,
    )


@pytest.mark.asyncio
async def test_process_and_store_continues_without_tags_if_tagger_fails():
    created_messages = []
    db = SimpleNamespace(create_message=lambda message: created_messages.append(message))
    llm = SimpleNamespace(
        ask_tagger=AsyncMock(side_effect=RuntimeError("tagger down")),
        ask_summarizer=AsyncMock(),
    )
    vector_memory = SimpleNamespace(add_message=AsyncMock())
    client = MemoryClient(db=db, llm=llm, vector_memory=vector_memory)

    message = make_message(content="hello")
    await client.process_and_store(message)

    assert created_messages[0].tags is None
    assert created_messages[0].summary is None
    vector_memory.add_message.assert_awaited_once_with(
        message_id=str(message.id),
        content="hello",
        author_name=message.author.display_name,
        server_id=message.guild.id,
        timestamp=message.created_at,
    )


@pytest.mark.asyncio
async def test_process_and_store_continues_without_summary_if_summarizer_fails():
    created_messages = []
    db = SimpleNamespace(create_message=lambda message: created_messages.append(message))
    llm = SimpleNamespace(
        ask_tagger=AsyncMock(return_value=["tag"]),
        ask_summarizer=AsyncMock(side_effect=RuntimeError("summarizer down")),
    )
    vector_memory = SimpleNamespace(add_message=AsyncMock())
    client = MemoryClient(db=db, llm=llm, vector_memory=vector_memory)
    client.SUMMARIZE_THRESHOLD = 1

    message = make_message(content="long enough")
    await client.process_and_store(message)

    assert created_messages[0].tags == ["tag"]
    assert created_messages[0].summary is None
    vector_memory.add_message.assert_awaited_once_with(
        message_id=str(message.id),
        content="long enough",
        author_name=message.author.display_name,
        server_id=message.guild.id,
        timestamp=message.created_at,
    )


@pytest.mark.asyncio
async def test_process_and_store_continues_to_vector_when_recall_store_fails():
    db = SimpleNamespace(create_message=lambda message: (_ for _ in ()).throw(RuntimeError("db down")))
    vector_memory = SimpleNamespace(add_message=AsyncMock())
    client = MemoryClient(db=db, vector_memory=vector_memory)

    await client.process_and_store(make_message(content="still embed me"))

    vector_memory.add_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_process_and_store_raises_after_vector_failure_even_if_recall_succeeds():
    created_messages = []
    db = SimpleNamespace(create_message=lambda message: created_messages.append(message))
    vector_memory = SimpleNamespace(add_message=AsyncMock(side_effect=RuntimeError("vector down")))
    client = MemoryClient(db=db, vector_memory=vector_memory)

    with pytest.raises(RuntimeError, match="vector down"):
        await client.process_and_store(make_message(content="store then explode"))

    assert len(created_messages) == 1


@pytest.mark.asyncio
async def test_process_and_store_raises_if_both_vector_and_recall_fail():
    db = SimpleNamespace(create_message=lambda message: (_ for _ in ()).throw(RuntimeError("recall down")))
    vector_memory = SimpleNamespace(add_message=AsyncMock(side_effect=RuntimeError("vector down")))
    client = MemoryClient(db=db, vector_memory=vector_memory)

    with pytest.raises(RuntimeError, match="vector down"):
        await client.process_and_store(make_message(content="lose both"))


@pytest.mark.asyncio
async def test_process_and_store_formats_multiple_images_for_recall_and_vector():
    created_messages = []
    db = SimpleNamespace(create_message=lambda message: created_messages.append(message))
    vector_memory = SimpleNamespace(add_message=AsyncMock())
    client = MemoryClient(db=db, vector_memory=vector_memory)
    message = make_message(content="look at this")

    await client.process_and_store(
        message,
        image_descriptions=["cat sleeping", "dog judging"],
    )

    expected = "look at this  [Image 1: cat sleeping]  [Image 2: dog judging]"
    assert created_messages[0].content == expected
    vector_memory.add_message.assert_awaited_once_with(
        message_id=str(message.id),
        content=expected,
        author_name=message.author.display_name,
        server_id=message.guild.id,
        timestamp=message.created_at,
    )


@pytest.mark.asyncio
async def test_vector_query_passes_server_filter_to_chroma():
    recorded = {}
    vector_memory = VectorMemory.__new__(VectorMemory)
    vector_memory._embed_client = SimpleNamespace(
        embed=AsyncMock(return_value=SimpleNamespace(embeddings=[[0.1, 0.2, 0.3]]))
    )
    vector_memory._collection = SimpleNamespace(
        count=lambda: 3,
        query=lambda **kwargs: recorded.update(kwargs) or {
            "documents": [["private message"]],
            "metadatas": [[{"author_name": "friend", "timestamp": "2026-03-13T12:00:00+00:00"}]],
            "distances": [[0.2]],
        },
    )

    result = await VectorMemory.query(vector_memory, "secret topic", server_id=4242, n_results=5)

    assert "<friend>: private message" in result
    assert recorded["where"] == {"server_id": 4242}
    assert recorded["n_results"] == 3
