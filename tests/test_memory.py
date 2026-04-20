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
async def test_enqueue_deferred_message_captures_attachment_metadata():
    deferred_messages = []
    db = SimpleNamespace(enqueue_deferred_message=lambda message: deferred_messages.append(message) or 1)
    client = MemoryClient(db=db)
    message = make_message(content="look later")
    message.attachments = [
        SimpleNamespace(
            filename="puppy.png",
            content_type="image/png",
            size=4242,
            url="https://cdn.test/puppy.png",
            proxy_url="https://proxy.test/puppy.png",
            width=320,
            height=200,
        )
    ]

    queue_id = await client.enqueue_deferred_message(message)

    assert queue_id == 1
    assert deferred_messages[0].discord_message_id == message.id
    assert deferred_messages[0].attachment_payload == [{
        "filename": "puppy.png",
        "content_type": "image/png",
        "size_bytes": 4242,
        "url": "https://cdn.test/puppy.png",
        "proxy_url": "https://proxy.test/puppy.png",
        "width": 320,
        "height": 200,
    }]


@pytest.mark.asyncio
async def test_drain_deferred_messages_stores_recall_and_vector_then_deletes_queue_row():
    deleted_ids = []
    queued_row = SimpleNamespace(
        id=7,
        discord_message_id=888,
        author_id=1,
        author_name="alice",
        channel_id=2,
        channel_name="general",
        server_id=3,
        server_name="Guild",
        content="queued later",
        timestamp=datetime.now(UTC),
        attachment_payload=None,
    )
    db = SimpleNamespace(
        get_deferred_messages=lambda limit=100: [queued_row],
        create_message=lambda message: None,
        delete_deferred_message=lambda queue_id: deleted_ids.append(queue_id) or True,
        record_deferred_message_failure=lambda queue_id, error: False,
    )
    llm = SimpleNamespace(
        ask_tagger=AsyncMock(return_value=["tag"]),
        ask_summarizer=AsyncMock(return_value=None),
    )
    vector_memory = SimpleNamespace(add_message=AsyncMock(return_value=True))
    client = MemoryClient(db=db, llm=llm, vector_memory=vector_memory)

    processed = await client.drain_deferred_messages()

    assert processed == 1
    assert deleted_ids == [7]
    vector_memory.add_message.assert_awaited_once_with(
        message_id="888",
        content="queued later",
        author_name="alice",
        server_id=3,
        timestamp=queued_row.timestamp,
    )


@pytest.mark.asyncio
async def test_drain_deferred_messages_keeps_queue_row_when_vector_fails_after_recall():
    deleted_ids = []
    failure_records = []
    created_messages = []
    queued_row = SimpleNamespace(
        id=7,
        discord_message_id=888,
        author_id=1,
        author_name="alice",
        channel_id=2,
        channel_name="general",
        server_id=3,
        server_name="Guild",
        content="queued later",
        timestamp=datetime.now(UTC),
        attachment_payload=None,
    )
    db = SimpleNamespace(
        get_deferred_messages=lambda limit=100: [queued_row],
        create_message=lambda message: created_messages.append(message),
        get_message_by_discord_id=lambda discord_message_id: created_messages[0] if created_messages else None,
        delete_deferred_message=lambda queue_id: deleted_ids.append(queue_id) or True,
        record_deferred_message_failure=lambda queue_id, error: failure_records.append((queue_id, error)) or True,
    )
    llm = SimpleNamespace(
        ask_tagger=AsyncMock(return_value=["tag"]),
        ask_summarizer=AsyncMock(return_value=None),
    )
    vector_memory = SimpleNamespace(add_message=AsyncMock(side_effect=RuntimeError("vector down")))
    client = MemoryClient(db=db, llm=llm, vector_memory=vector_memory)

    processed = await client.drain_deferred_messages(limit=1)

    assert processed == 0
    assert deleted_ids == []
    assert created_messages[0].discord_message_id == 888
    assert failure_records == [(7, "vector down")]


@pytest.mark.asyncio
async def test_drain_deferred_messages_retries_vector_without_duplicate_recall_insert():
    deleted_ids = []
    created_messages = []
    failure_records = []
    queued_row = SimpleNamespace(
        id=7,
        discord_message_id=888,
        author_id=1,
        author_name="alice",
        channel_id=2,
        channel_name="general",
        server_id=3,
        server_name="Guild",
        content="queued later",
        timestamp=datetime.now(UTC),
        attachment_payload=None,
    )
    calls = {"count": 0}

    def get_rows(limit=100):
        return [queued_row] if calls["count"] < 2 else []

    def get_message_by_discord_id(discord_message_id):
        return created_messages[0] if created_messages else None

    async def add_message(**kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("vector down")
        return True

    db = SimpleNamespace(
        get_deferred_messages=get_rows,
        create_message=lambda message: created_messages.append(message),
        get_message_by_discord_id=get_message_by_discord_id,
        delete_deferred_message=lambda queue_id: deleted_ids.append(queue_id) or True,
        record_deferred_message_failure=lambda queue_id, error: failure_records.append((queue_id, error)) or True,
    )
    llm = SimpleNamespace(
        ask_tagger=AsyncMock(return_value=["tag"]),
        ask_summarizer=AsyncMock(return_value=None),
    )
    vector_memory = SimpleNamespace(add_message=AsyncMock(side_effect=add_message))
    client = MemoryClient(db=db, llm=llm, vector_memory=vector_memory)

    first_processed = await client.drain_deferred_messages(limit=1)
    second_processed = await client.drain_deferred_messages(limit=1)

    assert first_processed == 0
    assert second_processed == 1
    assert len(created_messages) == 1
    assert deleted_ids == [7]
    assert failure_records == [(7, "vector down")]


@pytest.mark.asyncio
async def test_drain_deferred_messages_drops_image_only_row_when_attachment_is_gone():
    deleted_ids = []
    queued_row = SimpleNamespace(
        id=7,
        discord_message_id=888,
        author_id=1,
        author_name="alice",
        channel_id=2,
        channel_name="general",
        server_id=3,
        server_name="Guild",
        content="",
        timestamp=datetime.now(UTC),
        attachment_payload=[{"filename": "puppy.png", "url": "https://cdn.test/puppy.png"}],
    )
    db = SimpleNamespace(
        get_deferred_messages=lambda limit=100: [queued_row],
        delete_deferred_message=lambda queue_id: deleted_ids.append(queue_id) or True,
        record_deferred_message_failure=lambda queue_id, error: False,
    )
    client = MemoryClient(db=db, llm=SimpleNamespace(), vector_memory=SimpleNamespace())
    client._describe_deferred_attachments = AsyncMock(return_value=None)
    client._process_payload = AsyncMock()

    processed = await client.drain_deferred_messages(limit=1)

    assert processed == 0
    assert deleted_ids == [7]
    client._process_payload.assert_not_awaited()


@pytest.mark.asyncio
async def test_vector_query_passes_server_filter_to_chroma():
    recorded = {}
    vector_memory = VectorMemory.__new__(VectorMemory)
    vector_memory._embed_model = "mxbai-embed-large"
    vector_memory._max_distance = 0.6
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


@pytest.mark.asyncio
async def test_vector_add_message_raises_on_embed_failure():
    vector_memory = VectorMemory.__new__(VectorMemory)
    vector_memory._embed_model = "mxbai-embed-large"
    vector_memory._max_distance = 0.6
    vector_memory._embed_client = SimpleNamespace(
        embed=AsyncMock(side_effect=RuntimeError("embed down"))
    )
    vector_memory._collection = SimpleNamespace(upsert=lambda **kwargs: None)

    with pytest.raises(RuntimeError, match="embed down"):
        await VectorMemory.add_message(
            vector_memory,
            message_id="123",
            content="hello",
            author_name="friend",
            server_id=42,
            timestamp=datetime.now(UTC),
        )
