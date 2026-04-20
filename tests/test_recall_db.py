from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from sandy.recall import ChatDatabase, ChatMessageCreate, DeferredMessageCreate


def test_init_db_migrates_v3_database_and_adds_discord_message_id(tmp_path: Path) -> None:
    db_path = tmp_path / "recall.db"
    db = ChatDatabase(str(db_path))

    with db.get_connection() as conn:
        db._create_v2_tables(conn)
        db._migrate_v3_fts()
        conn.execute("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER PRIMARY KEY)")
        conn.execute("DELETE FROM schema_version")
        conn.execute("INSERT INTO schema_version (version) VALUES (3)")
        conn.commit()

    db.init_db()

    with db.get_connection() as conn:
        columns = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(chat_messages)").fetchall()
        }
        version = conn.execute("SELECT version FROM schema_version").fetchone()["version"]

    assert "discord_message_id" in columns
    assert version == 5


def test_create_and_fetch_message_preserves_discord_message_id(tmp_path: Path) -> None:
    db_path = tmp_path / "recall.db"
    db = ChatDatabase(str(db_path))
    db.init_db()

    recall_id = db.create_message(
        ChatMessageCreate(
            discord_message_id=1482282320600891422,
            author_id=1,
            author_name="alice",
            channel_id=2,
            channel_name="general",
            server_id=3,
            server_name="Guild",
            content="hello world",
            timestamp=datetime.now(UTC),
        )
    )

    row = db.get_message(recall_id)
    by_discord = db.get_message_by_discord_id(1482282320600891422)

    assert row is not None
    assert row.discord_message_id == 1482282320600891422
    assert by_discord is not None
    assert by_discord.id == recall_id


def test_init_db_migrates_v4_database_and_adds_deferred_message_queue(tmp_path: Path) -> None:
    db_path = tmp_path / "recall.db"
    db = ChatDatabase(str(db_path))

    with db.get_connection() as conn:
        db._create_v2_tables(conn)
        db._migrate_v3_fts()
        db._migrate_v4_discord_message_ids()
        conn.execute("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER PRIMARY KEY)")
        conn.execute("DELETE FROM schema_version")
        conn.execute("INSERT INTO schema_version (version) VALUES (4)")
        conn.commit()

    db.init_db()

    with db.get_connection() as conn:
        tables = {
            row["name"]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }
        version = conn.execute("SELECT version FROM schema_version").fetchone()["version"]

    assert "deferred_message_queue" in tables
    assert version == 5


def test_enqueue_and_fetch_deferred_messages_round_trip(tmp_path: Path) -> None:
    db = ChatDatabase(str(tmp_path / "recall.db"))
    db.init_db()

    db.enqueue_deferred_message(
        DeferredMessageCreate(
            discord_message_id=999999,
            author_id=1,
            author_name="alice",
            channel_id=2,
            channel_name="general",
            server_id=3,
            server_name="Guild",
            content="later please",
            timestamp=datetime.now(UTC),
            attachment_payload=[{"filename": "puppy.png", "url": "https://cdn.test/puppy.png"}],
        )
    )

    rows = db.get_deferred_messages()

    assert len(rows) == 1
    assert rows[0].discord_message_id == 999999
    assert rows[0].content == "later please"
    assert rows[0].attachment_payload == [{"filename": "puppy.png", "url": "https://cdn.test/puppy.png"}]


def test_record_deferred_message_failure_increments_attempt_count(tmp_path: Path) -> None:
    db = ChatDatabase(str(tmp_path / "recall.db"))
    db.init_db()

    queue_id = db.enqueue_deferred_message(
        DeferredMessageCreate(
            discord_message_id=123456,
            author_id=1,
            author_name="alice",
            channel_id=2,
            channel_name="general",
            server_id=3,
            server_name="Guild",
            content="later please",
            timestamp=datetime.now(UTC),
        )
    )

    assert db.record_deferred_message_failure(queue_id, "boom") is True
    rows = db.get_deferred_messages()

    assert rows[0].attempt_count == 1
    assert rows[0].last_error == "boom"
