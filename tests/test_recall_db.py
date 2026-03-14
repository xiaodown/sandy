from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from sandy.recall import ChatDatabase, ChatMessageCreate


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
    assert version == 4


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
