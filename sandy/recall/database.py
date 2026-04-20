"""
Recall database — SQLite storage for Sandy's long-term message history.

Previously served via FastAPI; now imported directly by memory.py and tools.py.
"""

import sqlite3
import json
import os
from datetime import UTC, datetime
from typing import Any
from contextlib import contextmanager

from .models import (
    ChatMessageCreate,
    ChatMessageResponse,
    DeferredMessageCreate,
    DeferredMessageResponse,
)

class ChatDatabase:
    """SQLite database handler for chat messages."""

    CURRENT_SCHEMA_VERSION = 5

    def __init__(self, db_path: str = "data/recall.db"):
        """Initialize database connection.

        db_path — path to the SQLite database file.  The directory is
        created automatically if it doesn't exist.
        """
        self.db_path = db_path

    @contextmanager
    def get_connection(self):
        """Context manager for database connections."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
        finally:
            conn.close()

    def init_db(self):
        """Initialize the database schema and run any needed migrations."""
        self.migrate_to_latest()

    def get_schema_version(self) -> int:
        """Get the current schema version from the database."""
        with self.get_connection() as conn:
            try:
                cursor = conn.execute("SELECT version FROM schema_version LIMIT 1")
                row = cursor.fetchone()
                return row[0] if row else 0
            except sqlite3.OperationalError:
                return 0

    def set_schema_version(self, version: int):
        """Set the schema version in the database."""
        with self.get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS schema_version (
                    version INTEGER PRIMARY KEY
                )
            """)
            conn.execute("DELETE FROM schema_version")
            conn.execute("INSERT INTO schema_version (version) VALUES (?)", (version,))
            conn.commit()

    def migrate_to_latest(self):
        """Run all necessary migrations to get to the current schema version."""
        current_version = self.get_schema_version()

        if current_version < self.CURRENT_SCHEMA_VERSION:
            print(f"Migrating database from version {current_version} to {self.CURRENT_SCHEMA_VERSION}...")

            if current_version < 1:
                self._migrate_v1_create_initial_schema()
                self.set_schema_version(1)
                print("✓ Migrated to version 1: Created initial schema")

            if current_version < 2:
                self._migrate_v2_ids_and_tags_table()
                self.set_schema_version(2)
                print("✓ Migrated to version 2: Added snowflake ID columns, proper tags table")

            if current_version < 3:
                self._migrate_v3_fts()
                self.set_schema_version(3)
                print("✓ Migrated to version 3: Added FTS5 full-text search index")

            if current_version < 4:
                self._migrate_v4_discord_message_ids()
                self.set_schema_version(4)
                print("✓ Migrated to version 4: Added Discord message snowflake column")

            if current_version < 5:
                self._migrate_v5_deferred_message_queue()
                self.set_schema_version(5)
                print("✓ Migrated to version 5: Added deferred message queue")

    def _migrate_v1_create_initial_schema(self):
        """Migration v1: Create the original name-based schema (kept for upgrade path)."""
        with self.get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    author TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    server TEXT NOT NULL,
                    channel TEXT NOT NULL,
                    tags TEXT,
                    summary TEXT
                )
            """)
            conn.commit()

    def _migrate_v2_ids_and_tags_table(self):
        """Migration v2: Replace name-only columns with ID+name pairs; proper tags table.

        If upgrading from v1, the old name columns are preserved as the *_name columns and
        *_id columns are set to 0 as a sentinel (they were never stored in v1).
        Fresh installs skip v1 entirely and go straight to this schema.
        """
        with self.get_connection() as conn:
            # Check if we're upgrading an existing v1 table or starting fresh
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='chat_messages'"
            )
            table_exists = cursor.fetchone() is not None

            if table_exists:
                # Upgrade path: rename old table, create new one, copy data, drop old
                conn.execute("ALTER TABLE chat_messages RENAME TO chat_messages_v1")
                self._create_v2_tables(conn)
                conn.execute("""
                    INSERT INTO chat_messages
                        (author_id, author_name, channel_id, channel_name,
                         server_id, server_name, content, timestamp, summary)
                    SELECT
                        0, author,
                        0, channel,
                        0, server,
                        content, timestamp, summary
                    FROM chat_messages_v1
                """)
                # Migrate old JSON tags into the new tags tables
                old_rows = conn.execute(
                    "SELECT id, tags FROM chat_messages_v1 WHERE tags IS NOT NULL"
                ).fetchall()
                new_ids = conn.execute(
                    "SELECT id FROM chat_messages ORDER BY id"
                ).fetchall()
                for old_row, new_id_row in zip(old_rows, new_ids):
                    try:
                        tag_list = json.loads(old_row["tags"])
                        self._insert_tags(conn, new_id_row["id"], tag_list)
                    except (json.JSONDecodeError, TypeError):
                        pass
                conn.execute("DROP TABLE chat_messages_v1")
            else:
                self._create_v2_tables(conn)

            conn.commit()

    def _migrate_v3_fts(self):
        """Migration v3: Add FTS5 virtual table for full-text content search.

        Uses the porter tokenizer for basic English stemming (handles regular
        morphology like play/playing/played) plus unicode61 for proper unicode
        handling.  Irregular forms (mouse/mice) are NOT unified by porter —
        that would require a full lemmatizer.

        A content FTS5 table is used (content="chat_messages") so there is no
        data duplication; triggers keep it in sync with the main table.
        """
        with self.get_connection() as conn:
            conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
                    content,
                    summary,
                    content="chat_messages",
                    content_rowid="id",
                    tokenize="porter unicode61"
                )
            """)
            # Populate from all existing messages
            conn.execute("""
                INSERT INTO messages_fts(rowid, content, summary)
                SELECT id, content, COALESCE(summary, '') FROM chat_messages
            """)
            # Keep FTS in sync via triggers
            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS messages_ai
                AFTER INSERT ON chat_messages BEGIN
                    INSERT INTO messages_fts(rowid, content, summary)
                    VALUES (new.id, new.content, COALESCE(new.summary, ''));
                END
            """)
            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS messages_ad
                AFTER DELETE ON chat_messages BEGIN
                    INSERT INTO messages_fts(messages_fts, rowid, content, summary)
                    VALUES ('delete', old.id, old.content, COALESCE(old.summary, ''));
                END
            """)
            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS messages_au
                AFTER UPDATE ON chat_messages BEGIN
                    INSERT INTO messages_fts(messages_fts, rowid, content, summary)
                    VALUES ('delete', old.id, old.content, COALESCE(old.summary, ''));
                    INSERT INTO messages_fts(rowid, content, summary)
                    VALUES (new.id, new.content, COALESCE(new.summary, ''));
                END
            """)
            conn.commit()

    def _migrate_v4_discord_message_ids(self):
        """Migration v4: add optional Discord message snowflake tracking."""
        with self.get_connection() as conn:
            columns = {
                row["name"]
                for row in conn.execute("PRAGMA table_info(chat_messages)").fetchall()
            }
            if "discord_message_id" not in columns:
                conn.execute(
                    "ALTER TABLE chat_messages ADD COLUMN discord_message_id INTEGER"
                )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_discord_message_id ON chat_messages(discord_message_id)"
            )
            conn.commit()

    def _migrate_v5_deferred_message_queue(self):
        """Migration v5: add a staging queue for text messages deferred during VC."""
        with self.get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS deferred_message_queue (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    discord_message_id INTEGER NOT NULL UNIQUE,
                    author_id INTEGER NOT NULL,
                    author_name TEXT NOT NULL DEFAULT '',
                    channel_id INTEGER NOT NULL,
                    channel_name TEXT NOT NULL DEFAULT '',
                    server_id INTEGER NOT NULL,
                    server_name TEXT NOT NULL DEFAULT '',
                    content TEXT NOT NULL DEFAULT '',
                    timestamp DATETIME NOT NULL,
                    attachment_payload_json TEXT,
                    queued_at DATETIME NOT NULL,
                    attempt_count INTEGER NOT NULL DEFAULT 0,
                    last_error TEXT
                )
            """)
            for idx_sql in [
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_deferred_discord_message_id ON deferred_message_queue(discord_message_id)",
                "CREATE INDEX IF NOT EXISTS idx_deferred_queued_at ON deferred_message_queue(queued_at)",
                "CREATE INDEX IF NOT EXISTS idx_deferred_server_channel ON deferred_message_queue(server_id, channel_id)",
            ]:
                conn.execute(idx_sql)
            conn.commit()

    def _create_v2_tables(self, conn: sqlite3.Connection):
        """Create the v2 schema tables (called by migration; reuses an open connection)."""
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chat_messages (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                discord_message_id INTEGER,
                author_id   INTEGER NOT NULL,
                author_name TEXT    NOT NULL DEFAULT '',
                channel_id  INTEGER NOT NULL,
                channel_name TEXT   NOT NULL DEFAULT '',
                server_id   INTEGER NOT NULL,
                server_name TEXT    NOT NULL DEFAULT '',
                content     TEXT    NOT NULL,
                timestamp   DATETIME NOT NULL,
                summary     TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS tags (
                id   INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS message_tags (
                message_id INTEGER NOT NULL,
                tag_id     INTEGER NOT NULL,
                PRIMARY KEY (message_id, tag_id),
                FOREIGN KEY (message_id) REFERENCES chat_messages(id) ON DELETE CASCADE,
                FOREIGN KEY (tag_id)     REFERENCES tags(id)
            )
        """)
        # Indexes for common query patterns
        for idx_sql in [
            "CREATE INDEX IF NOT EXISTS idx_discord_message_id ON chat_messages(discord_message_id)",
            "CREATE INDEX IF NOT EXISTS idx_author_id   ON chat_messages(author_id)",
            "CREATE INDEX IF NOT EXISTS idx_author_name ON chat_messages(author_name)",
            "CREATE INDEX IF NOT EXISTS idx_server_id   ON chat_messages(server_id)",
            "CREATE INDEX IF NOT EXISTS idx_server_name ON chat_messages(server_name)",
            "CREATE INDEX IF NOT EXISTS idx_channel_id  ON chat_messages(channel_id)",
            "CREATE INDEX IF NOT EXISTS idx_channel_name ON chat_messages(channel_name)",
            "CREATE INDEX IF NOT EXISTS idx_timestamp   ON chat_messages(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_server_channel_ts ON chat_messages(server_id, channel_id, timestamp DESC)",
        ]:
            conn.execute(idx_sql)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _insert_tags(self, conn: sqlite3.Connection, message_id: int, tags: list[str]):
        """Insert tags and link them to a message (within an open connection)."""
        for tag in tags:
            tag = tag.strip().lower()
            if not tag:
                continue
            conn.execute("INSERT OR IGNORE INTO tags (name) VALUES (?)", (tag,))
            tag_id = conn.execute("SELECT id FROM tags WHERE name = ?", (tag,)).fetchone()["id"]
            conn.execute(
                "INSERT OR IGNORE INTO message_tags (message_id, tag_id) VALUES (?, ?)",
                (message_id, tag_id)
            )

    def _get_tags_for_message(self, conn: sqlite3.Connection, message_id: int) -> list[str]:
        """Fetch tag names for a given message ID."""
        rows = conn.execute("""
            SELECT t.name FROM tags t
            JOIN message_tags mt ON t.id = mt.tag_id
            WHERE mt.message_id = ?
            ORDER BY t.name
        """, (message_id,)).fetchall()
        return [row["name"] for row in rows]

    def _row_to_response(self, row: sqlite3.Row, conn: sqlite3.Connection) -> ChatMessageResponse:
        """Convert a database row to a ChatMessageResponse."""
        tags = self._get_tags_for_message(conn, row["id"])
        timestamp = datetime.fromisoformat(row["timestamp"].replace("Z", "+00:00"))
        return ChatMessageResponse(
            id=row["id"],
            discord_message_id=row["discord_message_id"],
            author_id=row["author_id"],
            author_name=row["author_name"],
            channel_id=row["channel_id"],
            channel_name=row["channel_name"],
            server_id=row["server_id"],
            server_name=row["server_name"],
            content=row["content"],
            timestamp=timestamp,
            tags=tags if tags else None,
            summary=row["summary"],
        )

    def _deferred_row_to_response(self, row: sqlite3.Row) -> DeferredMessageResponse:
        """Convert a deferred queue row to a DeferredMessageResponse."""
        timestamp = datetime.fromisoformat(row["timestamp"].replace("Z", "+00:00"))
        queued_at = datetime.fromisoformat(row["queued_at"].replace("Z", "+00:00"))
        attachment_payload = None
        if row["attachment_payload_json"]:
            attachment_payload = json.loads(row["attachment_payload_json"])
        return DeferredMessageResponse(
            id=row["id"],
            discord_message_id=row["discord_message_id"],
            author_id=row["author_id"],
            author_name=row["author_name"],
            channel_id=row["channel_id"],
            channel_name=row["channel_name"],
            server_id=row["server_id"],
            server_name=row["server_name"],
            content=row["content"],
            timestamp=timestamp,
            attachment_payload=attachment_payload,
            queued_at=queued_at,
            attempt_count=row["attempt_count"],
            last_error=row["last_error"],
        )

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def create_message(self, message: ChatMessageCreate) -> int:
        """Create a new chat message and return its ID."""
        with self.get_connection() as conn:
            cursor = conn.execute("""
                INSERT INTO chat_messages
                    (discord_message_id, author_id, author_name, channel_id, channel_name,
                     server_id, server_name, content, timestamp, summary)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                message.discord_message_id,
                message.author_id, message.author_name,
                message.channel_id, message.channel_name,
                message.server_id, message.server_name,
                message.content,
                message.timestamp.isoformat(),
                message.summary,
            ))
            message_id = cursor.lastrowid
            if message.tags:
                self._insert_tags(conn, message_id, message.tags)
            conn.commit()
            return message_id

    def get_message(self, message_id: int) -> ChatMessageResponse | None:
        """Get a specific message by ID."""
        with self.get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM chat_messages WHERE id = ?", (message_id,)
            ).fetchone()
            if row:
                return self._row_to_response(row, conn)
            return None

    def get_messages(
        self,
        limit: int = 100,
        offset: int = 0,
        author_id: int | None = None,
        author_name: str | None = None,
        discord_message_id: int | None = None,
        server_id: int | None = None,
        server_name: str | None = None,
        channel_id: int | None = None,
        channel_name: str | None = None,
        tag: str | None = None,
        q: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        hours_ago: int | None = None,
        minutes_ago: int | None = None,
    ) -> list[ChatMessageResponse]:
        """Get messages with optional filtering. ID filters take precedence over name filters.

        hours_ago / minutes_ago — convenience shortcuts that override ``since``
            with a relative time offset from now (UTC).
        q — full-text search against message content and summary via FTS5.
            Uses porter stemming; phrase queries are wrapped in double-quotes
            automatically so arbitrary user input is safe to pass directly.
        """
        # Convert convenience time params into a since datetime
        if hours_ago is not None:
            from datetime import timedelta, timezone as _tz
            since = datetime.now(_tz.utc) - timedelta(hours=hours_ago)
        elif minutes_ago is not None:
            from datetime import timedelta, timezone as _tz
            since = datetime.now(_tz.utc) - timedelta(minutes=minutes_ago)

        with self.get_connection() as conn:
            query = "SELECT cm.* FROM chat_messages cm WHERE 1=1"
            params: list = []

            # ID filters are exact and fast; name filters are fallback / conveniences
            if author_id is not None:
                query += " AND cm.author_id = ?"
                params.append(author_id)
            elif author_name:
                query += " AND cm.author_name = ?"
                params.append(author_name)

            if discord_message_id is not None:
                query += " AND cm.discord_message_id = ?"
                params.append(discord_message_id)

            if server_id is not None:
                query += " AND cm.server_id = ?"
                params.append(server_id)
            elif server_name:
                query += " AND cm.server_name = ?"
                params.append(server_name)

            if channel_id is not None:
                query += " AND cm.channel_id = ?"
                params.append(channel_id)
            elif channel_name:
                query += " AND cm.channel_name = ?"
                params.append(channel_name)

            if tag:
                # LIKE-based substring match: searching "game" finds "gaming", "games", etc.
                # Tags are normalised to lowercase on insert so case is already handled.
                tag_normalized = tag.strip().lower()
                query += """
                    AND cm.id IN (
                        SELECT mt.message_id FROM message_tags mt
                        JOIN tags t ON mt.tag_id = t.id
                        WHERE t.name LIKE ?
                    )
                """
                params.append(f"%{tag_normalized}%")

            if q:
                # Pass the query through to FTS5 as-is so boolean operators work.
                # 'Rob OR Robst', 'memory AND system', plain words, all valid FTS5.
                # Queries come from the LLM, not raw user input, so operator syntax
                # is intentional. Strip bare double-quotes that could cause a parse
                # error if unbalanced.
                fts_query = q.replace('"', '')
                query += """
                    AND cm.id IN (
                        SELECT rowid FROM messages_fts WHERE messages_fts MATCH ?
                    )
                """
                params.append(fts_query)

            if since:
                query += " AND cm.timestamp >= ?"
                params.append(since.isoformat())

            if until:
                query += " AND cm.timestamp <= ?"
                params.append(until.isoformat())

            query += " ORDER BY cm.timestamp DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])

            rows = conn.execute(query, params).fetchall()
            return [self._row_to_response(row, conn) for row in rows]

    def get_message_by_discord_id(self, discord_message_id: int) -> ChatMessageResponse | None:
        """Get a specific message by its original Discord snowflake, if stored."""
        with self.get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM chat_messages WHERE discord_message_id = ?",
                (discord_message_id,),
            ).fetchone()
            if row:
                return self._row_to_response(row, conn)
            return None

    def enqueue_deferred_message(self, message: DeferredMessageCreate) -> int:
        """Insert a deferred queue row and return its queue ID."""
        with self.get_connection() as conn:
            cursor = conn.execute("""
                INSERT INTO deferred_message_queue
                    (discord_message_id, author_id, author_name, channel_id, channel_name,
                     server_id, server_name, content, timestamp, attachment_payload_json, queued_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                message.discord_message_id,
                message.author_id, message.author_name,
                message.channel_id, message.channel_name,
                message.server_id, message.server_name,
                message.content,
                message.timestamp.isoformat(),
                json.dumps(message.attachment_payload) if message.attachment_payload is not None else None,
                datetime.now(UTC).isoformat(),
            ))
            conn.commit()
            return cursor.lastrowid

    def get_deferred_messages(self, *, limit: int = 100) -> list[DeferredMessageResponse]:
        """Return deferred queue rows ordered oldest-first."""
        with self.get_connection() as conn:
            rows = conn.execute("""
                SELECT * FROM deferred_message_queue
                ORDER BY queued_at ASC, id ASC
                LIMIT ?
            """, (limit,)).fetchall()
            return [self._deferred_row_to_response(row) for row in rows]

    def delete_deferred_message(self, queue_id: int) -> bool:
        """Delete one deferred queue row."""
        with self.get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM deferred_message_queue WHERE id = ?",
                (queue_id,),
            )
            conn.commit()
            return cursor.rowcount > 0

    def record_deferred_message_failure(self, queue_id: int, error: str) -> bool:
        """Increment attempts and store the latest error for one deferred row."""
        with self.get_connection() as conn:
            cursor = conn.execute("""
                UPDATE deferred_message_queue
                SET attempt_count = attempt_count + 1,
                    last_error = ?
                WHERE id = ?
            """, (error, queue_id))
            conn.commit()
            return cursor.rowcount > 0

    def delete_message(self, message_id: int) -> bool:
        """Delete a message by ID. Returns True if deleted, False if not found."""
        with self.get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM chat_messages WHERE id = ?", (message_id,)
            )
            conn.commit()
            return cursor.rowcount > 0

    def get_stats(self) -> dict[str, Any]:
        """Get basic statistics about stored messages."""
        with self.get_connection() as conn:
            total = conn.execute(
                "SELECT COUNT(*) as total FROM chat_messages"
            ).fetchone()["total"]

            unique_authors = conn.execute(
                "SELECT COUNT(DISTINCT author_id) as n FROM chat_messages"
            ).fetchone()["n"]

            unique_servers = conn.execute(
                "SELECT COUNT(DISTINCT server_id) as n FROM chat_messages"
            ).fetchone()["n"]

            latest_row = conn.execute(
                "SELECT MAX(timestamp) as latest FROM chat_messages"
            ).fetchone()
            latest = latest_row["latest"] if latest_row["latest"] else None

            total_tags = conn.execute(
                "SELECT COUNT(*) as n FROM tags"
            ).fetchone()["n"]

            return {
                "total_messages": total,
                "unique_authors": unique_authors,
                "unique_servers": unique_servers,
                "total_tags": total_tags,
                "latest_message_time": latest,
            }
