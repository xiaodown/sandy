
"""
Class for handling the bot's knowledge of servers
and channels that she's seen, and the ability to
map a channel to a server.
Also stores seen users along with per server
user nicknames.

This is all pretty much so I don't have to spam
the discord API to get this info.

"""

import sqlite3
import os
import logging
import discord
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class Registry:
    """
    Tracks Discord servers (guilds) and channels the bot has seen,
    persisted to a local SQLite database to avoid repeated API calls.
    """

    def __init__(self):
        dbpath = os.getenv("DB_DIR") + os.getenv("SERVER_DB_NAME")
        self.db_path = dbpath
        self._initialize_db()

    def _get_conn(self) -> sqlite3.Connection:
        """Return a connection with foreign keys enabled and row_factory set."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # rows behave like dicts
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _initialize_db(self) -> None:
        with self._get_conn() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS servers (
                    server_id INTEGER PRIMARY KEY,
                    server_name TEXT NOT NULL DEFAULT ''
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS channels (
                    channel_id INTEGER PRIMARY KEY,
                    channel_name TEXT NOT NULL DEFAULT '',
                    server_id INTEGER NOT NULL,
                    FOREIGN KEY (server_id) REFERENCES servers(server_id)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    user_id INTEGER PRIMARY KEY,
                    user_name TEXT NOT NULL DEFAULT ''
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS user_nicknames (
                    user_id INTEGER NOT NULL,
                    server_id INTEGER NOT NULL,
                    nickname TEXT,
                    PRIMARY KEY (user_id, server_id),
                    FOREIGN KEY (user_id) REFERENCES users(user_id),
                    FOREIGN KEY (server_id) REFERENCES servers(server_id)
                )
                """
            )
            conn.commit()

    # ------------------------------------------------------------------
    # Presence checks
    # ------------------------------------------------------------------

    def server_seen(self, message: discord.Message) -> bool:
        """Return True if the guild from this message is already in the database."""
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT 1 FROM servers WHERE server_id = ?",
                (message.guild.id,)
            ).fetchone()
        return row is not None

    def channel_seen(self, message: discord.Message) -> bool:
        """Return True if the channel from this message is already in the database."""
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT 1 FROM channels WHERE channel_id = ?",
                (message.channel.id,)
            ).fetchone()
        return row is not None

    def user_seen(self, message: discord.Message) -> bool:
        """Return True if the author of this message is already in the database."""
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT 1 FROM users WHERE user_id = ?",
                (message.author.id,)
            ).fetchone()
        return row is not None

    # ------------------------------------------------------------------
    # Insertion
    # ------------------------------------------------------------------

    def add_server(self, message: discord.Message) -> None:
        """Record the guild from this message. No-op if already present."""
        with self._get_conn() as conn:
            cursor = conn.execute(
                "INSERT OR IGNORE INTO servers (server_id, server_name) VALUES (?, ?)",
                (message.guild.id, message.guild.name)
            )
            conn.commit()
        if cursor.rowcount:
            logger.info("New server seen: %s (%s)", message.guild.name, message.guild.id)

    def add_channel(self, message: discord.Message) -> None:
        """Record the channel from this message. Also ensures the parent guild is recorded."""
        self.add_server(message)
        with self._get_conn() as conn:
            cursor = conn.execute(
                "INSERT OR IGNORE INTO channels (channel_id, channel_name, server_id) VALUES (?, ?, ?)",
                (message.channel.id, message.channel.name, message.guild.id)
            )
            conn.commit()
        if cursor.rowcount:
            logger.info("New channel seen: #%s in %s", message.channel.name, message.guild.name)

    def add_user(self, message: discord.Message) -> None:
        """Record the author of this message. Stores their global username and
        their server-specific nickname (which may be None). Safe to call repeatedly;
        the nickname will be updated if it has changed."""
        self.add_server(message)  # user_nicknames foreign-keys into servers
        with self._get_conn() as conn:
            cursor = conn.execute(
                "INSERT OR IGNORE INTO users (user_id, user_name) VALUES (?, ?)",
                (message.author.id, message.author.name)
            )
            if cursor.rowcount:
                logger.info("New user seen: %s (%s)", message.author.name, message.author.id)
            # INSERT OR REPLACE so nickname changes are picked up over time
            conn.execute(
                "INSERT OR REPLACE INTO user_nicknames (user_id, server_id, nickname) VALUES (?, ?, ?)",
                (message.author.id, message.guild.id, message.author.nick)
            )
            conn.commit()

    def ensure_seen(self, message: discord.Message) -> None:
        """Record guild, channel, and author from a message if not already known.
        Call this at the top of your on_message handler."""
        self.add_channel(message)  # add_channel already handles the server
        self.add_user(message)

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get_channel_info(self, channel_id: int) -> dict | None:
        """
        Return full channel info as a dict, or None if not found.

        Returned dict keys: channel_id, channel_name, server_id, server_name
        """
        with self._get_conn() as conn:
            row = conn.execute(
                """
                SELECT c.channel_id, c.channel_name, s.server_id, s.server_name
                FROM channels c
                JOIN servers s ON c.server_id = s.server_id
                WHERE c.channel_id = ?
                """,
                (channel_id,)
            ).fetchone()
        return dict(row) if row is not None else None

    def get_user_info(self, user_id: int, server_id: int = None) -> dict | None:
        """
        Return user info as a dict, or None if not found.

        If server_id is provided, the dict also includes 'nickname' (may be None
        if the user hasn't set one on that server) and 'server_name'.
        Without server_id, returns only 'user_id' and 'user_name'.

        Returned dict keys (with server_id): user_id, user_name, nickname, server_id, server_name
        Returned dict keys (without server_id): user_id, user_name
        """
        with self._get_conn() as conn:
            if server_id is not None:
                row = conn.execute(
                    """
                    SELECT u.user_id, u.user_name, un.nickname, s.server_id, s.server_name
                    FROM users u
                    LEFT JOIN user_nicknames un ON u.user_id = un.user_id AND un.server_id = ?
                    LEFT JOIN servers s ON un.server_id = s.server_id
                    WHERE u.user_id = ?
                    """,
                    (server_id, user_id)
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT user_id, user_name FROM users WHERE user_id = ?",
                    (user_id,)
                ).fetchone()
        return dict(row) if row is not None else None