"""
Database operations for the recall API
"""

import sqlite3
import json
import os
from datetime import datetime
from typing import List, Optional, Dict, Any
from contextlib import contextmanager

from models import ChatMessageCreate, ChatMessageResponse
from settings import db_path


class ChatDatabase:
    """SQLite database handler for chat messages."""
    
    CURRENT_SCHEMA_VERSION = 1
    
    def __init__(self):
        """Initialize database connection."""
        self.db_path = db_path
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections."""
        # Ensure the database directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        try:
            yield conn
        finally:
            conn.close()
    
    def init_db(self):
        """Initialize the database schema and run any needed migrations."""
        # Run migrations to get to the current schema version
        self.migrate_to_latest()
    
    def get_schema_version(self) -> int:
        """Get the current schema version from the database."""
        with self.get_connection() as conn:
            try:
                cursor = conn.execute("SELECT version FROM schema_version LIMIT 1")
                row = cursor.fetchone()
                return row[0] if row else 0
            except sqlite3.OperationalError:
                # No schema_version table exists yet
                return 0
    
    def set_schema_version(self, version: int):
        """Set the schema version in the database."""
        with self.get_connection() as conn:
            # Create schema_version table if it doesn't exist
            conn.execute("""
                CREATE TABLE IF NOT EXISTS schema_version (
                    version INTEGER PRIMARY KEY
                )
            """)
            
            # Delete any existing version and insert the new one
            conn.execute("DELETE FROM schema_version")
            conn.execute("INSERT INTO schema_version (version) VALUES (?)", (version,))
            conn.commit()
    
    def migrate_to_latest(self):
        """Run all necessary migrations to get to the current schema version."""
        current_version = self.get_schema_version()
        
        if current_version < self.CURRENT_SCHEMA_VERSION:
            print(f"Migrating database from version {current_version} to {self.CURRENT_SCHEMA_VERSION}...")
            
            # Run migrations in order
            if current_version < 1:
                self._migrate_v1_create_initial_schema()
                self.set_schema_version(1)
                print("✓ Migrated to version 1: Created initial schema with server/channel fields")
    
    def _migrate_v1_create_initial_schema(self):
        """Migration v1: Create the initial schema with all current fields."""
        with self.get_connection() as conn:
            # Create the main table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    author TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    server TEXT NOT NULL,
                    channel TEXT NOT NULL,
                    tags TEXT,  -- JSON array as string
                    summary TEXT
                )
            """)
            
            # Create indexes for common query patterns
            # Single column indexes for filtering
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_author ON chat_messages(author)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_server ON chat_messages(server)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_channel ON chat_messages(channel)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp ON chat_messages(timestamp)
            """)
            
            # Composite indexes for common filter combinations
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_server_channel ON chat_messages(server, channel)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_server_channel_timestamp ON chat_messages(server, channel, timestamp DESC)
            """)
            
            conn.commit()
    
    def create_message(self, message: ChatMessageCreate) -> int:
        """Create a new chat message and return its ID."""
        with self.get_connection() as conn:
            # Convert tags list to JSON string if present
            tags_json = json.dumps(message.tags) if message.tags else None
            
            cursor = conn.execute("""
                INSERT INTO chat_messages (author, content, timestamp, server, channel, tags, summary)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (message.author, message.content, message.timestamp.isoformat(), 
                  message.server, message.channel, tags_json, message.summary))
            
            conn.commit()
            return cursor.lastrowid
    
    def get_message(self, message_id: int) -> Optional[ChatMessageResponse]:
        """Get a specific message by ID."""
        with self.get_connection() as conn:
            cursor = conn.execute("""
                SELECT * FROM chat_messages WHERE id = ?
            """, (message_id,))
            
            row = cursor.fetchone()
            if row:
                return self._row_to_response(row)
            return None
    
    def get_messages(
        self, 
        limit: int = 100, 
        offset: int = 0,
        author: Optional[str] = None,
        server: Optional[str] = None,
        channel: Optional[str] = None,
        tag: Optional[str] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None
    ) -> List[ChatMessageResponse]:
        """Get messages with optional filtering."""
        with self.get_connection() as conn:
            # Build query with filters
            query = "SELECT * FROM chat_messages WHERE 1=1"
            params = []
            
            if author:
                query += " AND author = ?"
                params.append(author)
            
            if server:
                query += " AND server = ?"
                params.append(server)
            
            if channel:
                query += " AND channel = ?"
                params.append(channel)
            
            if tag:
                query += " AND tags LIKE ?"
                params.append(f'%"{tag}"%')  # Search for tag within JSON array
            
            if since:
                query += " AND timestamp >= ?"
                params.append(since.isoformat())
            
            if until:
                query += " AND timestamp <= ?"
                params.append(until.isoformat())
            
            query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])
            
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()
            
            return [self._row_to_response(row) for row in rows]
    
    def delete_message(self, message_id: int) -> bool:
        """Delete a message by ID. Returns True if deleted, False if not found."""
        with self.get_connection() as conn:
            cursor = conn.execute("""
                DELETE FROM chat_messages WHERE id = ?
            """, (message_id,))
            
            conn.commit()
            return cursor.rowcount > 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get basic statistics about stored messages."""
        with self.get_connection() as conn:
            # Total message count
            cursor = conn.execute("SELECT COUNT(*) as total FROM chat_messages")
            total = cursor.fetchone()["total"]
            
            # Unique authors
            cursor = conn.execute("SELECT COUNT(DISTINCT author) as unique_authors FROM chat_messages")
            unique_authors = cursor.fetchone()["unique_authors"]
            
            # Recent message timestamp
            cursor = conn.execute("SELECT MAX(timestamp) as latest FROM chat_messages")
            latest_row = cursor.fetchone()
            latest = latest_row["latest"] if latest_row["latest"] else None
            
            return {
                "total_messages": total,
                "unique_authors": unique_authors,
                "latest_message_time": latest
            }
    
    def _row_to_response(self, row: sqlite3.Row) -> ChatMessageResponse:
        """Convert a database row to a ChatMessageResponse."""
        # Parse tags from JSON if present
        tags = None
        if row["tags"]:
            try:
                tags = json.loads(row["tags"])
            except json.JSONDecodeError:
                tags = None
        
        # Parse timestamp
        timestamp = datetime.fromisoformat(row["timestamp"].replace("Z", "+00:00"))
        
        return ChatMessageResponse(
            id=row["id"],
            author=row["author"],
            content=row["content"],
            timestamp=timestamp,
            server=row["server"],
            channel=row["channel"],
            tags=tags,
            summary=row["summary"]
        )