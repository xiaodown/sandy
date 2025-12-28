"""
Database operations for the chat history API
"""

import sqlite3
import json
from datetime import datetime
from typing import List, Optional, Dict, Any
from contextlib import contextmanager

from models import ChatMessageCreate, ChatMessageResponse
from settings import db_path


class ChatDatabase:
    """SQLite database handler for chat messages."""
    
    def __init__(self):
        """Initialize database connection."""
        self.db_path = db_path
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        try:
            yield conn
        finally:
            conn.close()
    
    def init_db(self):
        """Initialize the database schema."""
        with self.get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    author TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    tags TEXT,  -- JSON array as string
                    summary TEXT
                )
            """)
            
            # Create index for common queries
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_author ON chat_messages(author)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp ON chat_messages(timestamp)
            """)
            
            conn.commit()
    
    def create_message(self, message: ChatMessageCreate) -> int:
        """Create a new chat message and return its ID."""
        with self.get_connection() as conn:
            # Convert tags list to JSON string if present
            tags_json = json.dumps(message.tags) if message.tags else None
            
            cursor = conn.execute("""
                INSERT INTO chat_messages (author, content, timestamp, tags, summary)
                VALUES (?, ?, ?, ?, ?)
            """, (message.author, message.content, message.timestamp.isoformat(), tags_json, message.summary))
            
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
            tags=tags,
            summary=row["summary"]
        )