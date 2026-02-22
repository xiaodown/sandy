"""
Pydantic models for the chat history API.
"""

from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field


class ChatMessageCreate(BaseModel):
    """Model for creating a new chat message."""
    # Snowflake IDs — stable, globally unique Discord identifiers
    author_id: int = Field(..., description="Discord user ID of the message author")
    channel_id: int = Field(..., description="Discord channel ID where the message was sent")
    server_id: int = Field(..., description="Discord server (guild) ID where the message was sent")
    # Human-readable names captured at message time — historically accurate even if they change later
    author_name: str = Field(..., min_length=1, max_length=255, description="Author's display name at time of message")
    channel_name: str = Field(..., min_length=1, max_length=255, description="Channel name at time of message")
    server_name: str = Field(..., min_length=1, max_length=255, description="Server name at time of message")
    content: str = Field(..., min_length=1, description="Message content (may include markdown)")
    timestamp: datetime = Field(..., description="When the message was sent (ISO 8601, required)")
    tags: Optional[List[str]] = Field(default=None, description="Optional tags for the message (added by LLM)")
    summary: Optional[str] = Field(default=None, max_length=1000, description="Optional message summary (added by LLM)")

    class Config:
        json_schema_extra = {
            "example": {
                "author_id": 218896334130905090,
                "channel_id": 1359032772332621878,
                "server_id": 1359032772332621875,
                "author_name": "Xiaodown",
                "channel_name": "general",
                "server_name": "Xiaodown Bot Testing",
                "content": "Hello, this is a test message!",
                "timestamp": "2026-02-21T10:30:00",
                "tags": ["greeting", "test"],
                "summary": "A simple greeting message"
            }
        }


class ChatMessageResponse(BaseModel):
    """Model for chat message responses."""
    id: int
    author_id: int
    channel_id: int
    server_id: int
    author_name: str
    channel_name: str
    server_name: str
    content: str
    timestamp: datetime
    tags: Optional[List[str]] = None
    summary: Optional[str] = None

    class Config:
        from_attributes = True
        json_schema_extra = {
            "example": {
                "id": 1,
                "author_id": 218896334130905090,
                "channel_id": 1359032772332621878,
                "server_id": 1359032772332621875,
                "author_name": "Xiaodown",
                "channel_name": "general",
                "server_name": "Xiaodown Bot Testing",
                "content": "Hello, this is a test message!",
                "timestamp": "2026-02-21T10:30:00",
                "tags": ["greeting", "test"],
                "summary": "A simple greeting message"
            }
        }