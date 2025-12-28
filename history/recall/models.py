"""
Pydantic models for the chat history API.
"""

from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field


class ChatMessageCreate(BaseModel):
    """Model for creating a new chat message."""
    author: str = Field(..., min_length=1, max_length=255, description="Message author")
    content: str = Field(..., min_length=1, description="Message content")
    timestamp: datetime = Field(..., description="When the message was created (required)")
    server: str = Field(..., min_length=1, max_length=255, description="Server name (required)")
    channel: str = Field(..., min_length=1, max_length=255, description="Channel name (required)")
    tags: Optional[List[str]] = Field(default=None, description="Optional tags for the message")
    summary: Optional[str] = Field(default=None, max_length=1000, description="Optional message summary")

    class Config:
        json_schema_extra = {
            "example": {
                "author": "alice",
                "content": "Hello, this is a test message!",
                "timestamp": "2023-12-28T10:30:00",
                "server": "discord-main",
                "channel": "general",
                "tags": ["greeting", "test"],
                "summary": "A simple greeting message"
            }
        }


class ChatMessageResponse(BaseModel):
    """Model for chat message responses."""
    id: int
    author: str
    content: str
    timestamp: datetime
    server: str
    channel: str
    tags: Optional[List[str]] = None
    summary: Optional[str] = None

    class Config:
        from_attributes = True  # For SQLAlchemy compatibility
        json_schema_extra = {
            "example": {
                "id": 1,
                "author": "alice",
                "content": "Hello, this is a test message!",
                "timestamp": "2023-12-28T10:30:00",
                "server": "discord-main",
                "channel": "general",
                "tags": ["greeting", "test"],
                "summary": "A simple greeting message"
            }
        }