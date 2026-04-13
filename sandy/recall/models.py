"""
Pydantic models for the chat history API.
"""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class ChatMessageCreate(BaseModel):
    """Model for creating a new chat message."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "discord_message_id": 1482282320600891422,
                "author_id": 215896334130905090,
                "channel_id": 1359032552332621878,
                "server_id": 1359032272382621875,
                "author_name": "HappyUser",
                "channel_name": "general",
                "server_name": "Happy Friends Hangout",
                "content": "Hello, this is a test message!",
                "timestamp": "2026-02-21T10:30:00",
                "tags": ["greeting", "test"],
                "summary": "A simple greeting message",
            }
        }
    )

    discord_message_id: int | None = Field(
        default=None,
        description="Original Discord message snowflake, when known",
    )
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
    tags: list[str] | None = Field(default=None, description="Optional tags for the message (added by LLM)")
    summary: str | None = Field(default=None, max_length=1000, description="Optional message summary (added by LLM)")

class ChatMessageResponse(BaseModel):
    """Model for chat message responses."""
    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "id": 1,
                "discord_message_id": 1482282320600891422,
                "author_id": 215896334130905090,
                "channel_id": 1359032552332621878,
                "server_id": 1359032272382621875,
                "author_name": "HappyUser",
                "channel_name": "general",
                "server_name": "Happy Friends Hangout",
                "content": "Hello, this is a test message!",
                "timestamp": "2026-02-21T10:30:00",
                "tags": ["greeting", "test"],
                "summary": "A simple greeting message",
            }
        },
    )

    id: int
    discord_message_id: int | None = None
    author_id: int
    channel_id: int
    server_id: int
    author_name: str
    channel_name: str
    server_name: str
    content: str
    timestamp: datetime
    tags: list[str] | None = None
    summary: str | None = None
