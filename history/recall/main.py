#!/usr/bin/env python3
"""
Recall API Server

A simple FastAPI server for storing and retrieving chat messages.
"""

from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import List, Optional
import os
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")
RECALL_HOST = os.getenv("RECALL_HOST", "127.0.0.1")
RECALL_PORT = int(os.getenv("RECALL_PORT", "8000"))

from database import ChatDatabase
from models import ChatMessageCreate, ChatMessageResponse

# Initialize database
db = ChatDatabase()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan events."""
    # Startup
    db.init_db()
    yield
    # Shutdown (if needed)


# Initialize FastAPI app
app = FastAPI(
    title="Recall API",
    description="API for storing and retrieving chat messages",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware for web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Recall API",
        "version": "1.0.0",
        "endpoints": [
            "/docs",  # Swagger UI
            "/redoc",  # ReDoc
            "/messages",  # GET all messages
            "/messages/{message_id}",  # GET specific message
            "/messages/",  # POST new message
        ],
        "filtering": {
            "author": "Filter by message author (?author=alice)",
            "server": "Filter by server name (?server=discord-main)",
            "channel": "Filter by channel name (?channel=general)",
            "tag": "Filter by tag (?tag=project)",
            "pagination": "Use ?limit=50&offset=100 for pagination"
        },
        "time_filtering": {
            "hours_ago": "Get messages from last N hours (?hours_ago=24)",
            "minutes_ago": "Get messages from last N minutes (?minutes_ago=30)",
            "since": "Get messages since specific time (?since=2023-12-28T10:00:00)",
            "until": "Get messages until specific time (?until=2023-12-28T18:00:00)"
        }
    }


@app.post("/messages/", response_model=ChatMessageResponse)
async def create_message(message: ChatMessageCreate):
    """Create a new chat message."""
    try:
        message_id = db.create_message(message)
        stored_message = db.get_message(message_id)
        if not stored_message:
            raise HTTPException(status_code=500, detail="Failed to retrieve created message")
        return stored_message
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating message: {str(e)}")


@app.get("/messages/", response_model=List[ChatMessageResponse])
async def get_messages(
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    # ID filters (exact, preferred)
    author_id: Optional[int] = Query(default=None, description="Filter by Discord user ID"),
    server_id: Optional[int] = Query(default=None, description="Filter by Discord server (guild) ID"),
    channel_id: Optional[int] = Query(default=None, description="Filter by Discord channel ID"),
    # Name filters (convenience; ID takes precedence if both provided)
    author: Optional[str] = Query(default=None, description="Filter by author display name"),
    server: Optional[str] = Query(default=None, description="Filter by server name"),
    channel: Optional[str] = Query(default=None, description="Filter by channel name"),
    tag: Optional[str] = Query(default=None, description="Filter by tag (substring match, e.g. 'game' matches 'gaming')"),
    q: Optional[str] = Query(default=None, description="Full-text search against message content and summary (porter-stemmed)"),
    since: Optional[str] = Query(default=None, description="ISO datetime string (e.g. '2026-02-21T10:00:00')"),
    until: Optional[str] = Query(default=None, description="ISO datetime string (e.g. '2026-02-21T18:00:00')"),
    hours_ago: Optional[int] = Query(default=None, ge=0, description="Get messages from the last N hours"),
    minutes_ago: Optional[int] = Query(default=None, ge=0, description="Get messages from the last N minutes")
):
    """Get chat messages with optional filtering and time ranges."""
    try:
        since_dt = None
        until_dt = None

        if hours_ago is not None:
            since_dt = datetime.now() - timedelta(hours=hours_ago)
        elif minutes_ago is not None:
            since_dt = datetime.now() - timedelta(minutes=minutes_ago)

        if since:
            try:
                since_dt = datetime.fromisoformat(since.replace('Z', '+00:00'))
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid 'since' datetime format. Use ISO format like '2026-02-21T10:00:00'")

        if until:
            try:
                until_dt = datetime.fromisoformat(until.replace('Z', '+00:00'))
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid 'until' datetime format. Use ISO format like '2026-02-21T18:00:00'")

        messages = db.get_messages(
            limit=limit,
            offset=offset,
            author_id=author_id,
            author_name=author,
            server_id=server_id,
            server_name=server,
            channel_id=channel_id,
            channel_name=channel,
            tag=tag,
            q=q,
            since=since_dt,
            until=until_dt,
        )
        return messages
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving messages: {str(e)}")


@app.get("/messages/{message_id}", response_model=ChatMessageResponse)
async def get_message(message_id: int):
    """Get a specific chat message by ID."""
    message = db.get_message(message_id)
    if not message:
        raise HTTPException(status_code=404, detail="Message not found")
    return message


@app.delete("/messages/{message_id}")
async def delete_message(message_id: int):
    """Delete a chat message by ID."""
    success = db.delete_message(message_id)
    if not success:
        raise HTTPException(status_code=404, detail="Message not found")
    return {"message": "Message deleted successfully"}


@app.get("/stats/")
async def get_stats():
    """Get basic statistics about stored messages."""
    stats = db.get_stats()
    return stats


if __name__ == "__main__":
    uvicorn.run(app, host=RECALL_HOST, port=RECALL_PORT)