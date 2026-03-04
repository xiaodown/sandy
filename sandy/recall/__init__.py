"""
Recall — Sandy's long-term message storage.

Provides direct access to the Recall SQLite database for storing and
retrieving Discord messages.  Previously a standalone FastAPI microservice;
now a plain Python package imported directly by memory.py and tools.py.

Public API:
    from sandy.recall import ChatDatabase, ChatMessageCreate, ChatMessageResponse

    db = ChatDatabase("data/recall.db")
    db.init_db()
    msg_id = db.create_message(ChatMessageCreate(...))
    messages = db.get_messages(server_id=12345, limit=50)
"""

from .database import ChatDatabase
from .models import ChatMessageCreate, ChatMessageResponse

__all__ = ["ChatDatabase", "ChatMessageCreate", "ChatMessageResponse"]
