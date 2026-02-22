"""
Sandy's callable tools for the Brain model.

This module is the single place where tools are defined and dispatched.
Adding a new tool:
    1. Write an async handler function (_handle_<name>).
    2. Add its schema dict to TOOL_SCHEMAS.
    3. Register the name in _HANDLERS.
    4. If it queries per-server data, add the name to _SERVER_SCOPED_TOOLS.

Server isolation for Recall tools
----------------------------------
The model is never given server_id / server_name in its tool schemas — those
fields are intentionally absent so the model cannot request another server's
data. dispatch() forcibly injects the current server's IDs before calling the
handler, regardless of what the model provided.
"""

import logging
import os
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlencode
from zoneinfo import ZoneInfo

import httpx
from dotenv import load_dotenv

from registry import Registry

load_dotenv()

logger = logging.getLogger(__name__)

_RECALL_BASE = (
    f"http://{os.getenv('RECALL_HOST', '127.0.0.1')}"
    f":{os.getenv('RECALL_PORT', '8000')}"
)

_PACIFIC = ZoneInfo("America/Los_Angeles")

# Registry for resolving current nicknames from stored author_id + server_id.
# Uses the same SQLite db as the bot; reads are cheap and always up-to-date.
_registry = Registry()

# Maximum number of tool-call rounds per brain invocation.
# Prevents runaway loops if the model keeps calling tools.
MAX_TOOL_ROUNDS = 5


# ---------------------------------------------------------------------------
# Internal Recall HTTP helper
# ---------------------------------------------------------------------------

async def _recall_get(path: str, params: dict[str, Any]) -> list[dict] | None:
    """GET from the Recall API. Returns parsed JSON list or None on error.

    None values are dropped so the API treats missing params as "no filter".
    """
    clean = {k: v for k, v in params.items() if v is not None}
    url = f"{_RECALL_BASE}{path}"
    if clean:
        url += "?" + urlencode(clean)
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.get(url, headers={"Accept": "application/json"})
            r.raise_for_status()
            return r.json()
    except Exception as exc:
        logger.error("Recall API error (%s): %s", url, exc)
        return None


def _format_messages(data: list[dict]) -> str:
    """Format a Recall message list into a readable block for the model.

    Author display names are resolved via the registry so Sandy sees current
    nicknames rather than whatever name was stored at the time the message was
    archived. Falls back to the stored author_name if the user isn't in the
    registry (e.g. a message from before the bot joined).
    """
    lines = []
    for msg in data:
        try:
            dt = datetime.fromisoformat(msg["timestamp"].replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            ts = dt.astimezone(_PACIFIC).strftime("%Y-%m-%d %H:%M %Z")
        except Exception:
            ts = msg.get("timestamp", "?")

        # Prefer current nickname from registry; fall back to archived name.
        author_id  = msg.get("author_id")
        server_id  = msg.get("server_id")
        stored_name = msg.get("author_name", "?")
        if author_id and server_id:
            info = _registry.get_user_info(author_id, server_id)
            if info:
                author = info.get("nickname") or info.get("user_name") or stored_name
            else:
                author = stored_name
        else:
            author = stored_name

        channel = msg.get("channel_name", "?")
        content = msg.get("content",      "")
        tags    = msg.get("tags") or []
        summary = msg.get("summary") or ""
        line    = f"[{ts}] #{channel} <{author}>: {content}"
        if tags:
            line += f"  [tags: {', '.join(tags)}]"
        if summary:
            line += f"  (summary: {summary})"
        lines.append(line)
    return "\n".join(lines) if lines else "(no messages found)"


# ---------------------------------------------------------------------------
# Tool handlers — one async function per tool
# ---------------------------------------------------------------------------

async def _handle_get_chat_history(args: dict[str, Any]) -> str:
    """Retrieve messages from Recall with optional filtering."""
    data = await _recall_get("/messages/", args)
    if data is None:
        return "Error: could not reach the memory store."
    if not data:
        return "No messages found matching those filters."
    return f"{len(data)} message(s) retrieved:\n\n{_format_messages(data)}"


async def _handle_search_messages(args: dict[str, Any]) -> str:
    """Full-text search through Recall messages."""
    # The model uses the parameter name 'query'; the Recall API uses 'q'.
    api_args = {**args}
    if "query" in api_args:
        api_args["q"] = api_args.pop("query")
    data = await _recall_get("/messages/", api_args)
    if data is None:
        return "Error: could not reach the memory store."
    if not data:
        return f"No messages found for query: {args.get('query', '?')}"
    return f"{len(data)} message(s) found:\n\n{_format_messages(data)}"


# ---------------------------------------------------------------------------
# Tool schemas — passed verbatim to ollama as tools=
#
# server_id / server_name are intentionally absent from every schema here.
# They are injected by dispatch() and the model cannot override them.
# ---------------------------------------------------------------------------

TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "get_chat_history",
            "description": (
                "Retrieve past messages from this server's chat history. "
                "Use this to recall what was discussed before your current "
                "context window, or to check what someone said recently. "
                "Filter by channel, author, tag, or time window."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "author":      {"type": "string",  "description": "Filter by author display name"},
                    "author_id":   {"type": "integer", "description": "Filter by Discord user ID (more reliable than name)"},
                    "channel":     {"type": "string",  "description": "Filter by channel name"},
                    "channel_id":  {"type": "integer", "description": "Filter by Discord channel ID"},
                    "tag":         {"type": "string",  "description": "Tag substring to filter by (e.g. 'game' matches 'gaming')"},
                    "hours_ago":   {"type": "integer", "description": "Limit to messages from the last N hours"},
                    "minutes_ago": {"type": "integer", "description": "Limit to messages from the last N minutes"},
                    "since":       {"type": "string",  "description": "ISO datetime lower bound (e.g. '2026-02-01T00:00:00')"},
                    "until":       {"type": "string",  "description": "ISO datetime upper bound"},
                    "limit":       {"type": "integer", "description": "Maximum messages to return (default 100, max 1000)"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_messages",
            "description": (
                "Full-text search through past messages on this server. "
                "Uses stemmed matching, so 'run' finds 'running', 'ran', etc. "
                "Use this to find messages about a specific topic or keyword."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query":      {"type": "string",  "description": "Text to search for (stemmed, case-insensitive)"},
                    "author":     {"type": "string",  "description": "Limit to a specific author display name"},
                    "author_id":  {"type": "integer", "description": "Limit to a specific Discord user ID"},
                    "channel":    {"type": "string",  "description": "Limit to a specific channel name"},
                    "channel_id": {"type": "integer", "description": "Limit to a specific channel ID"},
                    "hours_ago":  {"type": "integer", "description": "Limit to the last N hours"},
                    "limit":      {"type": "integer", "description": "Maximum messages to return (default 50)"},
                },
                "required": ["query"],
            },
        },
    },
]

# Map tool name → handler function.
_HANDLERS: dict[str, Any] = {
    "get_chat_history": _handle_get_chat_history,
    "search_messages":  _handle_search_messages,
}

# Tools that query per-server data and require server_id injection.
# A future tool like 'search_web' would NOT appear here.
_SERVER_SCOPED_TOOLS: frozenset[str] = frozenset({
    "get_chat_history",
    "search_messages",
})


# ---------------------------------------------------------------------------
# Dispatcher — called by ask_brain for each tool call the model requests
# ---------------------------------------------------------------------------

async def dispatch(
    tool_name: str,
    arguments: dict[str, Any],
    server_id: int,
    server_name: str,
) -> str:
    """Execute one tool call and return a string result for the model.

    For server-scoped tools, server_id and server_name are forcibly
    overwritten — the model cannot request data from a different server.
    """
    handler = _HANDLERS.get(tool_name)
    if handler is None:
        logger.warning("Brain requested unknown tool: %r", tool_name)
        return f"Error: unknown tool '{tool_name}'."

    # Enforce server isolation: stamp server context onto all memory tools.
    # Note: the Recall API uses "server" (not "server_name") as the query param
    # for the name filter; "server_id" is the primary isolation key.
    if tool_name in _SERVER_SCOPED_TOOLS:
        arguments = {**arguments, "server_id": server_id, "server": server_name}

    # Log without server context to keep logs tidy (it's always the same value).
    loggable = {k: v for k, v in arguments.items() if k not in ("server_id", "server", "server_name")}
    logger.info("Tool dispatch → %s(%s)", tool_name, loggable)

    try:
        return await handler(arguments)
    except Exception as exc:
        logger.error("Tool %r raised: %s", tool_name, exc)
        return f"Error executing {tool_name}: {exc}"
