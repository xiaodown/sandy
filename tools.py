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

async def _handle_recall_recent(args: dict[str, Any]) -> str:
    """Retrieve recent messages from Recall, optionally filtered by time window."""
    data = await _recall_get("/messages/", args)
    if data is None:
        return "Error: could not reach the memory store."
    if not data:
        return "No messages found matching those filters."
    return f"{len(data)} message(s) retrieved:\n\n{_format_messages(data)}"


async def _handle_recall_from_user(args: dict[str, Any]) -> str:
    """Retrieve messages from Recall filtered to a specific author."""
    data = await _recall_get("/messages/", args)
    if data is None:
        return "Error: could not reach the memory store."
    if not data:
        return f"No messages found from author: {args.get('author', '?')}"
    return f"{len(data)} message(s) retrieved:\n\n{_format_messages(data)}"


async def _handle_recall_by_topic(args: dict[str, Any]) -> str:
    """Retrieve messages from Recall filtered to a topic tag."""
    data = await _recall_get("/messages/", args)
    if data is None:
        return "Error: could not reach the memory store."
    if not data:
        return f"No messages found for topic: {args.get('tag', '?')}"
    return f"{len(data)} message(s) retrieved:\n\n{_format_messages(data)}"


async def _handle_search_memories(args: dict[str, Any]) -> str:
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


async def _handle_get_current_time(_args: dict[str, Any]) -> str:
    """Return the current date and time in Pacific time."""
    now = datetime.now(_PACIFIC)
    day = now.day
    suffix = (
        "th" if 11 <= day <= 13
        else {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
    )
    return (
        f"Today is {now.strftime('%A, %B')} {day}{suffix}, {now.year}. "
        f"The current time is {now.strftime('%I:%M %p')} {now.strftime('%Z')}."
    )


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
            "name": "recall_recent",
            "description": (
                "Use this to catch up on what's been happening — what people have "
                "been talking about recently in the server. Good for 'what did I miss', "
                "'what's been going on lately', or any time you want a feel for recent "
                "activity without a specific topic or person in mind."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "hours_ago":   {"type": "integer", "description": "Look back N hours (e.g. 24 for the last day, 168 for the last week)"},
                    "minutes_ago": {"type": "integer", "description": "Look back N minutes"},
                    "since":       {"type": "string",  "description": "ISO datetime lower bound (e.g. '2026-02-01T00:00:00')"},
                    "until":       {"type": "string",  "description": "ISO datetime upper bound"},
                    "channel":     {"type": "string",  "description": "Limit to a specific channel name (only use when explicitly asked)"},
                    "limit":       {"type": "integer", "description": "Maximum messages to return (default 100)"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recall_from_user",
            "description": (
                "Use this when you want to remember what a specific person said or did. "
                "Good for 'what has Dave been up to', 'what did Sarah say about X', "
                "or any time a particular person's words or actions are relevant."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "author":    {"type": "string",  "description": "The person's display name as it appears in chat"},
                    "hours_ago": {"type": "integer", "description": "Limit to the last N hours"},
                    "since":     {"type": "string",  "description": "ISO datetime lower bound (e.g. '2026-02-01T00:00:00')"},
                    "channel":   {"type": "string",  "description": "Limit to a specific channel name (only use when explicitly asked)"},
                    "limit":     {"type": "integer", "description": "Maximum messages to return (default 50)"},
                },
                "required": ["author"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recall_by_topic",
            "description": (
                "Use this to remember past conversations around a theme or topic — "
                "things like 'gaming', 'music', 'movies', 'work', or whatever tends "
                "to come up. Each past message is tagged with topics it's about, and "
                "this searches those tags. Works best when you have a clear theme in "
                "mind rather than specific words."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "tag":       {"type": "string",  "description": "The theme or topic to look up (e.g. 'gaming', 'music', 'food')"},
                    "author":    {"type": "string",  "description": "Limit to a specific person"},
                    "hours_ago": {"type": "integer", "description": "Limit to the last N hours"},
                    "limit":     {"type": "integer", "description": "Maximum messages to return (default 50)"},
                },
                "required": ["tag"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_memories",
            "description": (
                "Use this to search for specific words, phrases, or concepts across "
                "all past messages. Good for 'did anyone mention X', 'find the "
                "conversation about Y', or when you need to dig for something specific. "
                "Uses stemmed matching, so 'run' finds 'running', 'ran', etc."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query":     {"type": "string",  "description": "Words or phrase to search for"},
                    "author":    {"type": "string",  "description": "Limit to a specific person"},
                    "hours_ago": {"type": "integer", "description": "Limit to the last N hours"},
                    "channel":   {"type": "string",  "description": "Limit to a specific channel name (only use when explicitly asked)"},
                    "limit":     {"type": "integer", "description": "Maximum messages to return (default 50)"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": (
                "Returns the current date and time. Use this when you need to know "
                "exactly what time or date it is right now — for example, to answer "
                "'what time is it', to work out what day of the week it is, or to "
                "calculate how long ago something happened."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
]

# Map tool name → handler function.
_HANDLERS: dict[str, Any] = {
    "recall_recent":    _handle_recall_recent,
    "recall_from_user": _handle_recall_from_user,
    "recall_by_topic":  _handle_recall_by_topic,
    "search_memories":  _handle_search_memories,
    "get_current_time": _handle_get_current_time,
}

# Tools that query per-server data and require server_id injection.
# A future tool like 'search_web' would NOT appear here.
_SERVER_SCOPED_TOOLS: frozenset[str] = frozenset({
    "recall_recent",
    "recall_from_user",
    "recall_by_topic",
    "search_memories",
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

    # Enforce server isolation: stamp server_id onto all memory tools.
    # Also strip channel_id — Sandy has no reliable way to know channel snowflakes;
    # she should search server-wide and filter by channel *name* only if the user
    # explicitly named one. Leaving channel_id in causes silent empty results when
    # the model hallucinates or copies a stale ID from prior context.
    if tool_name in _SERVER_SCOPED_TOOLS:
        # Strip any integer snowflake IDs the model may have hallucinated —
        # channel_id and author_id are never shown to the model so it can only
        # guess them, and wrong IDs silently return zero results.
        # Name-based filters (author, channel) are safe: names appear in context.
        arguments = {k: v for k, v in arguments.items() if k not in ("channel_id", "author_id")}
        arguments = {**arguments, "server_id": server_id}

    # Log without server context to keep logs tidy (it's always the same value).
    loggable = {k: v for k, v in arguments.items() if k not in ("server_id",)}
    logger.info("Sandy is using tool: %s  args=%s", tool_name, loggable)

    try:
        result = await handler(arguments)
        # Log first line of result so it's easy to see what came back without
        # flooding the log with full message dumps.
        first_line = result.splitlines()[0] if result else "(empty)"
        logger.info("Tool %s returned: %s", tool_name, first_line)
        return result
    except Exception as exc:
        logger.error("Tool %r raised: %s", tool_name, exc)
        return f"Error executing {tool_name}: {exc}"
