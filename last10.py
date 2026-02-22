"""
In-memory rolling cache of the last N messages per Discord channel.

Handles multiple servers and multiple channels per server.

Usage:
    cache = Last10()

    # In on_message:
    cache.add(message)

    # Get a channel's history:
    history = cache.get(server_id, channel_id)
    history[1].author.id        # most recent message's author ID  (1-based, 1 = newest)
    history[1].content          # most recent message's text
    history[2].author.display_name
    print(history.format())     # LLM-readable context block
"""

from collections import deque
from datetime import datetime, timezone
from typing import Optional

import discord


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_age(dt: datetime) -> str:
    """Convert a datetime to a compact human-readable age string.

    Examples: 'just now', '45s ago', '12m ago', '1h30m ago', '1d12h ago'
    """
    now = datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    total_seconds = max(0, int((now - dt).total_seconds()))

    days, remainder = divmod(total_seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)

    if days:
        return f"{days}d{hours}h ago" if hours else f"{days}d ago"
    if hours:
        return f"{hours}h{minutes}m ago" if minutes else f"{hours}h ago"
    if minutes:
        return f"{minutes}m ago"
    if seconds:
        return f"{seconds}s ago"
    return "just now"


# ---------------------------------------------------------------------------
# ChannelHistory — a snapshot view of one channel's recent messages
# ---------------------------------------------------------------------------

class ChannelHistory:
    """
    Ordered view of recent messages for a single channel.

    Indexing is 1-based with [1] being the *most recent* message:
        history[1]   → newest discord.Message
        history[2]   → second-newest
        ...

    Iterating goes oldest → newest (natural reading order).
    """

    def __init__(
        self,
        messages: list[discord.Message],
        registry=None,          # reserved for future registry integration
    ):
        # stored oldest-first
        self._messages = messages
        self.registry = registry

    # ------------------------------------------------------------------
    # Sequence interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._messages)

    def __bool__(self) -> bool:
        return bool(self._messages)

    def __iter__(self):
        """Iterate oldest → newest."""
        return iter(self._messages)

    def __getitem__(self, index: int) -> discord.Message:
        """1-based access: [1] is the most recent message."""
        if not isinstance(index, int):
            raise TypeError(f"Index must be int, not {type(index).__name__}")
        n = len(self._messages)
        if n == 0:
            raise IndexError("ChannelHistory is empty")
        if index < 1 or index > n:
            raise IndexError(f"Index {index} out of range (1..{n})")
        return self._messages[-index]

    # ------------------------------------------------------------------
    # LLM-readable formatting
    # ------------------------------------------------------------------

    def format(self, max_messages: Optional[int] = None) -> str:
        """Return a natural-language block of text suitable for LLM context.

        Messages are shown oldest → newest (natural reading order).

        Format per line:
            [<age>] [<display name>] <content>

        Args:
            max_messages: if set, only the N most recent messages are included.
        """
        if not self._messages:
            return "(no recent messages)"

        msgs = self._messages
        if max_messages is not None:
            msgs = msgs[-max_messages:]

        lines = []
        for msg in msgs:
            age = _format_age(msg.created_at)
            name = msg.author.display_name      # respects server nickname automatically
            content = msg.content.replace("\n", " ").strip()
            if not content:
                # Attachment-only or embed-only messages
                content = "(no text content)"
            lines.append(f"[{age}] [{name}] {content}")

        return "\n".join(lines)

    def to_ollama_messages(self, bot_id: int) -> list[dict]:
        """Convert history to ollama multi-turn message format for the Brain model.

        Sandy's own messages become role "assistant"; all others become role "user".
        Consecutive messages with the same role are merged into a single entry so
        the strict user/assistant alternation that ollama requires is maintained.
        User turns are prefixed with [DisplayName] so Sandy can distinguish between
        speakers within a merged block.

        bot_id — the Discord user ID of the bot (bot.user.id).
        """
        if not self._messages:
            return []

        turns: list[dict] = []

        for msg in self._messages:  # oldest → newest
            role = "assistant" if msg.author.id == bot_id else "user"
            content = msg.content.replace("\n", " ").strip() or "(no text content)"
            age = _format_age(msg.created_at)
            # Age prefix on every line so gaps are visible even inside merged turns.
            line = f"[{age}] {content}" if role == "assistant" else f"[{age}] [{msg.author.display_name}] {content}"

            # Note: age prefix is included on user turns so Sandy can perceive
            # temporal gaps, but omitted from assistant turns — she has no reason
            # to see timestamps on her own prior messages, and including them
            # causes the model to occasionally echo the format in new replies.
            if role == "assistant":
                line = content

            if turns and turns[-1]["role"] == role:
                turns[-1]["content"] += f"\n{line}"
            else:
                turns.append({"role": role, "content": line})

        return turns

    def __repr__(self) -> str:
        n = len(self._messages)
        if n:
            newest = _format_age(self._messages[-1].created_at)
            return f"<ChannelHistory messages={n} newest={newest}>"
        return "<ChannelHistory messages=0>"


# ---------------------------------------------------------------------------
# Last10 — the per-channel cache manager
# ---------------------------------------------------------------------------

class Last10:
    """
    Manages a rolling in-memory cache of the last `maxlen` messages
    for every (server_id, channel_id) pair the bot has seen.

    The name is conceptual; pass maxlen to change the window size.

    Example:
        cache = Last10()                              # or Last10(maxlen=20)
        cache.add(message)                            # call in on_message

        history = cache.get(server_id, channel_id)
        history[1].author.id                          # newest author's Discord ID
        history[1].content                            # newest message text
        print(history.format())                       # LLM context block
    """

    def __init__(self, maxlen: int = 10, registry=None):
        self.maxlen = maxlen
        self.registry = registry
        self._cache: dict[tuple[int, int], deque[discord.Message]] = {}

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

    def add(self, message: discord.Message) -> None:
        """Append a message to its channel's rolling cache.

        The oldest message is automatically evicted once the deque is full.
        Safe to call for every message in on_message.
        """
        key = (message.guild.id, message.channel.id)
        if key not in self._cache:
            self._cache[key] = deque(maxlen=self.maxlen)
        self._cache[key].append(message)

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    def get(self, server_id: int, channel_id: int) -> ChannelHistory:
        """Return a ChannelHistory snapshot for the given server + channel.

        Returns an empty ChannelHistory if nothing has been cached there yet.
        """
        msgs = list(self._cache.get((server_id, channel_id), []))
        return ChannelHistory(msgs, self.registry)

    def get_by_channel(self, channel_id: int) -> ChannelHistory:
        """Return a ChannelHistory using only channel_id.

        Discord channel IDs are globally unique snowflakes, so server_id is
        not required. Returns an empty ChannelHistory if not found.
        """
        for (sid, cid), dq in self._cache.items():
            if cid == channel_id:
                return ChannelHistory(list(dq), self.registry)
        return ChannelHistory([], self.registry)

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def clear_channel(self, server_id: int, channel_id: int) -> None:
        """Drop the cache for a specific channel."""
        self._cache.pop((server_id, channel_id), None)

    def clear_server(self, server_id: int) -> None:
        """Drop the cache for every channel in a server."""
        for key in [k for k in self._cache if k[0] == server_id]:
            del self._cache[key]

    def clear_all(self) -> None:
        """Wipe the entire cache."""
        self._cache.clear()

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def tracked_channels(self) -> list[tuple[int, int]]:
        """Return a list of (server_id, channel_id) tuples currently tracked."""
        return list(self._cache.keys())

    def __repr__(self) -> str:
        return f"<Last10 maxlen={self.maxlen} channels={len(self._cache)}>"


