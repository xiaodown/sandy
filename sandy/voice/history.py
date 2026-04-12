from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone


def _format_age(dt: datetime) -> str:
    now = datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    total_seconds = max(0, int((now - dt).total_seconds()))
    minutes, seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes}m ago" if minutes else f"{hours}h ago"
    if minutes:
        return f"{minutes}m ago"
    if seconds:
        return f"{seconds}s ago"
    return "just now"


@dataclass(slots=True)
class VoiceHistoryEntry:
    speaker_id: int
    speaker_name: str
    text: str
    created_at: datetime
    is_bot: bool


class VoiceHistory:
    def __init__(self, *, maxlen: int = 12) -> None:
        self._entries: deque[VoiceHistoryEntry] = deque(maxlen=maxlen)

    def add(self, entry: VoiceHistoryEntry) -> None:
        self._entries.append(entry)

    def clear(self) -> None:
        self._entries.clear()

    def entries(self) -> list[VoiceHistoryEntry]:
        return list(self._entries)

    def to_ollama_messages(self, bot_id: int) -> list[dict]:
        turns: list[dict] = []
        for entry in self._entries:
            role = "assistant" if entry.speaker_id == bot_id else "user"
            content = entry.text.strip() or "(no text content)"
            age = _format_age(entry.created_at)
            line = content if role == "assistant" else f"[{age}] [{entry.speaker_name}] {content}"
            if turns and turns[-1]["role"] == role:
                turns[-1]["content"] += f"\n{line}"
            else:
                turns.append({"role": role, "content": line})
        return turns
