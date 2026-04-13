"""Tests for sandy.voice.history — VoiceHistory and age formatting."""

from datetime import datetime, timedelta, timezone

import pytest

from sandy.voice.history import VoiceHistory, VoiceHistoryEntry, _format_age


# ── _format_age ──────────────────────────────────────────────────────────────

class TestFormatAge:
    def test_just_now(self):
        now = datetime.now(timezone.utc)
        assert _format_age(now) == "just now"

    def test_seconds_ago(self):
        dt = datetime.now(timezone.utc) - timedelta(seconds=30)
        assert _format_age(dt) == "30s ago"

    def test_minutes_only(self):
        dt = datetime.now(timezone.utc) - timedelta(minutes=5)
        result = _format_age(dt)
        assert result == "5m ago"

    def test_hours_and_minutes(self):
        dt = datetime.now(timezone.utc) - timedelta(hours=2, minutes=15)
        result = _format_age(dt)
        assert result == "2h15m ago"

    def test_hours_only(self):
        dt = datetime.now(timezone.utc) - timedelta(hours=3)
        result = _format_age(dt)
        assert result == "3h ago"

    def test_naive_datetime_treated_as_utc(self):
        # Naive datetime should not raise
        dt = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(minutes=1)
        result = _format_age(dt)
        assert "m ago" in result

    def test_future_timestamp_returns_just_now(self):
        # max(0, ...) clamp means future timestamps show "just now"
        future = datetime.now(timezone.utc) + timedelta(hours=1)
        assert _format_age(future) == "just now"


# ── VoiceHistory basic operations ────────────────────────────────────────────

class TestVoiceHistory:
    def _entry(self, text: str, *, speaker_id: int = 1, is_bot: bool = False) -> VoiceHistoryEntry:
        return VoiceHistoryEntry(
            speaker_id=speaker_id,
            speaker_name="alice" if not is_bot else "Sandy",
            text=text,
            created_at=datetime.now(timezone.utc),
            is_bot=is_bot,
        )

    def test_empty_history(self):
        h = VoiceHistory(maxlen=5)
        assert h.entries() == []

    def test_add_and_entries(self):
        h = VoiceHistory(maxlen=5)
        e = self._entry("hello")
        h.add(e)
        assert h.entries() == [e]

    def test_clear(self):
        h = VoiceHistory(maxlen=5)
        h.add(self._entry("hello"))
        h.clear()
        assert h.entries() == []

    def test_maxlen_eviction(self):
        h = VoiceHistory(maxlen=3)
        for i in range(5):
            h.add(self._entry(f"msg{i}"))
        texts = [e.text for e in h.entries()]
        assert texts == ["msg2", "msg3", "msg4"]

    def test_entries_returns_copy(self):
        h = VoiceHistory(maxlen=5)
        h.add(self._entry("hello"))
        entries = h.entries()
        entries.clear()
        assert len(h.entries()) == 1


# ── VoiceHistory.to_ollama_messages ──────────────────────────────────────────

class TestToOllamaMessages:
    BOT_ID = 999

    def _entry(
        self,
        text: str,
        *,
        speaker_id: int = 1,
        speaker_name: str = "alice",
        is_bot: bool = False,
    ) -> VoiceHistoryEntry:
        return VoiceHistoryEntry(
            speaker_id=speaker_id,
            speaker_name=speaker_name,
            text=text,
            created_at=datetime.now(timezone.utc),
            is_bot=is_bot,
        )

    def test_empty_history(self):
        h = VoiceHistory(maxlen=5)
        assert h.to_ollama_messages(self.BOT_ID) == []

    def test_single_user_message(self):
        h = VoiceHistory(maxlen=5)
        h.add(self._entry("hello"))
        msgs = h.to_ollama_messages(self.BOT_ID)
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"
        assert "[alice]" in msgs[0]["content"]
        assert "hello" in msgs[0]["content"]

    def test_bot_message_is_assistant_role(self):
        h = VoiceHistory(maxlen=5)
        h.add(self._entry("hey there", speaker_id=self.BOT_ID, speaker_name="Sandy", is_bot=True))
        msgs = h.to_ollama_messages(self.BOT_ID)
        assert len(msgs) == 1
        assert msgs[0]["role"] == "assistant"
        assert msgs[0]["content"] == "hey there"

    def test_consecutive_same_role_merged(self):
        h = VoiceHistory(maxlen=5)
        h.add(self._entry("first", speaker_id=1, speaker_name="alice"))
        h.add(self._entry("second", speaker_id=2, speaker_name="bob"))
        msgs = h.to_ollama_messages(self.BOT_ID)
        # Both are "user" role, so they merge
        assert len(msgs) == 1
        assert "alice" in msgs[0]["content"]
        assert "bob" in msgs[0]["content"]

    def test_alternating_roles_not_merged(self):
        h = VoiceHistory(maxlen=5)
        h.add(self._entry("hi", speaker_id=1, speaker_name="alice"))
        h.add(self._entry("hello", speaker_id=self.BOT_ID, speaker_name="Sandy", is_bot=True))
        h.add(self._entry("cool", speaker_id=1, speaker_name="alice"))
        msgs = h.to_ollama_messages(self.BOT_ID)
        assert len(msgs) == 3
        assert msgs[0]["role"] == "user"
        assert msgs[1]["role"] == "assistant"
        assert msgs[2]["role"] == "user"

    def test_empty_text_becomes_no_text_content(self):
        h = VoiceHistory(maxlen=5)
        h.add(self._entry("", speaker_id=1, speaker_name="alice"))
        msgs = h.to_ollama_messages(self.BOT_ID)
        assert "(no text content)" in msgs[0]["content"]

    def test_age_prefix_present_for_user(self):
        h = VoiceHistory(maxlen=5)
        h.add(self._entry("hello", speaker_id=1, speaker_name="alice"))
        msgs = h.to_ollama_messages(self.BOT_ID)
        # Should have age like "[just now]" and speaker name
        content = msgs[0]["content"]
        assert content.startswith("[")
        assert "[alice]" in content
