"""Tests for sandy.voice.models — pure utility functions and sanitization."""

import pytest

from sandy.voice.models import (
    _normalize_name,
    _sanitize_voice_reply,
    _truncate_sentences,
    _truncate_words,
)


# ── _normalize_name ───────────────────────────────────────────────────────────

class TestNormalizeName:
    def test_lowercases(self):
        assert _normalize_name("Xiao Down") == "xiao down"

    def test_collapses_whitespace(self):
        assert _normalize_name("  ops   war   room  ") == "ops war room"

    def test_empty_string(self):
        assert _normalize_name("") == ""


# ── _truncate_words ───────────────────────────────────────────────────────────

class TestTruncateWords:
    def test_under_limit_unchanged(self):
        assert _truncate_words("one two three", 5) == "one two three"

    def test_at_limit_unchanged(self):
        assert _truncate_words("one two three", 3) == "one two three"

    def test_over_limit_truncated(self):
        assert _truncate_words("one two three four five", 3) == "one two three"

    def test_strips_trailing_punctuation(self):
        assert _truncate_words("hello, world, foo, bar", 2) == "hello, world"

    def test_strips_trailing_comma(self):
        assert _truncate_words("alpha, beta, gamma", 2) == "alpha, beta"

    def test_single_word(self):
        assert _truncate_words("hello", 1) == "hello"


# ── _truncate_sentences ──────────────────────────────────────────────────────

class TestTruncateSentences:
    def test_under_limit_unchanged(self):
        assert _truncate_sentences("One sentence.", 3) == "One sentence."

    def test_at_limit_unchanged(self):
        text = "First. Second. Third."
        assert _truncate_sentences(text, 3) == text

    def test_over_limit_truncated(self):
        assert _truncate_sentences("First. Second. Third. Fourth.", 2) == "First. Second."

    def test_handles_exclamation_and_question(self):
        assert _truncate_sentences("What! Really? Yes. No.", 2) == "What! Really?"

    def test_single_sentence_no_trailing_period(self):
        assert _truncate_sentences("no punctuation here", 1) == "no punctuation here"


# ── _sanitize_voice_reply ────────────────────────────────────────────────────

class TestSanitizeVoiceReply:
    def test_empty_input(self):
        assert _sanitize_voice_reply("") == ""

    def test_whitespace_only(self):
        assert _sanitize_voice_reply("   \n\t  ") == ""

    def test_collapses_whitespace(self):
        assert _sanitize_voice_reply("hello   world") == "hello world."

    def test_adds_trailing_period_when_missing(self):
        assert _sanitize_voice_reply("hey there") == "hey there."

    def test_preserves_existing_period(self):
        assert _sanitize_voice_reply("hey there.") == "hey there."

    def test_preserves_existing_exclamation(self):
        assert _sanitize_voice_reply("hey there!") == "hey there!"

    def test_preserves_existing_question_mark(self):
        assert _sanitize_voice_reply("hey there?") == "hey there?"

    def test_truncates_to_max_sentences(self):
        result = _sanitize_voice_reply("First. Second. Third. Fourth.")
        # Default max is 2 sentences
        assert result == "First. Second."

    def test_truncates_to_max_words(self):
        # Default max is 32 words
        words = " ".join(f"word{i}" for i in range(40))
        result = _sanitize_voice_reply(words)
        assert len(result.split()) <= 33  # 32 + possible trailing "."

    def test_truncates_to_max_chars(self):
        # Default max is 220 chars
        long_text = "a " * 200  # 400 chars
        result = _sanitize_voice_reply(long_text)
        assert len(result) <= 225  # some margin for trailing period

    def test_combined_limits(self):
        # Long text with many sentences and words
        text = ". ".join(["word " * 20] * 5)
        result = _sanitize_voice_reply(text)
        assert result  # not empty
        assert result[-1] in ".!?"
        assert len(result) <= 225
