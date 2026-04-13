from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import os
import re
from time import time

import discord

from ..paths import resolve_runtime_path
from .history import VoiceHistory

# ── Constants ──────────────────────────────────────────────────────────────────

_VOICE_CAPTURE_DIR = resolve_runtime_path(os.getenv("VOICE_CAPTURE_DIR", "data/prod/voice_captures"))
_VOICE_PREROLL_MS = int(os.getenv("VOICE_PREROLL_MS", "250"))
_VOICE_STITCH_GAP_SECONDS = float(os.getenv("VOICE_STITCH_GAP_SECONDS", "1.0"))
_VOICE_STITCH_RELEASE_SECONDS = float(os.getenv("VOICE_STITCH_RELEASE_SECONDS", "1.35"))
_VOICE_HISTORY_MAXLEN = int(os.getenv("VOICE_HISTORY_MAXLEN", "12"))
_VOICE_IDLE_AUTO_LEAVE_SECONDS = int(os.getenv("VOICE_IDLE_AUTO_LEAVE_SECONDS", "300"))
_VOICE_FORCE_RELEASE_SECONDS = float(os.getenv("VOICE_FORCE_RELEASE_SECONDS", "3.25"))
_VOICE_REPLY_MAX_WORDS = int(os.getenv("VOICE_REPLY_MAX_WORDS", "32"))
_VOICE_REPLY_MAX_CHARS = int(os.getenv("VOICE_REPLY_MAX_CHARS", "220"))
_VOICE_REPLY_MAX_SENTENCES = int(os.getenv("VOICE_REPLY_MAX_SENTENCES", "2"))


def configure_voice(
    *,
    capture_dir: str | None = None,
    preroll_ms: int | None = None,
    stitch_gap_seconds: float | None = None,
    stitch_release_seconds: float | None = None,
    history_maxlen: int | None = None,
    idle_auto_leave_seconds: int | None = None,
    force_release_seconds: float | None = None,
    reply_max_words: int | None = None,
    reply_max_chars: int | None = None,
    reply_max_sentences: int | None = None,
) -> None:
    """Override voice constants from a VoiceConfig at startup."""
    global _VOICE_CAPTURE_DIR, _VOICE_PREROLL_MS, _VOICE_STITCH_GAP_SECONDS
    global _VOICE_STITCH_RELEASE_SECONDS, _VOICE_HISTORY_MAXLEN
    global _VOICE_IDLE_AUTO_LEAVE_SECONDS, _VOICE_FORCE_RELEASE_SECONDS
    global _VOICE_REPLY_MAX_WORDS, _VOICE_REPLY_MAX_CHARS, _VOICE_REPLY_MAX_SENTENCES
    if capture_dir is not None:
        _VOICE_CAPTURE_DIR = resolve_runtime_path(capture_dir)
    if preroll_ms is not None:
        _VOICE_PREROLL_MS = preroll_ms
    if stitch_gap_seconds is not None:
        _VOICE_STITCH_GAP_SECONDS = stitch_gap_seconds
    if stitch_release_seconds is not None:
        _VOICE_STITCH_RELEASE_SECONDS = stitch_release_seconds
    if history_maxlen is not None:
        _VOICE_HISTORY_MAXLEN = history_maxlen
    if idle_auto_leave_seconds is not None:
        _VOICE_IDLE_AUTO_LEAVE_SECONDS = idle_auto_leave_seconds
    if force_release_seconds is not None:
        _VOICE_FORCE_RELEASE_SECONDS = force_release_seconds
    if reply_max_words is not None:
        _VOICE_REPLY_MAX_WORDS = reply_max_words
    if reply_max_chars is not None:
        _VOICE_REPLY_MAX_CHARS = reply_max_chars
    if reply_max_sentences is not None:
        _VOICE_REPLY_MAX_SENTENCES = reply_max_sentences


# ── Utility functions ──────────────────────────────────────────────────────────

def _normalize_name(value: str) -> str:
    return " ".join(value.lower().split())


def _truncate_words(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]).rstrip(" ,;:-")


def _truncate_sentences(text: str, max_sentences: int) -> str:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    if len(parts) <= max_sentences:
        return text
    return " ".join(parts[:max_sentences]).strip()


def _sanitize_voice_reply(text: str) -> str:
    cleaned = " ".join(text.strip().split())
    if not cleaned:
        return ""
    cleaned = _truncate_sentences(cleaned, _VOICE_REPLY_MAX_SENTENCES)
    cleaned = _truncate_words(cleaned, _VOICE_REPLY_MAX_WORDS)
    if len(cleaned) > _VOICE_REPLY_MAX_CHARS:
        cleaned = cleaned[:_VOICE_REPLY_MAX_CHARS].rsplit(" ", 1)[0].rstrip(" ,;:-")
    if cleaned and cleaned[-1] not in ".!?":
        cleaned = f"{cleaned}."
    return cleaned


def resolve_target_channel(
    guild: discord.Guild,
    *,
    query: str,
    author_voice_channel: discord.VoiceChannel | None,
) -> discord.VoiceChannel | None:
    cleaned_query = query.strip()
    if not cleaned_query:
        return author_voice_channel

    normalized_query = _normalize_name(cleaned_query)
    channels = list(guild.voice_channels)

    for channel in channels:
        if _normalize_name(channel.name) == normalized_query:
            return channel

    partial_matches = [
        channel for channel in channels
        if normalized_query in _normalize_name(channel.name)
    ]
    if len(partial_matches) == 1:
        return partial_matches[0]
    return None


# ── Dataclasses ────────────────────────────────────────────────────────────────

@dataclass(slots=True)
class PendingSpeakerTurn:
    speaker_id: int
    speaker_name: str
    text: str
    started_at: float
    ended_at: float
    fragment_count: int = 0
    total_audio_seconds: float = 0.0
    total_stt_elapsed_seconds: float = 0.0
    transcripts: list[str] = field(default_factory=list)
    release_task: asyncio.Task | None = None
    force_release_task: asyncio.Task | None = None


@dataclass(slots=True)
class CompletedVoiceTurn:
    speaker_id: int
    speaker_name: str
    text: str
    started_at: float
    ended_at: float
    fragment_count: int
    total_audio_seconds: float
    total_stt_elapsed_seconds: float
    transcripts: list[str]


@dataclass(slots=True)
class VoiceSession:
    session_id: str
    guild_id: int
    guild_name: str
    channel_id: int
    channel_name: str
    requested_by_user_id: int
    requested_by_name: str
    participant_names: list[str]
    started_at: float
    voice_client: discord.VoiceProtocol | None = None
    history: VoiceHistory = field(default_factory=lambda: VoiceHistory(maxlen=_VOICE_HISTORY_MAXLEN))
    pending_by_speaker: dict[int, PendingSpeakerTurn] = field(default_factory=dict)
    pending_stt_counts: dict[int, int] = field(default_factory=dict)
    active_speakers: set[int] = field(default_factory=set)
    response_task: asyncio.Task | None = None
    playback_active: bool = False
    pending_response_needed: bool = False
    pending_response_turns: list[CompletedVoiceTurn] = field(default_factory=list)
    response_counter: int = 0
    reply_counter: int = 0
    last_activity_at: float = field(default_factory=time)
    idle_task: asyncio.Task | None = None
    sink: object | None = None


@dataclass(frozen=True, slots=True)
class VoiceCommandResult:
    handled: bool
    reply: str | None = None
    ok: bool = True
