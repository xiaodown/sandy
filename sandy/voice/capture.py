from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
import time
import wave

import discord

from ..logconf import get_logger

try:
    from discord.ext import voice_recv
except Exception:  # pragma: no cover - depends on runtime
    voice_recv = None

logger = get_logger("sandy.voice.capture")

WAV_CHANNELS = 2
WAV_SAMPLE_WIDTH = 2
WAV_SAMPLE_RATE = 48_000


def _voice_recv_listener():
    if voice_recv is None:
        def decorator(func):
            return func
        return decorator
    return voice_recv.AudioSink.listener()


def _slugify_capture_label(value: str) -> str:
    collapsed = "-".join(value.lower().split())
    sanitized = "".join(ch for ch in collapsed if ch.isalnum() or ch == "-")
    return sanitized.strip("-") or "unknown"


def _pcm_bytes_for_milliseconds(milliseconds: int) -> int:
    bytes_per_second = WAV_SAMPLE_RATE * WAV_CHANNELS * WAV_SAMPLE_WIDTH
    return int(bytes_per_second * (milliseconds / 1000))


@dataclass(slots=True)
class ActiveUtterance:
    speaker_id: int | None
    speaker_label: str
    ssrc: int
    started_at: float
    pcm_chunks: list[bytes]
    packet_count: int = 0


@dataclass(frozen=True, slots=True)
class CaptureJob:
    guild_id: int
    channel_id: int
    path: Path
    speaker_id: int | None
    speaker_label: str
    ssrc: int
    started_at: float
    ended_at: float
    duration_seconds: float
    packet_count: int
    saved_at: float


class UtteranceCaptureSink(voice_recv.AudioSink if voice_recv is not None else object):  # type: ignore[misc]
    def __init__(
        self,
        capture_dir: Path,
        *,
        preroll_ms: int,
        on_capture_saved,
        on_speaker_started=None,
        on_speaker_stopped=None,
    ) -> None:
        super().__init__()
        self.capture_dir = capture_dir
        self.capture_dir.mkdir(parents=True, exist_ok=True)
        self.preroll_ms = max(preroll_ms, 0)
        self._preroll_limit_bytes = _pcm_bytes_for_milliseconds(self.preroll_ms)
        self._active: dict[int, ActiveUtterance] = {}
        self._preroll_chunks: dict[int, deque[bytes]] = {}
        self._preroll_sizes: dict[int, int] = {}
        self._capture_index = self._next_capture_index()
        self._on_capture_saved = on_capture_saved
        self._on_speaker_started = on_speaker_started
        self._on_speaker_stopped = on_speaker_stopped

    def wants_opus(self) -> bool:
        return False

    def _next_capture_index(self) -> int:
        indices = []
        for path in self.capture_dir.glob("*.wav"):
            prefix, _, _ = path.name.partition("-")
            if prefix.isdigit():
                indices.append(int(prefix))
        return (max(indices) + 1) if indices else 1

    def _speaker_label(
        self,
        user: discord.User | discord.Member | None,
        data,
    ) -> str:
        return (
            getattr(user, "display_name", None)
            or getattr(data.source, "display_name", None)
            or getattr(user, "name", None)
            or getattr(data.source, "name", None)
            or "unknown"
        )

    def _append_preroll(self, ssrc: int, pcm: bytes) -> None:
        if self._preroll_limit_bytes <= 0:
            return
        chunks = self._preroll_chunks.setdefault(ssrc, deque())
        total = self._preroll_sizes.get(ssrc, 0) + len(pcm)
        chunks.append(pcm)
        while chunks and total > self._preroll_limit_bytes:
            total -= len(chunks.popleft())
        self._preroll_sizes[ssrc] = total

    def write(self, user, data) -> None:
        pcm = data.pcm or b""
        if not pcm:
            return
        ssrc = getattr(data.packet, "ssrc", None)
        if ssrc is None:
            return
        active = self._active.get(ssrc)
        if active is None:
            self._append_preroll(ssrc, pcm)
            return
        active.pcm_chunks.append(pcm)
        active.packet_count += 1

    def _finalize_ssrc(self, ssrc: int, *, reason: str) -> None:
        active = self._active.pop(ssrc, None)
        if active is None:
            return

        pcm_bytes = b"".join(active.pcm_chunks)
        if not pcm_bytes:
            return

        duration_seconds = len(pcm_bytes) / (WAV_SAMPLE_RATE * WAV_CHANNELS * WAV_SAMPLE_WIDTH)
        capture_name = (
            f"{self._capture_index:04d}-{_slugify_capture_label(active.speaker_label)}-"
            f"ssrc{ssrc}-{int(time.time())}.wav"
        )
        self._capture_index += 1
        capture_path = self.capture_dir / capture_name
        with wave.open(str(capture_path), "wb") as wav_file:
            wav_file.setnchannels(WAV_CHANNELS)
            wav_file.setsampwidth(WAV_SAMPLE_WIDTH)
            wav_file.setframerate(WAV_SAMPLE_RATE)
            wav_file.writeframes(pcm_bytes)

        ended_at = time.perf_counter()
        if self._on_capture_saved is not None:
            self._on_capture_saved(
                CaptureJob(
                    guild_id=self.voice_client.guild.id,
                    channel_id=self.voice_client.channel.id if self.voice_client.channel else 0,
                    path=capture_path,
                    speaker_id=active.speaker_id,
                    speaker_label=active.speaker_label,
                    ssrc=ssrc,
                    started_at=active.started_at,
                    ended_at=ended_at,
                    duration_seconds=duration_seconds,
                    packet_count=active.packet_count,
                    saved_at=time.perf_counter(),
                )
            )
        if self._on_speaker_stopped is not None:
            self._on_speaker_stopped(active.speaker_id, active.speaker_label, reason)

    @_voice_recv_listener()
    def on_voice_member_speaking_start(self, member: discord.Member) -> None:
        ssrc = self.voice_client._get_ssrc_from_id(member.id)
        if ssrc is None:
            return
        self._active[ssrc] = ActiveUtterance(
            speaker_id=member.id,
            speaker_label=member.display_name,
            ssrc=ssrc,
            started_at=time.perf_counter(),
            pcm_chunks=list(self._preroll_chunks.get(ssrc, ())),
        )
        self._preroll_chunks.pop(ssrc, None)
        self._preroll_sizes.pop(ssrc, None)
        if self._on_speaker_started is not None:
            self._on_speaker_started(member.id, member.display_name)

    @_voice_recv_listener()
    def on_voice_member_speaking_stop(self, member: discord.Member) -> None:
        ssrc = self.voice_client._get_ssrc_from_id(member.id)
        if ssrc is None:
            return
        self._finalize_ssrc(ssrc, reason="speaking-stop")

    @_voice_recv_listener()
    def on_voice_member_disconnect(self, member: discord.Member, ssrc: int | None) -> None:
        if ssrc is not None:
            self._finalize_ssrc(ssrc, reason="member-disconnect")

    def cleanup(self) -> None:
        for ssrc in list(self._active):
            self._finalize_ssrc(ssrc, reason="cleanup")
