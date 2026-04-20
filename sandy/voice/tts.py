from __future__ import annotations

import asyncio
import io
import wave
from dataclasses import dataclass

import discord
import httpx

try:
    import audioop
except ImportError:  # pragma: no cover - only on Python 3.13+
    import audioop_lts as audioop

DISCORD_PCM_CHANNELS = 2
DISCORD_PCM_SAMPLE_WIDTH = 2
DISCORD_PCM_SAMPLE_RATE = 48_000


@dataclass(frozen=True, slots=True)
class TtsServiceConfig:
    base_url: str = "http://127.0.0.1:8777"
    timeout_seconds: float = 180.0
    default_instruct: str | None = None
    default_language: str | None = None


class TtsServiceClient:
    def __init__(self, config: TtsServiceConfig) -> None:
        self.config = config

    def warmup_sync(self) -> None:
        response = httpx.post(
            f"{self.config.base_url.rstrip('/')}/warmup",
            timeout=self.config.timeout_seconds,
        )
        response.raise_for_status()

    async def warmup(self) -> None:
        await asyncio.to_thread(self.warmup_sync)

    def unload_sync(self) -> None:
        response = httpx.post(
            f"{self.config.base_url.rstrip('/')}/unload",
            timeout=self.config.timeout_seconds,
        )
        response.raise_for_status()

    async def unload(self) -> None:
        await asyncio.to_thread(self.unload_sync)

    def synthesize_bytes_sync(
        self,
        text: str,
        *,
        instruct: str | None = None,
        language: str | None = None,
    ) -> bytes:
        response = httpx.post(
            f"{self.config.base_url.rstrip('/')}/synthesize",
            json={
                "text": text,
                "instruct": instruct if instruct is not None else self.config.default_instruct,
                "language": language if language is not None else self.config.default_language,
            },
            timeout=self.config.timeout_seconds,
        )
        response.raise_for_status()
        return response.content

    async def synthesize_bytes(
        self,
        text: str,
        *,
        instruct: str | None = None,
        language: str | None = None,
    ) -> bytes:
        return await asyncio.to_thread(
            self.synthesize_bytes_sync,
            text,
            instruct=instruct,
            language=language,
        )


def wav_bytes_to_audio_source(wav_bytes: bytes) -> discord.AudioSource:
    with wave.open(io.BytesIO(wav_bytes), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        pcm = wav_file.readframes(wav_file.getnframes())

    if sample_width != DISCORD_PCM_SAMPLE_WIDTH:
        raise ValueError(
            "Unsupported WAV sample width: "
            f"expected {DISCORD_PCM_SAMPLE_WIDTH * 8}-bit PCM, got {sample_width * 8}-bit",
        )

    if channels == 1:
        pcm = audioop.tostereo(pcm, sample_width, 1, 1)
        channels = 2
    elif channels != DISCORD_PCM_CHANNELS:
        raise ValueError(f"Unsupported WAV channel count: {channels}")

    if sample_rate != DISCORD_PCM_SAMPLE_RATE:
        pcm, _ = audioop.ratecv(
            pcm,
            sample_width,
            channels,
            sample_rate,
            DISCORD_PCM_SAMPLE_RATE,
            None,
        )

    return discord.PCMAudio(io.BytesIO(pcm))
