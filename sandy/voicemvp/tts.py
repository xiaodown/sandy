from __future__ import annotations

import audioop
import io
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import discord

logger = logging.getLogger("sandy.voicemvp.tts")

DISCORD_PCM_CHANNELS = 2
DISCORD_PCM_SAMPLE_WIDTH = 2
DISCORD_PCM_SAMPLE_RATE = 48_000


@dataclass(frozen=True, slots=True)
class QwenTtsConfig:
    model_name: str
    language: str
    instruct: str
    device_map: str = "cuda:0"
    dtype: str = "bfloat16"
    attn_implementation: str = "sdpa"
    max_new_tokens: int = 2048


class QwenVoiceDesignTts:
    def __init__(self, config: QwenTtsConfig) -> None:
        self.config = config
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return self._model

        import torch
        from qwen_tts import Qwen3TTSModel

        dtype_name = self.config.dtype.lower()
        if dtype_name == "bfloat16":
            dtype = torch.bfloat16
        elif dtype_name == "float16":
            dtype = torch.float16
        else:
            raise ValueError(f"Unsupported Qwen TTS dtype: {self.config.dtype}")

        logger.info(
            "Loading Qwen TTS model=%s device_map=%s dtype=%s attn=%s",
            self.config.model_name,
            self.config.device_map,
            self.config.dtype,
            self.config.attn_implementation,
        )
        self._model = Qwen3TTSModel.from_pretrained(
            self.config.model_name,
            device_map=self.config.device_map,
            dtype=dtype,
            attn_implementation=self.config.attn_implementation,
        )
        logger.info("Loaded Qwen TTS model=%s", self.config.model_name)
        return self._model

    def synthesize_bytes(self, text: str) -> tuple[bytes, int]:
        model = self._load_model()
        wavs, sample_rate = model.generate_voice_design(
            text=text,
            language=self.config.language,
            instruct=self.config.instruct,
            max_new_tokens=self.config.max_new_tokens,
        )
        if not wavs:
            raise RuntimeError("Qwen TTS returned no audio")

        pcm_bytes = _float_wave_to_discord_pcm_bytes(wavs[0], sample_rate)
        return pcm_bytes, DISCORD_PCM_SAMPLE_RATE


def _float_wave_to_discord_pcm_bytes(waveform, sample_rate: int) -> bytes:
    import numpy as np

    audio = np.asarray(waveform)
    if audio.ndim == 1:
        audio = np.repeat(audio[:, None], DISCORD_PCM_CHANNELS, axis=1)
    elif audio.ndim == 2:
        if audio.shape[0] in {1, 2} and audio.shape[1] > 8:
            audio = audio.T
        if audio.shape[1] == 1:
            audio = np.repeat(audio, DISCORD_PCM_CHANNELS, axis=1)
        elif audio.shape[1] != DISCORD_PCM_CHANNELS:
            raise ValueError(f"Unsupported waveform shape: {audio.shape}")
    else:
        raise ValueError(f"Unsupported waveform dimensions: {audio.ndim}")

    clipped = np.clip(audio, -1.0, 1.0)
    pcm_int16 = (clipped * 32767.0).astype(np.int16, copy=False)
    pcm_bytes = pcm_int16.tobytes()

    if sample_rate != DISCORD_PCM_SAMPLE_RATE:
        pcm_bytes, _ = audioop.ratecv(
            pcm_bytes,
            DISCORD_PCM_SAMPLE_WIDTH,
            DISCORD_PCM_CHANNELS,
            sample_rate,
            DISCORD_PCM_SAMPLE_RATE,
            None,
        )

    return pcm_bytes


def pcm_bytes_to_audio_source(pcm_bytes: bytes) -> "discord.AudioSource":
    import discord

    return discord.PCMAudio(io.BytesIO(pcm_bytes))
