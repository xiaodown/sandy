from __future__ import annotations

import io
import logging
import os
import wave
from dataclasses import dataclass

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

load_dotenv()

logger = logging.getLogger("tts_service")
WAV_CHANNELS = 1
WAV_SAMPLE_WIDTH = 2


@dataclass(frozen=True, slots=True)
class ServiceConfig:
    model_name: str = os.getenv("TTS_SERVICE_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign")
    language: str = os.getenv("TTS_SERVICE_LANGUAGE", "English")
    instruct: str = os.getenv(
        "TTS_SERVICE_INSTRUCT",
        (
            "Voice Identity: warm, mid-range, female-coded, calm and slightly amused, "
            "natural and conversational, medium speaking pace, clear diction, gentle "
            "dry humor, not cutesy, not breathy, not theatrical."
        ),
    )
    device_map: str = os.getenv("TTS_SERVICE_DEVICE_MAP", "cuda:0")
    dtype: str = os.getenv("TTS_SERVICE_DTYPE", "bfloat16")
    attn_implementation: str = os.getenv("TTS_SERVICE_ATTN_IMPLEMENTATION", "sdpa")
    max_new_tokens: int = int(os.getenv("TTS_SERVICE_MAX_NEW_TOKENS", "2048"))


class SynthesizeRequest(BaseModel):
    text: str


class QwenVoiceDesignService:
    def __init__(self, config: ServiceConfig) -> None:
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
            raise ValueError(f"Unsupported TTS dtype: {self.config.dtype}")

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

    def synthesize_wav_bytes(self, text: str) -> bytes:
        model = self._load_model()
        wavs, sample_rate = model.generate_voice_design(
            text=text,
            language=self.config.language,
            instruct=self.config.instruct,
            max_new_tokens=self.config.max_new_tokens,
        )
        if not wavs:
            raise RuntimeError("Qwen TTS returned no audio")
        return _waveform_to_wav_bytes(wavs[0], sample_rate)


def _waveform_to_wav_bytes(waveform, sample_rate: int) -> bytes:
    audio = np.asarray(waveform)
    if audio.ndim == 2:
        if audio.shape[0] == 1:
            audio = audio[0]
        elif audio.shape[1] == 1:
            audio = audio[:, 0]
        else:
            raise ValueError(f"Unsupported waveform shape: {audio.shape}")
    elif audio.ndim != 1:
        raise ValueError(f"Unsupported waveform dimensions: {audio.ndim}")

    clipped = np.clip(audio, -1.0, 1.0)
    pcm_int16 = (clipped * 32767.0).astype(np.int16, copy=False)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav_file:
        wav_file.setnchannels(WAV_CHANNELS)
        wav_file.setsampwidth(WAV_SAMPLE_WIDTH)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_int16.tobytes())
    return buf.getvalue()


def create_app() -> FastAPI:
    app = FastAPI()
    service = QwenVoiceDesignService(ServiceConfig())

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/synthesize")
    async def synthesize(request: SynthesizeRequest) -> Response:
        text = request.text.strip()
        if not text:
            raise HTTPException(status_code=400, detail="text must not be empty")
        try:
            wav_bytes = service.synthesize_wav_bytes(text)
        except Exception as exc:
            logger.exception("TTS synthesis failed for text=%r", text)
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return Response(content=wav_bytes, media_type="audio/wav")

    return app
