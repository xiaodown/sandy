from __future__ import annotations

import io
import logging
import os
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from faster_qwen3_tts import FasterQwen3TTS
from pydantic import BaseModel

load_dotenv()

logger = logging.getLogger("tts_service_fast")
WAV_CHANNELS = 1
WAV_SAMPLE_WIDTH = 2


@dataclass(frozen=True, slots=True)
class ServiceConfig:
    model_name: str = os.getenv("TTS_SERVICE_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    language: str = os.getenv("TTS_SERVICE_LANGUAGE", "English")
    clone_ref_audio_path: str = os.getenv(
        "TTS_SERVICE_CLONE_REF_AUDIO",
        "/home/xiaodown/code/sandy/tts_service/assets/clone_reference.wav",
    )
    clone_ref_text: str = os.getenv(
        "TTS_SERVICE_CLONE_REF_TEXT",
        "goodnight robst. sleep well, dream of snacks and revenge.",
    )
    clone_xvec_only: bool = os.getenv(
        "TTS_SERVICE_CLONE_XVECTOR_ONLY",
        "true",
    ).strip().lower() in {"1", "true", "yes", "on"}
    device: str = os.getenv("TTS_SERVICE_DEVICE", "cuda")
    dtype: str = os.getenv("TTS_SERVICE_DTYPE", "bfloat16")
    attn_implementation: str = os.getenv("TTS_SERVICE_ATTN_IMPLEMENTATION", "sdpa")
    max_seq_len: int = int(os.getenv("TTS_SERVICE_MAX_SEQ_LEN", "2048"))
    max_new_tokens: int = int(os.getenv("TTS_SERVICE_MAX_NEW_TOKENS", "512"))
    min_new_tokens: int = int(os.getenv("TTS_SERVICE_MIN_NEW_TOKENS", "2"))
    do_sample: bool = os.getenv("TTS_SERVICE_DO_SAMPLE", "false").strip().lower() in {
        "1", "true", "yes", "on",
    }
    temperature: float = float(os.getenv("TTS_SERVICE_TEMPERATURE", "0.9"))
    top_k: int = int(os.getenv("TTS_SERVICE_TOP_K", "50"))
    top_p: float = float(os.getenv("TTS_SERVICE_TOP_P", "1.0"))
    repetition_penalty: float = float(os.getenv("TTS_SERVICE_REPETITION_PENALTY", "1.05"))
    non_streaming_mode: bool = os.getenv("TTS_SERVICE_NON_STREAMING_MODE", "false").strip().lower() in {
        "1", "true", "yes", "on",
    }
    max_audio_seconds: float = float(os.getenv("TTS_SERVICE_MAX_AUDIO_SECONDS", "20"))
    warmup_text: str = os.getenv("TTS_SERVICE_WARMUP_TEXT", "hello there")


class SynthesizeRequest(BaseModel):
    text: str
    language: str | None = None
    instruct: str | None = None


class FasterCloneService:
    def __init__(self, config: ServiceConfig) -> None:
        self.config = config
        self._model: FasterQwen3TTS | None = None

    def _resolve_dtype(self) -> torch.dtype:
        name = self.config.dtype.lower()
        if name in {"bfloat16", "bf16"}:
            return torch.bfloat16
        if name in {"float16", "fp16"}:
            return torch.float16
        if name in {"float32", "fp32"}:
            return torch.float32
        raise ValueError(f"Unsupported TTS dtype: {self.config.dtype}")

    def _load_model(self) -> FasterQwen3TTS:
        if self._model is not None:
            return self._model

        logger.info(
            "Loading FasterQwen3TTS model=%s device=%s dtype=%s attn=%s max_seq_len=%s",
            self.config.model_name,
            self.config.device,
            self.config.dtype,
            self.config.attn_implementation,
            self.config.max_seq_len,
        )
        self._model = FasterQwen3TTS.from_pretrained(
            self.config.model_name,
            device=self.config.device,
            dtype=self._resolve_dtype(),
            attn_implementation=self.config.attn_implementation,
            max_seq_len=self.config.max_seq_len,
        )
        logger.info("Loaded FasterQwen3TTS model=%s", self.config.model_name)
        return self._model

    def warmup(self) -> None:
        self.synthesize_wav_bytes(self.config.warmup_text)

    def synthesize_wav_bytes(
        self,
        text: str,
        *,
        language: str | None = None,
    ) -> bytes:
        model = self._load_model()
        ref_audio = Path(self.config.clone_ref_audio_path)
        if not ref_audio.exists():
            raise FileNotFoundError(f"Clone reference audio not found: {ref_audio}")
        if not self.config.clone_ref_text.strip():
            raise ValueError("TTS_SERVICE_CLONE_REF_TEXT must not be empty")

        audio_list, sample_rate = model.generate_voice_clone(
            text=text,
            language=language or self.config.language,
            ref_audio=str(ref_audio),
            ref_text=self.config.clone_ref_text,
            max_new_tokens=self.config.max_new_tokens,
            min_new_tokens=self.config.min_new_tokens,
            temperature=self.config.temperature,
            top_k=self.config.top_k,
            top_p=self.config.top_p,
            do_sample=self.config.do_sample,
            repetition_penalty=self.config.repetition_penalty,
            xvec_only=self.config.clone_xvec_only,
            non_streaming_mode=self.config.non_streaming_mode,
        )
        if not audio_list:
            raise RuntimeError("FasterQwen3TTS returned no audio")
        waveform = np.asarray(audio_list[0])
        duration = _waveform_duration_seconds(waveform, sample_rate)
        if duration > self.config.max_audio_seconds:
            raise RuntimeError(
                "FasterQwen3TTS generated audio that exceeded the configured limit: "
                f"{duration:.2f}s > {self.config.max_audio_seconds:.2f}s",
            )
        return _waveform_to_wav_bytes(waveform, sample_rate)


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


def _waveform_duration_seconds(waveform, sample_rate: int) -> float:
    audio = np.asarray(waveform)
    if sample_rate <= 0:
        raise ValueError(f"Invalid sample rate: {sample_rate}")
    if audio.ndim == 1:
        frame_count = audio.shape[0]
    elif audio.ndim == 2:
        frame_count = max(audio.shape)
    else:
        raise ValueError(f"Unsupported waveform dimensions: {audio.ndim}")
    return frame_count / float(sample_rate)


def create_app() -> FastAPI:
    app = FastAPI()
    service = FasterCloneService(ServiceConfig())

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/warmup")
    async def warmup() -> dict[str, str]:
        try:
            service.warmup()
        except Exception as exc:
            logger.exception("Fast TTS warmup failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return {"status": "warmed"}

    @app.post("/synthesize")
    async def synthesize(request: SynthesizeRequest) -> Response:
        text = request.text.strip()
        if not text:
            raise HTTPException(status_code=400, detail="text must not be empty")
        try:
            wav_bytes = service.synthesize_wav_bytes(
                text,
                language=request.language.strip() if request.language else None,
            )
        except Exception as exc:
            logger.exception("Fast TTS synthesis failed for text=%r", text)
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return Response(content=wav_bytes, media_type="audio/wav")

    return app
