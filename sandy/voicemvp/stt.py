from __future__ import annotations

import asyncio
import ctypes
from dataclasses import dataclass
import importlib.util
import logging
from pathlib import Path
import time

logger = logging.getLogger("sandy.voicemvp.stt")


@dataclass(frozen=True, slots=True)
class TranscriptResult:
    text: str
    language: str | None
    language_probability: float | None
    elapsed_seconds: float
    device: str
    compute_type: str


class FasterWhisperTranscriber:
    def __init__(
        self,
        *,
        model_name: str,
        device: str,
        compute_type: str,
        language: str | None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self._model = None
        self._resolved_device: str | None = None
        self._resolved_compute_type: str | None = None

    def _cuda_library_dirs(self) -> list[Path]:
        spec = importlib.util.find_spec("nvidia")
        if spec is None or not spec.submodule_search_locations:
            return []

        base = Path(next(iter(spec.submodule_search_locations)))
        candidates = [
            base / "cublas" / "lib",
            base / "cudnn" / "lib",
            base / "cuda_nvrtc" / "lib",
        ]
        return [path for path in candidates if path.exists()]

    def _try_preload_cuda_runtime(self) -> None:
        if self.device != "cuda":
            return

        library_paths = [
            "libcublas.so.12",
            "libcublasLt.so.12",
            "libcudnn.so.9",
            "libcudnn_ops.so.9",
            "libcudnn_cnn.so.9",
            "libnvrtc.so.12",
        ]
        directories = self._cuda_library_dirs()
        if not directories:
            return

        for name in library_paths:
            for directory in directories:
                path = directory / name
                if not path.exists():
                    continue
                try:
                    ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
                except OSError as exc:
                    logger.debug("Failed to preload %s: %r", path, exc)
                break

    def _load_model(self):
        if self._model is not None:
            return self._model

        self._try_preload_cuda_runtime()
        from faster_whisper import WhisperModel

        attempts = [(self.device, self.compute_type)]
        if self.device != "cpu":
            attempts.append(("cpu", "int8"))

        last_error: Exception | None = None
        for device, compute_type in attempts:
            try:
                started = time.perf_counter()
                logger.info(
                    "Loading faster-whisper model=%s device=%s compute_type=%s",
                    self.model_name,
                    device,
                    compute_type,
                )
                model = WhisperModel(
                    self.model_name,
                    device=device,
                    compute_type=compute_type,
                )
                elapsed = time.perf_counter() - started
                self._model = model
                self._resolved_device = device
                self._resolved_compute_type = compute_type
                logger.info(
                    "Loaded faster-whisper model=%s device=%s compute_type=%s in %.2fs",
                    self.model_name,
                    device,
                    compute_type,
                    elapsed,
                )
                return model
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "Failed to load faster-whisper model=%s device=%s compute_type=%s: %r",
                    self.model_name,
                    device,
                    compute_type,
                    exc,
                )

        raise RuntimeError(
            f"Failed to load faster-whisper model {self.model_name!r}"
        ) from last_error

    def transcribe_file_sync(self, path: Path) -> TranscriptResult:
        model = self._load_model()
        started = time.perf_counter()
        segments, info = model.transcribe(
            str(path),
            language=self.language,
            beam_size=1,
            vad_filter=False,
            condition_on_previous_text=False,
            without_timestamps=True,
        )
        text = " ".join(segment.text.strip() for segment in segments).strip()
        elapsed = time.perf_counter() - started
        return TranscriptResult(
            text=text,
            language=getattr(info, "language", None),
            language_probability=getattr(info, "language_probability", None),
            elapsed_seconds=elapsed,
            device=self._resolved_device or self.device,
            compute_type=self._resolved_compute_type or self.compute_type,
        )

    async def transcribe_file(self, path: Path) -> TranscriptResult:
        return await asyncio.to_thread(self.transcribe_file_sync, path)
