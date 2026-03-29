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
        package_names = (
            "nvidia.cublas",
            "nvidia.cudnn",
            "nvidia.cuda_nvrtc",
        )
        directories: list[Path] = []
        for package_name in package_names:
            spec = importlib.util.find_spec(package_name)
            if spec is None or not spec.submodule_search_locations:
                continue

            # The pip-installed NVIDIA CUDA wheels use namespace packages, so
            # modules like nvidia.cublas.lib often have no __file__. Resolve
            # the package root from submodule_search_locations instead and then
            # point ctranslate2 at the sibling lib/ directory explicitly.
            directory = Path(next(iter(spec.submodule_search_locations))).resolve() / "lib"
            if directory.exists() and directory not in directories:
                directories.append(directory)

        return directories

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

        started = time.perf_counter()
        logger.info(
            "Loading faster-whisper model=%s device=%s compute_type=%s",
            self.model_name,
            self.device,
            self.compute_type,
        )
        try:
            model = WhisperModel(
                self.model_name,
                device=self.device,
                compute_type=self.compute_type,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load faster-whisper on configured device={self.device!r} "
                f"compute_type={self.compute_type!r}. "
                "This voice spike assumes the configured runtime libraries are present."
            ) from exc

        elapsed = time.perf_counter() - started
        self._model = model
        self._resolved_device = self.device
        self._resolved_compute_type = self.compute_type
        logger.info(
            "Loaded faster-whisper model=%s device=%s compute_type=%s in %.2fs",
            self.model_name,
            self.device,
            self.compute_type,
            elapsed,
        )
        return model

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
