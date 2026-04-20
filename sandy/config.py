"""Centralized configuration for the Sandy bot.

All environment variable reads happen here.  Modules receive typed config
slices instead of calling ``os.getenv()`` themselves.

Usage::

    from .config import SandyConfig
    cfg = SandyConfig.from_env()       # once, at startup
    llm = OllamaInterface(cfg.llm)     # pass slices
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

from .paths import resolve_runtime_path


# ── Sub-configs ────────────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class LlmConfig:
    """Model names, temperatures, context sizes, and generation limits."""

    brain_model: str = "qwen2.5:14b"
    bouncer_model: str = "qwen2.5:14b"
    tagger_model: str = "hf.co/bartowski/Llama-3.2-3B-Instruct-GGUF:Q8_0"
    summarizer_model: str = "hf.co/bartowski/Llama-3.2-3B-Instruct-GGUF:Q8_0"
    vision_model: str | None = None
    vision_router_model: str | None = None

    brain_temperature: float = 1.1
    bouncer_temperature: float = 0.1
    tagger_temperature: float = 0.1
    summarizer_temperature: float = 0.1
    vision_temperature: float = 0.3
    vision_router_temperature: float = 0.1

    brain_num_predict: int = 512
    voice_brain_num_predict: int = 80
    brain_num_ctx: int = 8192
    bouncer_num_ctx: int = 8192
    tagger_num_ctx: int = 4096
    summarizer_num_ctx: int = 4096
    vision_num_ctx: int = 4096
    vision_num_predict: int = 224
    vision_router_num_ctx: int = 2048
    vision_router_num_predict: int = 48
    prewarm_num_ctx: int | None = None  # defaults to bouncer_num_ctx

    keep_alive: str = "1h"

    @property
    def effective_prewarm_num_ctx(self) -> int:
        return self.prewarm_num_ctx if self.prewarm_num_ctx is not None else self.bouncer_num_ctx


@dataclass(frozen=True, slots=True)
class VoiceConfig:
    """Voice pipeline: STT, TTS, capture, timing, reply limits."""

    stt_model: str = "base.en"
    stt_device: str = "cuda"
    stt_compute_type: str = "float16"
    stt_language: str | None = "en"

    tts_service_url: str = "http://127.0.0.1:8777"
    tts_service_timeout_seconds: float = 180.0
    tts_instruct: str | None = None
    tts_language: str = "English"

    capture_dir: str = "data/prod/voice_captures"
    preroll_ms: int = 250

    stitch_gap_seconds: float = 1.0
    stitch_release_seconds: float = 1.35
    force_release_seconds: float = 3.25

    history_maxlen: int = 12
    idle_auto_leave_seconds: int = 300

    reply_max_words: int = 32
    reply_max_chars: int = 220
    reply_max_sentences: int = 2


@dataclass(frozen=True, slots=True)
class StorageConfig:
    """Database directories and file names."""

    db_dir: str = "data/prod/"
    recall_db_name: str = "recall.db"
    server_db_name: str = "server.db"
    embed_model: str = "mxbai-embed-large"
    vector_max_distance: float = 0.6
    summarize_threshold: int = 144

    @property
    def resolved_db_dir(self) -> Path:
        return resolve_runtime_path(self.db_dir)

    @property
    def recall_db_path(self) -> Path:
        return self.resolved_db_dir / self.recall_db_name

    @property
    def server_db_path(self) -> Path:
        return self.resolved_db_dir / self.server_db_name


@dataclass(frozen=True, slots=True)
class SearchConfig:
    """Web search and Steam browsing."""

    searxng_host: str = "127.0.0.1"
    searxng_port: str = "8888"
    steam_cache_ttl_seconds: int = 600

    @property
    def searxng_base_url(self) -> str:
        return f"http://{self.searxng_host}:{self.searxng_port}"


@dataclass(frozen=True, slots=True)
class LogConfig:
    """Logging rotation and trace retention."""

    rotate_bytes: int = 20 * 1024 * 1024
    backup_count: int = 10
    trace_retention_days: int = 14


@dataclass(frozen=True, slots=True)
class ApiConfig:
    """Observability API settings."""

    enabled: bool = True
    host: str = "127.0.0.1"
    port: int = 8765


# ── Top-level config ───────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class SandyConfig:
    """Complete Sandy configuration, constructed once at startup."""

    llm: LlmConfig = field(default_factory=LlmConfig)
    voice: VoiceConfig = field(default_factory=VoiceConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    log: LogConfig = field(default_factory=LogConfig)
    api: ApiConfig = field(default_factory=ApiConfig)

    discord_api_key: str | None = None

    prewarm_enabled: bool = False
    prewarm_model_name: str | None = None
    test_mode: bool = False

    @classmethod
    def from_env(cls, *, test_mode: bool = False) -> SandyConfig:
        """Read all environment variables and return a fully populated config."""
        load_dotenv(find_dotenv(usecwd=True))

        def _str(name: str, default: str) -> str:
            return os.getenv(name, default)

        def _int(name: str, default: int) -> int:
            return int(os.getenv(name, str(default)))

        def _float(name: str, default: float) -> float:
            return float(os.getenv(name, str(default)))

        def _opt_str(name: str) -> str | None:
            v = os.getenv(name)
            return v.strip() or None if v else None

        def _bool_flag(name: str, default: bool = False) -> bool:
            v = os.getenv(name, "").strip().lower()
            if not v:
                return default
            return v not in {"0", "false", "no"}

        # DB_DIR may have been overridden by --test in __main__.py
        db_dir = _str("DB_DIR", "data/test/" if test_mode else "data/prod/")

        bouncer_num_ctx = _int("BOUNCER_NUM_CTX", 8192)

        # Frozen slotted dataclasses don't expose defaults as class attrs.
        # Instantiate defaults once as a reference.
        _llm = LlmConfig()
        _voice = VoiceConfig()
        _storage = StorageConfig()
        _search = SearchConfig()
        _log = LogConfig()
        _api = ApiConfig()

        return cls(
            llm=LlmConfig(
                brain_model=_str("BRAIN_MODEL", _llm.brain_model),
                bouncer_model=_str("BOUNCER_MODEL", _llm.bouncer_model),
                tagger_model=_str("TAGGER_MODEL", _llm.tagger_model),
                summarizer_model=_str("SUMMARIZER_MODEL", _llm.summarizer_model),
                vision_model=_opt_str("VISION_MODEL"),
                vision_router_model=_opt_str("VISION_ROUTER_MODEL"),
                brain_temperature=_float("BRAIN_TEMPERATURE", _llm.brain_temperature),
                bouncer_temperature=_float("BOUNCER_TEMPERATURE", _llm.bouncer_temperature),
                tagger_temperature=_float("TAGGER_TEMPERATURE", _llm.tagger_temperature),
                summarizer_temperature=_float("SUMMARIZER_TEMPERATURE", _llm.summarizer_temperature),
                vision_temperature=_float("VISION_TEMPERATURE", _llm.vision_temperature),
                vision_router_temperature=_float("VISION_ROUTER_TEMPERATURE", _llm.vision_router_temperature),
                brain_num_predict=_int("BRAIN_NUM_PREDICT", _llm.brain_num_predict),
                voice_brain_num_predict=_int("VOICE_BRAIN_NUM_PREDICT", _llm.voice_brain_num_predict),
                brain_num_ctx=_int("BRAIN_NUM_CTX", _llm.brain_num_ctx),
                bouncer_num_ctx=bouncer_num_ctx,
                tagger_num_ctx=_int("TAGGER_NUM_CTX", _llm.tagger_num_ctx),
                summarizer_num_ctx=_int("SUMMARIZER_NUM_CTX", _llm.summarizer_num_ctx),
                vision_num_ctx=_int("VISION_NUM_CTX", _llm.vision_num_ctx),
                vision_num_predict=_int("VISION_NUM_PREDICT", _llm.vision_num_predict),
                vision_router_num_ctx=_int("VISION_ROUTER_NUM_CTX", _llm.vision_router_num_ctx),
                vision_router_num_predict=_int("VISION_ROUTER_NUM_PREDICT", _llm.vision_router_num_predict),
                prewarm_num_ctx=_int("PREWARM_NUM_CTX", bouncer_num_ctx),
                keep_alive=_str("OLLAMA_KEEP_ALIVE", _llm.keep_alive),
            ),
            voice=VoiceConfig(
                stt_model=_str("VOICE_STT_MODEL", _voice.stt_model),
                stt_device=_str("VOICE_STT_DEVICE", _voice.stt_device),
                stt_compute_type=_str("VOICE_STT_COMPUTE_TYPE", _voice.stt_compute_type),
                stt_language=_opt_str("VOICE_STT_LANGUAGE") or _voice.stt_language,
                tts_service_url=_str("VOICE_TTS_SERVICE_URL", _voice.tts_service_url),
                tts_service_timeout_seconds=_float("VOICE_TTS_SERVICE_TIMEOUT_SECONDS", _voice.tts_service_timeout_seconds),
                tts_instruct=_opt_str("VOICE_TTS_INSTRUCT"),
                tts_language=_str("VOICE_TTS_LANGUAGE", _voice.tts_language) or _voice.tts_language,
                capture_dir=_str("VOICE_CAPTURE_DIR", f"{db_dir.rstrip('/')}/voice_captures" if not test_mode else "data/test/voice_captures"),
                preroll_ms=_int("VOICE_PREROLL_MS", _voice.preroll_ms),
                stitch_gap_seconds=_float("VOICE_STITCH_GAP_SECONDS", _voice.stitch_gap_seconds),
                stitch_release_seconds=_float("VOICE_STITCH_RELEASE_SECONDS", _voice.stitch_release_seconds),
                force_release_seconds=_float("VOICE_FORCE_RELEASE_SECONDS", _voice.force_release_seconds),
                history_maxlen=_int("VOICE_HISTORY_MAXLEN", _voice.history_maxlen),
                idle_auto_leave_seconds=_int("VOICE_IDLE_AUTO_LEAVE_SECONDS", _voice.idle_auto_leave_seconds),
                reply_max_words=_int("VOICE_REPLY_MAX_WORDS", _voice.reply_max_words),
                reply_max_chars=_int("VOICE_REPLY_MAX_CHARS", _voice.reply_max_chars),
                reply_max_sentences=_int("VOICE_REPLY_MAX_SENTENCES", _voice.reply_max_sentences),
            ),
            storage=StorageConfig(
                db_dir=db_dir,
                recall_db_name=_str("RECALL_DB_NAME", _storage.recall_db_name),
                server_db_name=_str("SERVER_DB_NAME", _storage.server_db_name),
                embed_model=_str("EMBED_MODEL", _storage.embed_model),
                vector_max_distance=_float("VECTOR_MAX_DISTANCE", _storage.vector_max_distance),
                summarize_threshold=_int("SUMMARIZE_THRESHOLD", _storage.summarize_threshold),
            ),
            search=SearchConfig(
                searxng_host=_str("SEARXNG_HOST", _search.searxng_host),
                searxng_port=_str("SEARXNG_PORT", _search.searxng_port),
                steam_cache_ttl_seconds=_int("STEAM_BROWSE_CACHE_TTL_SECONDS", _search.steam_cache_ttl_seconds),
            ),
            log=LogConfig(
                rotate_bytes=_int("LOG_ROTATE_BYTES", _log.rotate_bytes),
                backup_count=_int("LOG_BACKUP_COUNT", _log.backup_count),
                trace_retention_days=_int("TRACE_RETENTION_DAYS", _log.trace_retention_days),
            ),
            api=ApiConfig(
                enabled=_bool_flag("SANDY_API_ENABLED", default=True),
                host=_str("SANDY_API_HOST", _api.host).strip() or _api.host,
                port=_int("SANDY_API_PORT", _api.port),
            ),
            discord_api_key=_opt_str("DISCORD_API_KEY"),
            prewarm_enabled=_str("PREWARM_MODEL", "") == "True",
            prewarm_model_name=_opt_str("PREWARM_MODEL_NAME"),
            test_mode=test_mode,
        )
