"""
Centralized logging configuration for Sandy.

Logs are split into:
  - human-readable console output
  - rotating JSONL files on disk
  - a small SQLite trace-event store for turn inspection

The root logger writes through a QueueHandler so normal app code never blocks
on I/O. A QueueListener drains records on a background thread.
"""

import atexit
from dotenv import load_dotenv
import json
import logging
import logging.handlers
import os
import queue
import sqlite3
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from .paths import resolve_runtime_path

load_dotenv()

_ANSI_RESET = "\033[0m"
_ANSI_DIM = "\033[2m"
_LEVEL_COLORS = {
    "DEBUG": "\033[36m",
    "INFO": "\033[32m",
    "WARNING": "\033[33m",
    "ERROR": "\033[31m",
    "CRITICAL": "\033[41;97m",
}
_LOGGER_COLORS = (
    "\033[38;5;39m",
    "\033[38;5;45m",
    "\033[38;5;81m",
    "\033[38;5;112m",
    "\033[38;5;141m",
    "\033[38;5;172m",
)


def _logs_dir() -> Path:
    db_dir = resolve_runtime_path(os.getenv("DB_DIR", "data/prod/"))
    path = db_dir / "logs"
    path.mkdir(parents=True, exist_ok=True)
    return path


class JsonlFormatter(logging.Formatter):
    """Render log records as one JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        event_payload = getattr(record, "event_payload", None)
        forensic_payload = getattr(record, "forensic_payload", None)
        if event_payload is not None:
            payload["record_type"] = "trace"
            payload["event"] = event_payload
        elif forensic_payload is not None:
            payload["record_type"] = "forensic"
            payload["forensic"] = forensic_payload
        else:
            payload["record_type"] = "log"
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=True, sort_keys=True)


class ConsoleFormatter(logging.Formatter):
    """Human-friendly ANSI console formatter with stable source coloring."""

    def __init__(self) -> None:
        super().__init__(datefmt="%Y-%m-%d %H:%M:%S")

    def format(self, record: logging.LogRecord) -> str:
        timestamp = self.formatTime(record, self.datefmt)
        separator = f"{_ANSI_DIM} | {_ANSI_RESET}"
        level_color = _LEVEL_COLORS.get(record.levelname, "")
        level = f"{level_color}{record.levelname:<8}{_ANSI_RESET}"
        logger_color = _LOGGER_COLORS[hash(record.name) % len(_LOGGER_COLORS)]
        logger_name = f"{logger_color}{record.name}{_ANSI_RESET}"
        message = record.getMessage()
        formatted = (
            f"{_ANSI_DIM}{timestamp}{_ANSI_RESET}"
            f"{separator}{level}"
            f"{separator}{logger_name}"
            f"{separator}{message}"
        )
        if record.exc_info:
            formatted += "\n" + self.formatException(record.exc_info)
        return formatted


class _SinkFilter(logging.Filter):
    """Route records to specific sinks via per-record boolean flags."""

    def __init__(self, sink_name: str) -> None:
        super().__init__()
        self._attr = f"log_to_{sink_name}"

    def filter(self, record: logging.LogRecord) -> bool:
        return bool(getattr(record, self._attr, True))


class _HttpxConsoleFilter(logging.Filter):
    """Keep httpx request spam out of the human console while preserving file logs."""

    def filter(self, record: logging.LogRecord) -> bool:
        return not (record.name == "httpx" and record.levelno < logging.WARNING)


class TraceStoreHandler(logging.Handler):
    """Persist structured trace events to a small local SQLite database."""

    def __init__(self, db_path: Path) -> None:
        super().__init__(level=logging.INFO)
        self._db_path = db_path
        self._lock = threading.Lock()
        self._emit_count = 0
        self._retention_days = int(os.getenv("TRACE_RETENTION_DAYS", "14"))
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS trace_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    trace_id TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    status TEXT NOT NULL,
                    message_id INTEGER,
                    guild_id INTEGER,
                    channel_id INTEGER,
                    author_id INTEGER,
                    payload_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_trace_events_created_at ON trace_events(created_at)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_trace_events_trace_id ON trace_events(trace_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_trace_events_stage ON trace_events(stage)"
            )

    def emit(self, record: logging.LogRecord) -> None:
        event_payload = getattr(record, "event_payload", None)
        if not isinstance(event_payload, dict):
            return

        created_at = datetime.fromtimestamp(record.created, UTC).isoformat()
        payload_json = json.dumps(event_payload, ensure_ascii=True, sort_keys=True)

        try:
            with self._lock, self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO trace_events (
                        created_at,
                        trace_id,
                        stage,
                        status,
                        message_id,
                        guild_id,
                        channel_id,
                        author_id,
                        payload_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        created_at,
                        event_payload.get("trace_id"),
                        event_payload.get("stage"),
                        event_payload.get("status", "ok"),
                        event_payload.get("message_id"),
                        event_payload.get("guild_id"),
                        event_payload.get("channel_id"),
                        event_payload.get("author_id"),
                        payload_json,
                    ),
                )
                self._emit_count += 1
                if self._emit_count % 100 == 0:
                    cutoff = datetime.now(UTC) - timedelta(days=self._retention_days)
                    conn.execute(
                        "DELETE FROM trace_events WHERE created_at < ?",
                        (cutoff.isoformat(),),
                    )
        except Exception:
            self.handleError(record)


_console_formatter = ConsoleFormatter()
_console_handler = logging.StreamHandler()
_console_handler.setFormatter(_console_formatter)
_console_handler.addFilter(_SinkFilter("console"))
_console_handler.addFilter(_HttpxConsoleFilter())

_jsonl_handler = logging.handlers.RotatingFileHandler(
    _logs_dir() / "sandy.jsonl",
    maxBytes=int(os.getenv("LOG_ROTATE_BYTES", str(20 * 1024 * 1024))),
    backupCount=int(os.getenv("LOG_BACKUP_COUNT", "10")),
    encoding="utf-8",
)
_jsonl_handler.setFormatter(JsonlFormatter())
_jsonl_handler.addFilter(_SinkFilter("jsonl"))

_trace_store_handler = TraceStoreHandler(_logs_dir() / "trace_events.db")
_trace_store_handler.addFilter(_SinkFilter("trace_store"))

_log_queue: queue.SimpleQueue[logging.LogRecord] = queue.SimpleQueue()
_queue_handler = logging.handlers.QueueHandler(_log_queue)
_listener = logging.handlers.QueueListener(
    _log_queue,
    _console_handler,
    _jsonl_handler,
    _trace_store_handler,
    respect_handler_level=True,
)
_listener.start()
atexit.register(_listener.stop)

logging.root.setLevel(logging.INFO)
logging.root.handlers.clear()
logging.root.addHandler(_queue_handler)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def emit_forensic_record(
    logger: logging.Logger,
    message: str,
    payload: dict[str, Any],
) -> None:
    """Write a forensic artifact to JSONL without polluting the console or trace DB."""
    logger.info(
        message,
        extra={
            "forensic_payload": payload,
            "log_to_console": False,
            "log_to_trace_store": False,
        },
    )
