import json
import mimetypes
import logging
import os
import shutil
import subprocess
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from typing import Any
from urllib.parse import parse_qs, urlparse

from . import logs
from .paths import resolve_db_dir, web_root
from .registry import Registry

logger = logging.getLogger("sandy.api")

try:
    import pynvml
except ImportError:  # pragma: no cover - tested via fallback path
    pynvml = None


def _nvml_number(value: object) -> int | float | None:
    if value is None or not isinstance(value, (int, float)):
        return None
    unavailable = getattr(pynvml, "NVML_VALUE_NOT_AVAILABLE", object()) if pynvml is not None else object()
    if int(value) == unavailable:
        return None
    return value


def _gpu_payload_from_nvml() -> dict[str, Any] | None:
    if pynvml is None:
        return None

    try:
        pynvml.nvmlInit()
    except Exception as exc:
        logger.debug("NVML init failed, falling back to nvidia-smi: %s", exc)
        return None

    try:
        devices: list[dict[str, Any]] = []
        for index in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8", errors="replace")
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            power_draw_mw = _nvml_number(pynvml.nvmlDeviceGetPowerUsage(handle))
            power_limit_mw = _nvml_number(pynvml.nvmlDeviceGetEnforcedPowerLimit(handle))
            temp_c = _nvml_number(
                pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU),
            )
            memory_used_mb = int(memory.used / (1024 * 1024))
            memory_total_mb = int(memory.total / (1024 * 1024))
            devices.append(
                {
                    "index": index,
                    "name": name,
                    "utilization_gpu_percent": int(utilization.gpu),
                    "memory_used_mb": memory_used_mb,
                    "memory_total_mb": memory_total_mb,
                    "memory_utilization_percent": round((memory_used_mb / memory_total_mb) * 100, 1)
                    if memory_total_mb
                    else 0.0,
                    "temperature_c": int(temp_c) if temp_c is not None else None,
                    "power_draw_w": round(power_draw_mw / 1000, 1) if power_draw_mw is not None else None,
                    "power_limit_w": round(power_limit_mw / 1000, 1) if power_limit_mw is not None else None,
                }
            )
        return {
            "available": True,
            "backend": "nvml",
            "devices": devices,
        }
    except Exception as exc:
        logger.debug("NVML query failed, falling back to nvidia-smi: %s", exc)
        return None
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


def _gpu_payload_from_nvidia_smi() -> dict[str, Any]:
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        return {
            "available": False,
            "backend": None,
            "devices": [],
            "error": "Neither NVML nor nvidia-smi is available",
        }

    query = ",".join([
        "index",
        "name",
        "utilization.gpu",
        "memory.used",
        "memory.total",
        "temperature.gpu",
        "power.draw",
        "power.limit",
    ])
    try:
        result = subprocess.run(
            [
                nvidia_smi,
                f"--query-gpu={query}",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            check=True,
            text=True,
            timeout=2,
        )
    except Exception as exc:
        return {
            "available": False,
            "backend": "nvidia-smi",
            "devices": [],
            "error": str(exc),
        }

    devices: list[dict[str, Any]] = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 8:
            continue
        index, name, gpu_util, mem_used, mem_total, temp, power_draw, power_limit = parts
        devices.append(
            {
                "index": int(index),
                "name": name,
                "utilization_gpu_percent": int(float(gpu_util)),
                "memory_used_mb": int(float(mem_used)),
                "memory_total_mb": int(float(mem_total)),
                "memory_utilization_percent": round((float(mem_used) / float(mem_total)) * 100, 1) if float(mem_total) else 0.0,
                "temperature_c": int(float(temp)),
                "power_draw_w": float(power_draw),
                "power_limit_w": float(power_limit),
            }
        )

    return {
        "available": True,
        "backend": "nvidia-smi",
        "devices": devices,
    }


def _gpu_payload() -> dict[str, Any]:
    return _gpu_payload_from_nvml() or _gpu_payload_from_nvidia_smi()


def _resolve_static_path(url_path: str, *, prefix: str, root: Path) -> Path | None:
    if not url_path.startswith(prefix):
        return None
    relative = url_path[len(prefix):].lstrip("/")
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError:
        return None
    if not candidate.is_file():
        return None
    return candidate


def _build_registry(*, test_mode: bool) -> Registry | None:
    original_db_dir = os.getenv("DB_DIR")
    try:
        os.environ["DB_DIR"] = str(resolve_db_dir(test_mode=test_mode))
        return Registry()
    except Exception:
        return None
    finally:
        if original_db_dir is None:
            os.environ.pop("DB_DIR", None)
        else:
            os.environ["DB_DIR"] = original_db_dir


def _enrich_trace_detail(detail: dict[str, Any], *, registry: Registry | None) -> dict[str, Any]:
    turn_input = detail.get("turn_input", {})
    timeline = detail.get("timeline", [])
    enriched_timeline: list[dict[str, Any]] = []

    for event in timeline:
        payload = dict(event)
        guild_id = payload.get("guild_id")
        channel_id = payload.get("channel_id")
        author_id = payload.get("author_id")

        if turn_input.get("guild_name"):
            payload.setdefault("guild_name", turn_input.get("guild_name"))
        if turn_input.get("channel_name"):
            payload.setdefault("channel_name", turn_input.get("channel_name"))
        if turn_input.get("author_name"):
            payload.setdefault("author_name", turn_input.get("author_name"))

        if registry is not None and channel_id is not None:
            channel_info = registry.get_channel_info(channel_id)
            if channel_info is not None:
                payload["channel_name"] = channel_info.get("channel_name") or payload.get("channel_name")
                payload["guild_name"] = channel_info.get("server_name") or payload.get("guild_name")
                guild_id = channel_info.get("server_id", guild_id)
                payload["guild_id"] = guild_id

        if registry is not None and author_id is not None:
            user_info = registry.get_user_info(author_id, guild_id)
            if user_info is not None:
                payload["author_name"] = (
                    user_info.get("nickname")
                    or user_info.get("user_name")
                    or payload.get("author_name")
                )
                payload["author_username"] = user_info.get("user_name")
                if user_info.get("nickname"):
                    payload["author_nickname"] = user_info.get("nickname")

        enriched_timeline.append(payload)

    return {
        **detail,
        "timeline": enriched_timeline,
    }


@dataclass(slots=True)
class ApiService:
    pipeline: Any
    runtime_state: Any
    test_mode: bool

    def status_payload(self) -> dict[str, Any]:
        runtime = self.runtime_state.snapshot()
        active_turns = runtime["active_turns"]
        current_turn = active_turns[0] if active_turns else None
        return {
            "mode": "test" if self.test_mode else "prod",
            "discord": runtime["discord"],
            "voice": runtime["voice"],
            "current_turn": current_turn,
            "active_turn_count": len(active_turns),
            "memory_worker": runtime["memory_worker"],
            "llm": {
                "busy": self.pipeline.llm.is_busy(),
            },
            "last_bouncer_decision": runtime["last_bouncer_decision"],
        }

    def gpu_payload(self) -> dict[str, Any]:
        return _gpu_payload()

    def recent_turns_payload(self, *, limit: int = 10, human_only: bool = False) -> dict[str, Any]:
        turns = logs.get_recent_turns(test_mode=self.test_mode, limit=limit, human_only=human_only)
        return {
            "turns": turns,
            "count": len(turns),
        }

    def trace_detail_payload(self, trace_id: str) -> dict[str, Any] | None:
        detail = logs.get_trace_detail(test_mode=self.test_mode, trace_id=trace_id)
        if detail is None:
            return None
        registry = _build_registry(test_mode=self.test_mode)
        return _enrich_trace_detail(detail, registry=registry)


class _ApiHandler(BaseHTTPRequestHandler):
    server_version = "SandyAPI/0.1"

    @property
    def api_service(self) -> ApiService:
        return self.server.api_service  # type: ignore[attr-defined]

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        query = parse_qs(parsed.query)

        try:
            if path in {"/", "/dashboard"}:
                self._write_file(HTTPStatus.OK, web_root() / "dashboard" / "index.html")
                return
            if path == "/favicon.svg":
                self._write_file(HTTPStatus.OK, web_root() / "dashboard" / "favicon.svg")
                return
            dashboard_asset = _resolve_static_path(
                path,
                prefix="/dashboard/",
                root=web_root() / "dashboard",
            )
            if dashboard_asset is not None:
                self._write_file(HTTPStatus.OK, dashboard_asset)
                return
            if path == "/api/status":
                self._write_json(HTTPStatus.OK, self.api_service.status_payload())
                return
            if path == "/api/gpu":
                self._write_json(HTTPStatus.OK, self.api_service.gpu_payload())
                return
            if path == "/api/turns/recent":
                limit = max(1, min(int(query.get("limit", ["10"])[0]), 100))
                human_only = query.get("human_only", ["false"])[0].lower() in {"1", "true", "yes"}
                self._write_json(
                    HTTPStatus.OK,
                    self.api_service.recent_turns_payload(limit=limit, human_only=human_only),
                )
                return
            if path.startswith("/api/turns/"):
                trace_id = path.split("/")[-1]
                detail = self.api_service.trace_detail_payload(trace_id)
                if detail is None:
                    self._write_json(HTTPStatus.NOT_FOUND, {"error": f"trace_id {trace_id!r} not found"})
                    return
                self._write_json(HTTPStatus.OK, detail)
                return
            self._write_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
        except ValueError as exc:
            self._write_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
        except Exception as exc:
            logger.exception("API request failed for %s", self.path)
            self._write_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})

    def log_message(self, fmt: str, *args: object) -> None:
        logger.debug("HTTP %s", fmt % args)

    def _write_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=True, sort_keys=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _write_file(self, status: HTTPStatus, path: Path) -> None:
        body = path.read_bytes()
        content_type, _ = mimetypes.guess_type(path.name)
        content_type = content_type or "application/octet-stream"
        self.send_response(status)
        if content_type.startswith("text/") or content_type in {"application/javascript", "application/json"}:
            content_type += "; charset=utf-8"
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


class ApiServer:
    def __init__(self, service: ApiService, *, host: str, port: int) -> None:
        self._server = ThreadingHTTPServer((host, port), _ApiHandler)
        self._server.api_service = service  # type: ignore[attr-defined]
        self._thread = Thread(target=self._server.serve_forever, name="sandy-api", daemon=True)

    @property
    def address(self) -> tuple[str, int]:
        host, port = self._server.server_address[:2]
        return str(host), int(port)

    def start(self) -> None:
        self._thread.start()

    def shutdown(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=2)


def api_enabled() -> bool:
    return os.getenv("SANDY_API_ENABLED", "true").strip().lower() not in {"0", "false", "no"}


def api_host() -> str:
    return os.getenv("SANDY_API_HOST", "127.0.0.1").strip() or "127.0.0.1"


def api_port() -> int:
    return int(os.getenv("SANDY_API_PORT", "8765"))
