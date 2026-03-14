"""Startup and operator health checks for Sandy.

This module classifies checks into:
  - hard requirements: startup should fail if these are broken
  - soft dependencies: startup should warn, but Sandy may still run in degraded mode

It also exposes a small CLI via ``python -m sandy.health`` for explicit operator
preflight checks over SSH.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import httpx
import ollama
from dotenv import load_dotenv

from .recall import ChatDatabase
from .registry import Registry
from .vector_memory import VectorMemory

load_dotenv()

logger = logging.getLogger("sandy.health")

_INT_ENV_VARS: tuple[str, ...] = (
    "BRAIN_NUM_PREDICT",
    "BRAIN_NUM_CTX",
    "BOUNCER_NUM_CTX",
    "TAGGER_NUM_CTX",
    "SUMMARIZER_NUM_CTX",
    "VISION_NUM_CTX",
    "VISION_NUM_PREDICT",
    "PREWARM_NUM_CTX",
    "SUMMARIZE_THRESHOLD",
    "LOG_ROTATE_BYTES",
    "LOG_BACKUP_COUNT",
    "TRACE_RETENTION_DAYS",
    "STEAM_BROWSE_CACHE_TTL_SECONDS",
    "SEARXNG_PORT",
)
_FLOAT_ENV_VARS: tuple[str, ...] = (
    "BRAIN_TEMPERATURE",
    "BOUNCER_TEMPERATURE",
    "TAGGER_TEMPERATURE",
    "SUMMARIZER_TEMPERATURE",
    "VECTOR_MAX_DISTANCE",
)

_MODEL_DEFAULTS: dict[str, str] = {
    "BRAIN_MODEL": "qwen2.5:14b",
    "BOUNCER_MODEL": "qwen2.5:14b",
    "TAGGER_MODEL": "hf.co/bartowski/Llama-3.2-3B-Instruct-GGUF:Q8_0",
    "SUMMARIZER_MODEL": "hf.co/bartowski/Llama-3.2-3B-Instruct-GGUF:Q8_0",
    "VISION_MODEL": "",
    "EMBED_MODEL": "mxbai-embed-large",
    "PREWARM_MODEL_NAME": "",
}


@dataclass(slots=True)
class CheckResult:
    name: str
    severity: str
    ok: bool
    summary: str
    detail: str | None = None


@dataclass(slots=True)
class HealthReport:
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def hard_failures(self) -> list[CheckResult]:
        return [check for check in self.checks if check.severity == "hard" and not check.ok]

    @property
    def soft_failures(self) -> list[CheckResult]:
        return [check for check in self.checks if check.severity == "soft" and not check.ok]

    @property
    def ok_checks(self) -> list[CheckResult]:
        return [check for check in self.checks if check.ok]

    def exit_code(self) -> int:
        return 1 if self.hard_failures else 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "checks": [asdict(check) for check in self.checks],
            "hard_failures": len(self.hard_failures),
            "soft_failures": len(self.soft_failures),
        }


def _resolve_db_dir(*, test_mode: bool) -> Path:
    if test_mode:
        return Path(os.getenv("TEST_DB_DIR", "data/test/"))
    return Path(os.getenv("DB_DIR", "data/prod/"))


def _logs_dir(db_dir: Path) -> Path:
    return db_dir / "logs"


def _recall_db_path(db_dir: Path) -> Path:
    return db_dir / os.getenv("RECALL_DB_NAME", "recall.db")


def _configured_model_names() -> list[str]:
    models: list[str] = []
    for env_name, default in _MODEL_DEFAULTS.items():
        value = os.getenv(env_name, default).strip()
        if env_name == "VISION_MODEL" and not value:
            value = os.getenv("BRAIN_MODEL", _MODEL_DEFAULTS["BRAIN_MODEL"]).strip()
        if not value:
            continue
        if value not in models:
            models.append(value)
    return models


def _ollama_name_variants(model_name: str) -> set[str]:
    """Return acceptable Ollama name variants for one configured model.

    Ollama often reports pulled models with an explicit ``:latest`` suffix
    even when the configured tag omits it. Treat those as equivalent.
    """
    normalized = model_name.strip()
    if not normalized:
        return set()
    if ":" not in normalized.rsplit("/", 1)[-1]:
        return {normalized, f"{normalized}:latest"}
    if normalized.endswith(":latest"):
        return {normalized, normalized[:-7]}
    return {normalized}


def _check_env_cast(
    env_name: str,
    cast: Callable[[str], Any],
    *,
    severity: str = "hard",
) -> CheckResult:
    raw = os.getenv(env_name)
    if raw in (None, ""):
        return CheckResult(
            name=env_name,
            severity=severity,
            ok=True,
            summary=f"{env_name} not set; code defaults will apply",
        )
    try:
        cast(raw)
        return CheckResult(
            name=env_name,
            severity=severity,
            ok=True,
            summary=f"{env_name} parsed successfully",
        )
    except Exception as exc:
        return CheckResult(
            name=env_name,
            severity=severity,
            ok=False,
            summary=f"{env_name} is invalid",
            detail=f"value={raw!r} error={exc}",
        )


def _check_writable_dir(path: Path, *, name: str, severity: str = "hard") -> CheckResult:
    try:
        path.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path, prefix=".sandy-health-", delete=True):
            pass
        return CheckResult(
            name=name,
            severity=severity,
            ok=True,
            summary=f"{path} is writable",
        )
    except Exception as exc:
        return CheckResult(
            name=name,
            severity=severity,
            ok=False,
            summary=f"{path} is not writable",
            detail=str(exc),
        )


def validate_startup_config() -> list[CheckResult]:
    results: list[CheckResult] = []

    token = (os.getenv("DISCORD_API_KEY") or "").strip()
    if token:
        results.append(CheckResult(
            name="DISCORD_API_KEY",
            severity="hard",
            ok=True,
            summary="Discord API token is present",
        ))
    else:
        results.append(CheckResult(
            name="DISCORD_API_KEY",
            severity="hard",
            ok=False,
            summary="Discord API token is missing",
            detail="Set DISCORD_API_KEY in .env or the environment before startup.",
        ))

    for env_name in _INT_ENV_VARS:
        results.append(_check_env_cast(env_name, int))
    for env_name in _FLOAT_ENV_VARS:
        results.append(_check_env_cast(env_name, float))

    return results


def validate_local_state(*, test_mode: bool) -> list[CheckResult]:
    results: list[CheckResult] = []
    db_dir = _resolve_db_dir(test_mode=test_mode)
    logs_dir = _logs_dir(db_dir)

    results.append(_check_writable_dir(db_dir, name="DB_DIR"))
    results.append(_check_writable_dir(logs_dir, name="LOGS_DIR"))

    try:
        db = ChatDatabase(str(_recall_db_path(db_dir)))
        db.init_db()
        results.append(CheckResult(
            name="recall_db",
            severity="hard",
            ok=True,
            summary="Recall database initialized successfully",
        ))
    except Exception as exc:
        results.append(CheckResult(
            name="recall_db",
            severity="hard",
            ok=False,
            summary="Recall database initialization failed",
            detail=str(exc),
        ))

    try:
        Registry()
        results.append(CheckResult(
            name="registry_db",
            severity="hard",
            ok=True,
            summary="Registry database initialized successfully",
        ))
    except Exception as exc:
        results.append(CheckResult(
            name="registry_db",
            severity="hard",
            ok=False,
            summary="Registry database initialization failed",
            detail=str(exc),
        ))

    return results


async def check_ollama() -> list[CheckResult]:
    client = ollama.AsyncClient()
    try:
        response = await client.list()
    except Exception as exc:
        return [
            CheckResult(
                name="ollama",
                severity="soft",
                ok=False,
                summary="Ollama is unreachable",
                detail=str(exc),
            )
        ]

    available_models: set[str] = set()
    raw_models = getattr(response, "models", None)
    if raw_models is None and isinstance(response, dict):
        raw_models = response.get("models", [])
    for item in raw_models or []:
        if isinstance(item, dict):
            model_name = item.get("model") or item.get("name")
        else:
            model_name = getattr(item, "model", None) or getattr(item, "name", None)
        if model_name:
            available_models.add(model_name)

    configured = _configured_model_names()
    missing = [
        model for model in configured
        if not (_ollama_name_variants(model) & available_models)
    ]

    results = [
        CheckResult(
            name="ollama",
            severity="soft",
            ok=True,
            summary=f"Ollama reachable; {len(available_models)} model(s) available",
        )
    ]
    if missing:
        results.append(CheckResult(
            name="ollama_models",
            severity="soft",
            ok=False,
            summary="Some configured Ollama models are missing",
            detail=", ".join(missing),
        ))
    else:
        results.append(CheckResult(
            name="ollama_models",
            severity="soft",
            ok=True,
            summary="All configured Ollama models are present",
        ))
    return results


async def check_searxng() -> CheckResult:
    host = os.getenv("SEARXNG_HOST", "127.0.0.1")
    port = os.getenv("SEARXNG_PORT", "8888")
    base_url = f"http://{host}:{port}"
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(
                f"{base_url}/search",
                params={"q": "test", "format": "json"},
                headers={"Accept": "application/json"},
            )
            response.raise_for_status()
            response.json()
        return CheckResult(
            name="searxng",
            severity="soft",
            ok=True,
            summary=f"SearXNG reachable at {base_url}",
        )
    except Exception as exc:
        return CheckResult(
            name="searxng",
            severity="soft",
            ok=False,
            summary=f"SearXNG is unreachable at {base_url}",
            detail=str(exc),
        )


async def check_vector_memory(*, test_mode: bool) -> CheckResult:
    original_db_dir = os.getenv("DB_DIR")
    try:
        os.environ["DB_DIR"] = str(_resolve_db_dir(test_mode=test_mode))
        VectorMemory()
        return CheckResult(
            name="vector_memory",
            severity="soft",
            ok=True,
            summary="Vector memory opened successfully",
        )
    except Exception as exc:
        return CheckResult(
            name="vector_memory",
            severity="soft",
            ok=False,
            summary="Vector memory initialization failed",
            detail=str(exc),
        )
    finally:
        if original_db_dir is None:
            os.environ.pop("DB_DIR", None)
        else:
            os.environ["DB_DIR"] = original_db_dir


async def collect_health_report(*, test_mode: bool) -> HealthReport:
    checks = [
        *validate_startup_config(),
        *validate_local_state(test_mode=test_mode),
    ]
    checks.extend(await check_ollama())
    checks.append(await check_vector_memory(test_mode=test_mode))
    checks.append(await check_searxng())
    return HealthReport(checks=checks)


def log_startup_report(report: HealthReport, startup_logger: logging.Logger) -> None:
    for check in report.hard_failures:
        startup_logger.error("Startup check failed [%s]: %s", check.name, check.summary)
        if check.detail:
            startup_logger.error("  %s", check.detail)

    for check in report.soft_failures:
        startup_logger.warning("Dependency check warning [%s]: %s", check.name, check.summary)
        if check.detail:
            startup_logger.warning("  %s", check.detail)

    if not report.hard_failures and not report.soft_failures:
        startup_logger.info("Startup and dependency checks passed cleanly")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sandy health and preflight checks")
    parser.add_argument("--test", action="store_true", help="Use TEST_DB_DIR instead of DB_DIR")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of human-readable output")
    return parser


def _print_human_report(report: HealthReport) -> None:
    severity_order = {"hard": 0, "soft": 1}
    for check in sorted(report.checks, key=lambda item: (severity_order[item.severity], item.name)):
        status = "OK" if check.ok else "FAIL"
        print(f"[{status}] {check.severity.upper():4} {check.name}: {check.summary}")
        if check.detail:
            print(f"  {check.detail}")
    print(
        f"\nSummary: {len(report.ok_checks)} ok, "
        f"{len(report.soft_failures)} soft failure(s), "
        f"{len(report.hard_failures)} hard failure(s)"
    )


async def _async_main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    report = await collect_health_report(test_mode=args.test)
    if args.json:
        print(json.dumps(report.to_dict(), ensure_ascii=True, sort_keys=True))
    else:
        _print_human_report(report)
    return report.exit_code()


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(_async_main(argv))


if __name__ == "__main__":
    raise SystemExit(main())
