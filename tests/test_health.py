from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from sandy import health


def test_validate_startup_config_flags_missing_token_and_bad_numeric(monkeypatch):
    monkeypatch.delenv("DISCORD_API_KEY", raising=False)
    monkeypatch.setenv("BRAIN_NUM_CTX", "not-an-int")
    monkeypatch.setenv("BRAIN_TEMPERATURE", "0.8")

    results = health.validate_startup_config()
    by_name = {result.name: result for result in results}

    assert by_name["DISCORD_API_KEY"].ok is False
    assert by_name["DISCORD_API_KEY"].severity == "hard"
    assert by_name["BRAIN_NUM_CTX"].ok is False
    assert by_name["BRAIN_TEMPERATURE"].ok is True


def test_validate_local_state_initializes_recall_and_registry(monkeypatch, tmp_path: Path):
    db_dir = tmp_path / "prod"
    monkeypatch.setenv("DB_DIR", str(db_dir))
    monkeypatch.setenv("SERVER_DB_NAME", "server.db")
    monkeypatch.setenv("RECALL_DB_NAME", "recall.db")

    results = health.validate_local_state(test_mode=False)
    by_name = {result.name: result for result in results}

    assert by_name["DB_DIR"].ok is True
    assert by_name["LOGS_DIR"].ok is True
    assert by_name["recall_db"].ok is True
    assert by_name["registry_db"].ok is True
    assert (db_dir / "recall.db").exists()
    assert (db_dir / "server.db").exists()


@pytest.mark.asyncio
async def test_check_ollama_reports_missing_models(monkeypatch):
    fake_response = {
        "models": [
            {"name": "brain-model"},
            {"name": "mxbai-embed-large"},
        ]
    }
    fake_client = SimpleNamespace(list=AsyncMock(return_value=fake_response))
    monkeypatch.setattr(health.ollama, "AsyncClient", lambda: fake_client)
    monkeypatch.setenv("BRAIN_MODEL", "brain-model")
    monkeypatch.setenv("BOUNCER_MODEL", "bouncer-model")
    monkeypatch.setenv("TAGGER_MODEL", "bouncer-model")
    monkeypatch.setenv("SUMMARIZER_MODEL", "bouncer-model")
    monkeypatch.setenv("VISION_MODEL", "vision-model")
    monkeypatch.setenv("EMBED_MODEL", "mxbai-embed-large")
    monkeypatch.setenv("PREWARM_MODEL_NAME", "")

    results = await health.check_ollama()
    by_name = {result.name: result for result in results}

    assert by_name["ollama"].ok is True
    assert by_name["ollama_models"].ok is False
    assert "bouncer-model" in (by_name["ollama_models"].detail or "")
    assert "vision-model" in (by_name["ollama_models"].detail or "")


@pytest.mark.asyncio
async def test_check_ollama_treats_bare_name_and_latest_tag_as_equivalent(monkeypatch):
    fake_response = {
        "models": [
            {"name": "brain-model:latest"},
            {"name": "mxbai-embed-large:latest"},
        ]
    }
    fake_client = SimpleNamespace(list=AsyncMock(return_value=fake_response))
    monkeypatch.setattr(health.ollama, "AsyncClient", lambda: fake_client)
    monkeypatch.setenv("BRAIN_MODEL", "brain-model")
    monkeypatch.setenv("BOUNCER_MODEL", "brain-model")
    monkeypatch.setenv("TAGGER_MODEL", "brain-model")
    monkeypatch.setenv("SUMMARIZER_MODEL", "brain-model")
    monkeypatch.setenv("VISION_MODEL", "brain-model")
    monkeypatch.setenv("EMBED_MODEL", "mxbai-embed-large")
    monkeypatch.setenv("PREWARM_MODEL_NAME", "")

    results = await health.check_ollama()
    by_name = {result.name: result for result in results}

    assert by_name["ollama"].ok is True
    assert by_name["ollama_models"].ok is True


@pytest.mark.asyncio
async def test_collect_health_report_combines_hard_and_soft_checks(monkeypatch):
    monkeypatch.setattr(
        health,
        "validate_startup_config",
        lambda: [health.CheckResult("cfg", "hard", True, "ok")],
    )
    monkeypatch.setattr(
        health,
        "validate_local_state",
        lambda test_mode: [health.CheckResult("state", "hard", False, "broken")],
    )
    monkeypatch.setattr(
        health,
        "check_ollama",
        AsyncMock(return_value=[health.CheckResult("ollama", "soft", False, "down")]),
    )
    monkeypatch.setattr(
        health,
        "check_vector_memory",
        AsyncMock(return_value=health.CheckResult("vector_memory", "soft", True, "ok")),
    )
    monkeypatch.setattr(
        health,
        "check_searxng",
        AsyncMock(return_value=health.CheckResult("searxng", "soft", False, "down")),
    )

    report = await health.collect_health_report(test_mode=False)

    assert [check.name for check in report.hard_failures] == ["state"]
    assert [check.name for check in report.soft_failures] == ["ollama", "searxng"]
    assert report.exit_code() == 1


def test_health_main_json_output(monkeypatch, capsys):
    async def fake_collect(*, test_mode: bool):
        assert test_mode is True
        return health.HealthReport(
            checks=[
                health.CheckResult("cfg", "hard", True, "ok"),
                health.CheckResult("ollama", "soft", False, "down"),
            ]
        )

    monkeypatch.setattr(health, "collect_health_report", fake_collect)

    exit_code = health.main(["--test", "--json"])
    output = capsys.readouterr().out.strip()
    payload = json.loads(output)

    assert exit_code == 0
    assert payload["hard_failures"] == 0
    assert payload["soft_failures"] == 1
