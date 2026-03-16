from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import sandy.api as api_module
from sandy.api import ApiService
from sandy.runtime_state import RuntimeState
from sandy.trace import TurnTrace


def _make_trace_db(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE trace_events (
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
    return conn


def _seed_logs(base: Path) -> None:
    logs_dir = base / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    conn = _make_trace_db(logs_dir / "trace_events.db")
    conn.execute(
        """
        INSERT INTO trace_events (created_at, trace_id, stage, status, message_id, guild_id, channel_id, author_id, payload_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "2026-03-14T06:00:00+00:00",
            "123",
            "message_received",
            "ok",
            123,
            1,
            2,
            3,
            json.dumps({"trace_id": "123", "stage": "message_received", "author_is_bot": False}),
        ),
    )
    conn.execute(
        """
        INSERT INTO trace_events (created_at, trace_id, stage, status, message_id, guild_id, channel_id, author_id, payload_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "2026-03-14T06:00:03+00:00",
            "123",
            "turn_completed",
            "ok",
            123,
            1,
            2,
            3,
            json.dumps({"trace_id": "123", "stage": "turn_completed", "replied": True, "duration_ms": 321}),
        ),
    )
    conn.commit()
    conn.close()

    records = [
        {
            "record_type": "forensic",
            "timestamp": "2026-03-14T06:00:00+00:00",
            "forensic": {
                "trace_id": "123",
                "artifact": "turn_input",
                "author_name": "alice",
                "channel_name": "general",
                "guild_name": "Guild",
                "resolved_content": "hello there",
            },
        },
        {
            "record_type": "forensic",
            "timestamp": "2026-03-14T06:00:01+00:00",
            "forensic": {
                "trace_id": "123",
                "artifact": "bouncer_decision",
                "parsed_result": {
                    "should_respond": True,
                    "use_tool": False,
                    "recommended_tool": None,
                },
            },
        },
    ]
    with (logs_dir / "sandy.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def _trace() -> TurnTrace:
    return TurnTrace(
        trace_id="123",
        message_id=123,
        guild_id=1,
        guild_name="Guild",
        channel_id=2,
        channel_name="general",
        author_id=3,
        author_name="alice",
        started_at=1.0,
    )


def test_api_service_status_and_gpu_payload(monkeypatch) -> None:
    state = RuntimeState()
    state.set_discord_connected(True, user_name="Sandy")
    state.begin_turn(_trace(), author_is_bot=False)
    state.update_turn_stage(_trace(), "bouncer")
    state.set_last_bouncer_decision(
        trace_id="123",
        should_respond=True,
        use_tool=False,
        tool_name=None,
    )
    pipeline = SimpleNamespace(llm=SimpleNamespace(is_busy=lambda: True))
    service = ApiService(pipeline=pipeline, runtime_state=state, test_mode=False)

    status = service.status_payload()

    assert status["discord"]["connected"] is True
    assert status["llm"]["busy"] is True
    assert status["current_turn"]["stage"] == "bouncer"
    assert status["last_bouncer_decision"]["should_respond"] is True

    fake_pynvml = SimpleNamespace(
        NVML_TEMPERATURE_GPU=0,
        NVML_VALUE_NOT_AVAILABLE=-1,
        nvmlInit=lambda: None,
        nvmlShutdown=lambda: None,
        nvmlDeviceGetCount=lambda: 1,
        nvmlDeviceGetHandleByIndex=lambda index: f"gpu-{index}",
        nvmlDeviceGetName=lambda handle: b"NVIDIA RTX 3090",
        nvmlDeviceGetUtilizationRates=lambda handle: SimpleNamespace(gpu=55),
        nvmlDeviceGetMemoryInfo=lambda handle: SimpleNamespace(used=12000 * 1024 * 1024, total=24576 * 1024 * 1024),
        nvmlDeviceGetPowerUsage=lambda handle: 280500,
        nvmlDeviceGetEnforcedPowerLimit=lambda handle: 350000,
        nvmlDeviceGetTemperature=lambda handle, sensor: 67,
    )
    with patch.object(api_module, "pynvml", fake_pynvml):
        gpu = service.gpu_payload()

    assert gpu["available"] is True
    assert gpu["backend"] == "nvml"
    assert gpu["devices"][0]["index"] == 0
    assert gpu["devices"][0]["memory_utilization_percent"] == 48.8


def test_api_service_gpu_payload_falls_back_to_nvidia_smi() -> None:
    state = RuntimeState()
    pipeline = SimpleNamespace(llm=SimpleNamespace(is_busy=lambda: False))
    service = ApiService(pipeline=pipeline, runtime_state=state, test_mode=False)

    completed = SimpleNamespace(
        stdout="0, NVIDIA RTX 3090, 55, 12000, 24576, 67, 280.5, 350.0\n",
    )
    with patch.object(api_module, "pynvml", None):
        with patch("sandy.api.shutil.which", return_value="/usr/bin/nvidia-smi"):
            with patch("sandy.api.subprocess.run", return_value=completed):
                gpu = service.gpu_payload()

    assert gpu["available"] is True
    assert gpu["backend"] == "nvidia-smi"
    assert gpu["devices"][0]["index"] == 0


def test_api_service_serves_recent_and_trace_detail(monkeypatch, tmp_path: Path) -> None:
    db_dir = tmp_path / "prod"
    _seed_logs(db_dir)
    monkeypatch.setenv("DB_DIR", str(db_dir))

    state = RuntimeState()
    pipeline = SimpleNamespace(llm=SimpleNamespace(is_busy=lambda: False))
    service = ApiService(pipeline=pipeline, runtime_state=state, test_mode=False)

    recent = service.recent_turns_payload(limit=5, human_only=False)
    assert recent["count"] == 1
    assert recent["turns"][0]["trace_id"] == "123"

    detail = service.trace_detail_payload("123")
    assert detail is not None
    assert detail["trace_id"] == "123"
    assert detail["turn_input"]["author_name"] == "alice"

    missing = service.trace_detail_payload("missing")
    assert missing is None
