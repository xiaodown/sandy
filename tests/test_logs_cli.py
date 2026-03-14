from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from sandy.logs import (
    _find_matches,
    _forensic_map,
    _index_records_by_trace,
    _summarize_recent_turns,
    build_parser,
)


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


def test_summarize_recent_turns_uses_trace_and_forensic_data(tmp_path: Path) -> None:
    conn = _make_trace_db(tmp_path / "trace.db")
    conn.execute(
        """
        INSERT INTO trace_events (created_at, trace_id, stage, status, message_id, guild_id, channel_id, author_id, payload_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "2026-03-14T05:59:59+00:00",
            "123",
            "message_received",
            "ok",
            123,
            1,
            2,
            3,
            json.dumps({"trace_id": "123", "author_is_bot": False}),
        ),
    )
    conn.execute(
        """
        INSERT INTO trace_events (created_at, trace_id, stage, status, message_id, guild_id, channel_id, author_id, payload_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "2026-03-14T06:00:00+00:00",
            "123",
            "turn_completed",
            "ok",
            123,
            1,
            2,
            3,
            json.dumps({"trace_id": "123", "replied": True, "duration_ms": 456}),
        ),
    )
    forensic_records = {
        "123": {
            "turn_input": {
                "author_name": "alice",
                "channel_name": "general",
                "guild_name": "Test Guild",
                "resolved_content": "hello there",
            },
            "bouncer_decision": {
                "parsed_result": {"recommended_tool": None},
            },
        }
    }

    turns = _summarize_recent_turns(conn, forensic_records, limit=5, human_only=False)

    assert turns == [
        {
            "created_at": "2026-03-14T06:00:00+00:00",
            "trace_id": "123",
            "author_name": "alice",
            "channel_name": "general",
            "guild_name": "Test Guild",
            "content": "hello there",
            "replied": True,
            "tool_name": None,
            "duration_ms": 456,
            "author_is_bot": False,
        }
    ]


def test_find_matches_filters_forensic_records() -> None:
    records = [
        {
            "record_type": "forensic",
            "timestamp": "2026-03-14T06:00:00+00:00",
            "forensic": {
                "trace_id": "123",
                "artifact": "turn_input",
                "author_name": "alice",
                "raw_content": "whales are neat",
            },
        },
        {
            "record_type": "forensic",
            "timestamp": "2026-03-14T06:00:01+00:00",
            "forensic": {
                "trace_id": "124",
                "artifact": "turn_input",
                "author_name": "Other",
                "raw_content": "cats are neat",
            },
        },
    ]

    matches = _find_matches(records, text="whales", author="alice", limit=10)

    assert len(matches) == 1
    assert matches[0]["forensic"]["trace_id"] == "123"


def test_indexers_group_trace_and_forensic_records() -> None:
    records = [
        {"record_type": "trace", "event": {"trace_id": "123", "stage": "message_received"}},
        {"record_type": "forensic", "forensic": {"trace_id": "123", "artifact": "turn_input"}},
        {"record_type": "forensic", "forensic": {"trace_id": "123", "artifact": "reply_output"}},
    ]

    by_trace = _index_records_by_trace(records)
    forensic = _forensic_map(records)

    assert len(by_trace["123"]) == 3
    assert set(forensic["123"]) == {"turn_input", "reply_output"}


def test_build_parser_accepts_show_command() -> None:
    parser = build_parser()
    args = parser.parse_args(["--test", "show", "123"])

    assert args.test is True
    assert args.command == "show"
    assert args.trace_id == "123"


def test_summarize_recent_turns_can_filter_bot_messages(tmp_path: Path) -> None:
    conn = _make_trace_db(tmp_path / "trace.db")
    rows = [
        ("bot", True, "Sandy-test", "bot message"),
        ("human", False, "alice", "human message"),
    ]
    for idx, (trace_id, author_is_bot, _author_name, _content) in enumerate(rows, start=1):
        conn.execute(
            """
            INSERT INTO trace_events (created_at, trace_id, stage, status, message_id, guild_id, channel_id, author_id, payload_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                f"2026-03-14T06:00:0{idx}+00:00",
                trace_id,
                "message_received",
                "ok",
                idx,
                1,
                2,
                3,
                json.dumps({"trace_id": trace_id, "author_is_bot": author_is_bot}),
            ),
        )
        conn.execute(
            """
            INSERT INTO trace_events (created_at, trace_id, stage, status, message_id, guild_id, channel_id, author_id, payload_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                f"2026-03-14T06:00:1{idx}+00:00",
                trace_id,
                "turn_completed",
                "ok",
                idx,
                1,
                2,
                3,
                json.dumps({"trace_id": trace_id, "replied": False, "duration_ms": idx}),
            ),
        )
    forensic_records = {
        "bot": {"turn_input": {"author_name": "Sandy-test", "channel_name": "general", "guild_name": "Guild", "resolved_content": "bot message"}},
        "human": {"turn_input": {"author_name": "alice", "channel_name": "general", "guild_name": "Guild", "resolved_content": "human message"}},
    }

    turns = _summarize_recent_turns(conn, forensic_records, limit=10, human_only=True)

    assert len(turns) == 1
    assert turns[0]["trace_id"] == "human"


def test_summarize_recent_turns_human_only_backfills_past_bot_rows(tmp_path: Path) -> None:
    conn = _make_trace_db(tmp_path / "trace.db")
    rows = [
        ("bot-newest", True),
        ("bot-mid", True),
        ("human-a", False),
        ("human-b", False),
    ]
    forensic_records = {}
    for idx, (trace_id, author_is_bot) in enumerate(rows, start=1):
        conn.execute(
            """
            INSERT INTO trace_events (created_at, trace_id, stage, status, message_id, guild_id, channel_id, author_id, payload_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                f"2026-03-14T06:00:0{idx}+00:00",
                trace_id,
                "message_received",
                "ok",
                idx,
                1,
                2,
                3,
                json.dumps({"trace_id": trace_id, "author_is_bot": author_is_bot}),
            ),
        )
        conn.execute(
            """
            INSERT INTO trace_events (created_at, trace_id, stage, status, message_id, guild_id, channel_id, author_id, payload_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                f"2026-03-14T06:00:1{idx}+00:00",
                trace_id,
                "turn_completed",
                "ok",
                idx,
                1,
                2,
                3,
                json.dumps({"trace_id": trace_id, "replied": False, "duration_ms": idx}),
            ),
        )
        forensic_records[trace_id] = {
            "turn_input": {
                "author_name": "Sandy-test" if author_is_bot else f"alice-{idx}",
                "channel_name": "general",
                "guild_name": "Guild",
                "resolved_content": trace_id,
            }
        }

    turns = _summarize_recent_turns(conn, forensic_records, limit=2, human_only=True)

    assert [turn["trace_id"] for turn in turns] == ["human-b", "human-a"]
