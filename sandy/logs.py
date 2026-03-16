"""Utilities for inspecting Sandy's structured logs and trace data.

Usage examples:
    python -m sandy.logs recent
    python -m sandy.logs --test recent
    python -m sandy.logs show 1234567890
    python -m sandy.logs find --text whale
    python -m sandy.logs failures
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from textwrap import shorten
from typing import Any

from dotenv import load_dotenv

from .paths import resolve_db_dir

load_dotenv()


@dataclass(slots=True)
class LogPaths:
    db_dir: Path
    logs_dir: Path
    jsonl_path: Path
    trace_db_path: Path


def _resolve_paths(*, test_mode: bool) -> LogPaths:
    db_dir = resolve_db_dir(test_mode=test_mode)
    logs_dir = db_dir / "logs"
    return LogPaths(
        db_dir=db_dir,
        logs_dir=logs_dir,
        jsonl_path=logs_dir / "sandy.jsonl",
        trace_db_path=logs_dir / "trace_events.db",
    )


def _connect_trace_db(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def _load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def _index_records_by_trace(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    by_trace: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        trace_id: str | None = None
        if record.get("record_type") == "forensic":
            trace_id = record.get("forensic", {}).get("trace_id")
        elif record.get("record_type") == "trace":
            trace_id = record.get("event", {}).get("trace_id")
        if trace_id:
            by_trace[trace_id].append(record)
    return by_trace


def _forensic_map(records: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    result: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in records:
        if record.get("record_type") != "forensic":
            continue
        payload = record.get("forensic", {})
        trace_id = payload.get("trace_id")
        artifact = payload.get("artifact")
        if trace_id and artifact:
            result[trace_id][artifact] = payload
    return result


def _trace_map(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record.get("record_type") != "trace":
            continue
        payload = record.get("event", {})
        trace_id = payload.get("trace_id")
        if trace_id:
            result[trace_id].append(payload)
    return result


def _summarize_recent_turns(
    conn: sqlite3.Connection,
    forensic_records: dict[str, dict[str, dict[str, Any]]],
    *,
    limit: int,
    human_only: bool = False,
) -> list[dict[str, Any]]:
    fetch_limit = limit if not human_only else max(limit * 4, limit + 10)
    turn_rows = conn.execute(
        """
        SELECT created_at, trace_id, payload_json
        FROM trace_events
        WHERE stage = 'turn_completed'
        ORDER BY created_at DESC
        LIMIT ?
        """,
        (fetch_limit,),
    ).fetchall()
    results: list[dict[str, Any]] = []
    for row in turn_rows:
        trace_id = row["trace_id"]
        turn_payload = json.loads(row["payload_json"])
        message_row = conn.execute(
            """
            SELECT payload_json
            FROM trace_events
            WHERE trace_id = ? AND stage = 'message_received'
            ORDER BY created_at ASC, id ASC
            LIMIT 1
            """,
            (trace_id,),
        ).fetchone()
        message_payload = json.loads(message_row["payload_json"]) if message_row else {}
        author_is_bot = bool(message_payload.get("author_is_bot"))
        if human_only and author_is_bot:
            continue
        forensic = forensic_records.get(trace_id, {})
        turn_input = forensic.get("turn_input", {})
        bouncer = forensic.get("bouncer_decision", {})
        tool_call = forensic.get("tool_call", {})
        results.append(
            {
                "created_at": row["created_at"],
                "trace_id": trace_id,
                "author_name": turn_input.get("author_name", "?"),
                "channel_name": turn_input.get("channel_name", "?"),
                "guild_name": turn_input.get("guild_name", "?"),
                "content": turn_input.get("resolved_content") or turn_input.get("raw_content") or "",
                "replied": turn_payload.get("replied"),
                "tool_name": tool_call.get("tool_name") or bouncer.get("parsed_result", {}).get("recommended_tool"),
                "duration_ms": turn_payload.get("duration_ms"),
                "author_is_bot": author_is_bot,
            }
        )
        if len(results) >= limit:
            break
    return results


def get_recent_turns(*, test_mode: bool, limit: int = 10, human_only: bool = False) -> list[dict[str, Any]]:
    paths = _resolve_paths(test_mode=test_mode)
    if not paths.trace_db_path.exists() or not paths.jsonl_path.exists():
        return []
    records = _load_jsonl_records(paths.jsonl_path)
    forensic_records = _forensic_map(records)
    with _connect_trace_db(paths.trace_db_path) as conn:
        return _summarize_recent_turns(
            conn,
            forensic_records,
            limit=limit,
            human_only=human_only,
        )


def get_trace_detail(*, test_mode: bool, trace_id: str) -> dict[str, Any] | None:
    paths = _resolve_paths(test_mode=test_mode)
    if not paths.trace_db_path.exists() or not paths.jsonl_path.exists():
        return None

    records = _load_jsonl_records(paths.jsonl_path)
    records_by_trace = _index_records_by_trace(records)
    if trace_id not in records_by_trace:
        return None

    with _connect_trace_db(paths.trace_db_path) as conn:
        trace_events = [
            json.loads(row["payload_json"])
            for row in conn.execute(
                """
                SELECT payload_json
                FROM trace_events
                WHERE trace_id = ?
                ORDER BY created_at ASC, id ASC
                """,
                (trace_id,),
            ).fetchall()
        ]
        forensic: dict[str, dict[str, Any]] = {}
        for record in records_by_trace.get(trace_id, []):
            if record.get("record_type") == "forensic":
                payload = record["forensic"]
                forensic[payload["artifact"]] = payload

        turn_input = forensic.get("turn_input", {})
        turn_row = conn.execute(
            """
            SELECT payload_json
            FROM trace_events
            WHERE trace_id = ? AND stage = 'turn_completed'
            ORDER BY created_at DESC, id DESC
            LIMIT 1
            """,
            (trace_id,),
        ).fetchone()
        turn_payload = json.loads(turn_row["payload_json"]) if turn_row else {}

    return {
        "trace_id": trace_id,
        "turn_input": turn_input,
        "timeline": trace_events,
        "turn_completed": turn_payload,
        "artifacts": {
            "bouncer_decision": forensic.get("bouncer_decision", {}),
            "bouncer_context": forensic.get("bouncer_context", {}),
            "vision_artifacts": forensic.get("vision_artifacts", {}),
            "tool_call": forensic.get("tool_call", {}),
            "retrieval": forensic.get("retrieval", {}),
            "brain_generation": forensic.get("brain_generation", {}),
            "reply_output": forensic.get("reply_output", {}),
            "reply_delivery": forensic.get("reply_delivery", {}),
        },
    }


def _print_recent(turns: list[dict[str, Any]]) -> None:
    if not turns:
        print("No recent turns found.")
        return
    for turn in turns:
        content = shorten(turn["content"].replace("\n", " "), width=90, placeholder="...")
        print(
            f"{turn['created_at']} | {turn['trace_id']} | "
            f"{turn['guild_name']}/#{turn['channel_name']} | {turn['author_name']} | "
            f"replied={turn['replied']} tool={turn['tool_name'] or '-'} "
            f"duration={turn['duration_ms']}ms"
        )
        print(f"  {content}")


def _find_reply_source_traces(
    content: str,
    *,
    exclude_trace_id: str,
    records_by_trace: dict[str, list[dict[str, Any]]],
) -> list[str]:
    candidates: list[str] = []
    for trace_id, records in records_by_trace.items():
        if trace_id == exclude_trace_id:
            continue
        for record in records:
            if record.get("record_type") != "forensic":
                continue
            payload = record["forensic"]
            if payload.get("artifact") == "reply_output" and payload.get("finalized_reply") == content:
                candidates.append(trace_id)
                break
    return candidates


def _print_show(trace_id: str, conn: sqlite3.Connection, records_by_trace: dict[str, list[dict[str, Any]]]) -> int:
    records = records_by_trace.get(trace_id, [])
    if not records:
        print(f"No records found for trace_id={trace_id}")
        return 1

    forensic: dict[str, dict[str, Any]] = {}
    trace_events = [
        json.loads(row["payload_json"])
        for row in conn.execute(
            """
            SELECT payload_json
            FROM trace_events
            WHERE trace_id = ?
            ORDER BY created_at ASC, id ASC
            """,
            (trace_id,),
        ).fetchall()
    ]
    for record in records:
        if record.get("record_type") == "forensic":
            payload = record["forensic"]
            forensic[payload["artifact"]] = payload

    turn_input = forensic.get("turn_input", {})
    bouncer = forensic.get("bouncer_decision", {})
    retrieval = forensic.get("retrieval", {})
    tool_call = forensic.get("tool_call", {})
    brain = forensic.get("brain_generation", {})
    reply = forensic.get("reply_output", {})
    delivery = forensic.get("reply_delivery", {})

    print(f"Trace ID: {trace_id}")
    content = "(empty)"
    if turn_input:
        print(
            "Message: "
            f"{turn_input.get('guild_name', '?')}/#{turn_input.get('channel_name', '?')} "
            f"<{turn_input.get('author_name', '?')}>"
        )
        content = turn_input.get("resolved_content") or turn_input.get("raw_content") or "(empty)"
        print(f"Content: {content}")

    turn_row = conn.execute(
        """
        SELECT payload_json
        FROM trace_events
        WHERE trace_id = ? AND stage = 'turn_completed'
        ORDER BY created_at DESC, id DESC
        LIMIT 1
        """,
        (trace_id,),
    ).fetchone()
    turn_payload = json.loads(turn_row["payload_json"]) if turn_row else {}
    if turn_payload.get("bot_message"):
        print("\nNote:")
        print("  This trace is for a bot-authored Discord message after it was already sent.")
        print("  It only covers post-send ingestion and memory enqueue, not the earlier generation pipeline.")
        source_candidates = _find_reply_source_traces(
            content,
            exclude_trace_id=trace_id,
            records_by_trace=records_by_trace,
        )
        if source_candidates:
            print(f"  Likely source trace(s): {', '.join(source_candidates[:3])}")

    if trace_events:
        print("\nTimeline:")
        for event in trace_events:
            bits = [event["stage"], f"status={event.get('status', 'ok')}"]
            if event.get("duration_ms") is not None:
                bits.append(f"duration={event['duration_ms']}ms")
            if event.get("tool_name"):
                bits.append(f"tool={event['tool_name']}")
            if event.get("reply_chars") is not None:
                bits.append(f"reply_chars={event['reply_chars']}")
            print("  - " + " ".join(bits))

    parsed_bouncer = bouncer.get("parsed_result", {})
    if bouncer:
        print("\nBouncer:")
        print(f"  decision: respond={parsed_bouncer.get('should_respond')} use_tool={parsed_bouncer.get('use_tool')}")
        print(f"  reason: {parsed_bouncer.get('reason')}")
        if bouncer.get("prompt_user"):
            print(f"  prompt_user: {shorten(bouncer['prompt_user'].replace(chr(10), ' '), width=160, placeholder='...')}")

    if tool_call:
        print("\nTool:")
        print(f"  name: {tool_call.get('tool_name')}")
        print(f"  args: {tool_call.get('arguments')}")
        print(f"  result: {shorten((tool_call.get('result') or '').replace(chr(10), ' '), width=220, placeholder='...')}")

    if retrieval:
        print("\nRetrieval:")
        print(f"  query: {retrieval.get('query_text')}")
        print(f"  rag_context: {shorten((retrieval.get('rag_context') or '').replace(chr(10), ' '), width=260, placeholder='...')}")

    if brain:
        print("\nBrain:")
        print(f"  model: {brain.get('model')}")
        print(f"  done_reason: {brain.get('done_reason')}")
        print(f"  prompt_user: {shorten((brain.get('prompt_user') or '').replace(chr(10), ' '), width=180, placeholder='...')}")
        print(f"  tool_context: {shorten((brain.get('tool_context') or '').replace(chr(10), ' '), width=220, placeholder='...')}")

    if reply:
        print("\nReply:")
        print(f"  finalized: {reply.get('finalized_reply') or '(none)'}")
    if delivery:
        print(f"  parts: {delivery.get('message_parts')}")
    return 0


def _matches_record(
    record: dict[str, Any],
    *,
    text: str | None,
    author: str | None,
) -> bool:
    haystack = json.dumps(record, ensure_ascii=False).lower()
    if text and text.lower() not in haystack:
        return False
    if author:
        author_lower = author.lower()
        forensic = record.get("forensic", {})
        if forensic.get("author_name", "").lower() != author_lower:
            return False
    return True


def _find_matches(
    records: list[dict[str, Any]],
    *,
    text: str | None,
    author: str | None,
    limit: int,
) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for record in reversed(records):
        if record.get("record_type") not in {"forensic", "trace", "log"}:
            continue
        if not _matches_record(record, text=text, author=author):
            continue
        trace_id = (
            record.get("forensic", {}).get("trace_id")
            or record.get("event", {}).get("trace_id")
            or "-"
        )
        label = record.get("forensic", {}).get("artifact") or record.get("event", {}).get("stage") or record.get("logger")
        key = (trace_id, str(label))
        if key in seen:
            continue
        seen.add(key)
        matches.append(record)
        if len(matches) >= limit:
            break
    return matches


def _print_find(matches: list[dict[str, Any]]) -> None:
    if not matches:
        print("No matching log records found.")
        return
    for record in matches:
        timestamp = record.get("timestamp", "?")
        if record.get("record_type") == "forensic":
            payload = record["forensic"]
            label = payload.get("artifact")
            trace_id = payload.get("trace_id")
            snippet = shorten(json.dumps(payload, ensure_ascii=False), width=220, placeholder="...")
        elif record.get("record_type") == "trace":
            payload = record["event"]
            label = payload.get("stage")
            trace_id = payload.get("trace_id")
            snippet = shorten(json.dumps(payload, ensure_ascii=False), width=220, placeholder="...")
        else:
            label = record.get("logger")
            trace_id = "-"
            snippet = shorten(record.get("message", ""), width=220, placeholder="...")
        print(f"{timestamp} | {trace_id} | {label} | {snippet}")


def _print_failures(conn: sqlite3.Connection, *, limit: int) -> None:
    rows = conn.execute(
        """
        SELECT created_at, trace_id, stage, status, payload_json
        FROM trace_events
        WHERE status != 'ok'
        ORDER BY created_at DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    if not rows:
        print("No failing trace stages found.")
        return
    for row in rows:
        payload = json.loads(row["payload_json"])
        print(
            f"{row['created_at']} | {row['trace_id']} | {row['stage']} | "
            f"status={row['status']} | payload={shorten(json.dumps(payload), width=220, placeholder='...')}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect Sandy trace and forensic logs",
        epilog=(
            "Examples:\n"
            "  python -m sandy.logs recent --limit 5\n"
            "  python -m sandy.logs recent --human-only\n"
            "  python -m sandy.logs show 1482258945799094444\n"
            "  python -m sandy.logs find --text \"color coding\"\n"
            "  python -m sandy.logs find --author alice --text whales\n"
            "  python -m sandy.logs --test failures\n\n"
            "Notes:\n"
            "  - A trace_id identifies one Discord message turn.\n"
            "  - Human-authored turns include the full bouncer/tool/RAG/brain pipeline.\n"
            "  - Bot-authored turns usually only show message ingestion and memory enqueue,\n"
            "    because the generation happened in the earlier human-authored trace."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Use TEST_DB_DIR instead of DB_DIR",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    recent = subparsers.add_parser(
        "recent",
        help="Show recent message turns",
        description="List recent turns with author, channel, reply/tool summary, and duration.",
    )
    recent.add_argument("--limit", type=int, default=10)
    recent.add_argument(
        "--human-only",
        action="store_true",
        help="Hide bot-authored reply turns and show only human-originated traces",
    )

    show = subparsers.add_parser(
        "show",
        help="Show one trace end-to-end",
        description=(
            "Reconstruct one trace from trace_events.db and sandy.jsonl.\n"
            "For human-authored turns this usually includes bouncer, tool, retrieval,\n"
            "brain, and reply artifacts. Bot-authored turns are typically much smaller."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    show.add_argument("trace_id")

    find = subparsers.add_parser(
        "find",
        help="Search forensic/log records",
        description="Search JSONL records by text and/or exact forensic author name.",
    )
    find.add_argument("--text", help="Case-insensitive text to search for")
    find.add_argument("--author", help="Exact forensic author name to filter on")
    find.add_argument("--limit", type=int, default=20)

    failures = subparsers.add_parser(
        "failures",
        help="List failing trace stages",
        description="Show trace stages whose status was not 'ok'.",
    )
    failures.add_argument("--limit", type=int, default=20)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    paths = _resolve_paths(test_mode=args.test)

    if not paths.trace_db_path.exists():
        print(f"Trace DB not found: {paths.trace_db_path}", file=sys.stderr)
        return 1
    if not paths.jsonl_path.exists():
        print(f"JSONL log not found: {paths.jsonl_path}", file=sys.stderr)
        return 1

    records = _load_jsonl_records(paths.jsonl_path)
    records_by_trace = _index_records_by_trace(records)
    forensic_records = _forensic_map(records)

    with _connect_trace_db(paths.trace_db_path) as conn:
        if args.command == "recent":
            _print_recent(
                _summarize_recent_turns(
                    conn,
                    forensic_records,
                    limit=args.limit,
                    human_only=args.human_only,
                )
            )
            return 0
        if args.command == "show":
            return _print_show(args.trace_id, conn, records_by_trace)
        if args.command == "find":
            if not args.text and not args.author:
                parser.error("find requires at least --text or --author")
            _print_find(_find_matches(records, text=args.text, author=args.author, limit=args.limit))
            return 0
        if args.command == "failures":
            _print_failures(conn, limit=args.limit)
            return 0

    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
