"""Maintenance utilities for Recall and vector memory.

Examples:
    python -m sandy.maintenance recall-find --query "steam" --test
    python -m sandy.maintenance delete-vector --discord-message-id 1482282320600891422 --test
    python -m sandy.maintenance purge-vector-from-recall --query "Vault of the Vanquished" --test --yes
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from textwrap import shorten

from dotenv import load_dotenv

from .paths import resolve_db_dir
from .recall import ChatDatabase
from .vector_memory import VectorMemory

load_dotenv()


def _build_recall_db(*, test_mode: bool) -> ChatDatabase:
    db_dir = resolve_db_dir(test_mode=test_mode)
    db = ChatDatabase(str(db_dir / os.getenv("RECALL_DB_NAME", "recall.db")))
    db.init_db()
    return db


def _build_vector_memory(*, test_mode: bool) -> VectorMemory:
    db_dir = resolve_db_dir(test_mode=test_mode)
    os.environ["DB_DIR"] = str(db_dir)
    return VectorMemory()


def _print_recall_rows(rows) -> None:
    if not rows:
        print("No Recall messages found.")
        return
    for row in rows:
        snippet = shorten(row.content.replace("\n", " "), width=140, placeholder="...")
        print(
            f"recall_id={row.id} discord_message_id={row.discord_message_id} "
            f"{row.timestamp.isoformat()} {row.server_name}/#{row.channel_name} <{row.author_name}>"
        )
        print(f"  {snippet}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Maintenance utilities for Recall/Chroma cleanup",
    )
    parser.add_argument("--test", action="store_true", help="Use TEST_DB_DIR instead of DB_DIR")
    subparsers = parser.add_subparsers(dest="command", required=True)

    recall_find = subparsers.add_parser("recall-find", help="Search Recall via FTS and show Discord message ids")
    recall_find.add_argument("--query", required=True, help="FTS query for Recall")
    recall_find.add_argument("--limit", type=int, default=20)

    delete_vector = subparsers.add_parser("delete-vector", help="Delete one Chroma document by Discord message id")
    delete_vector.add_argument("--discord-message-id", type=int, required=True)

    purge = subparsers.add_parser(
        "purge-vector-from-recall",
        help="Search Recall, then delete matching Chroma docs using stored Discord message ids",
    )
    purge.add_argument("--query", required=True, help="FTS query for Recall")
    purge.add_argument("--limit", type=int, default=20)
    purge.add_argument("--yes", action="store_true", help="Actually perform deletions")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "recall-find":
        db = _build_recall_db(test_mode=args.test)
        rows = db.get_messages(q=args.query, limit=args.limit)
        _print_recall_rows(rows)
        return 0

    if args.command == "delete-vector":
        vector_memory = _build_vector_memory(test_mode=args.test)
        deleted = vector_memory.delete_message(str(args.discord_message_id))
        if deleted:
            print(f"Deleted vector document for Discord message {args.discord_message_id}.")
            return 0
        print(f"No vector document found for Discord message {args.discord_message_id}.", file=sys.stderr)
        return 1

    if args.command == "purge-vector-from-recall":
        db = _build_recall_db(test_mode=args.test)
        rows = db.get_messages(q=args.query, limit=args.limit)
        if not rows:
            print("No Recall messages found.")
            return 0
        _print_recall_rows(rows)
        if not args.yes:
            print("\nDry run only. Re-run with --yes to delete matching vector docs.")
            return 0
        vector_memory = _build_vector_memory(test_mode=args.test)
        deleted = 0
        skipped = 0
        for row in rows:
            if row.discord_message_id is None:
                skipped += 1
                continue
            if vector_memory.delete_message(str(row.discord_message_id)):
                deleted += 1
        print(f"\nDeleted {deleted} vector document(s); skipped {skipped} Recall row(s) with no Discord message id.")
        return 0

    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
