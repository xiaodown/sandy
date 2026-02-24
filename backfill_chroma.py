#!/usr/bin/env python3
"""
Backfill ChromaDB from the Recall SQLite database.

Reads all messages from the Recall SQLite database directly and generates
embeddings for any message not already present in the Chroma collection.
Safe to re-run: already-embedded messages are skipped (upsert by ID).

Usage
-----
    python backfill_chroma.py
    python backfill_chroma.py --db path/to/history.db
    python backfill_chroma.py --dry-run
    python backfill_chroma.py --limit 500

Options
-------
    --db PATH      Path to the Recall SQLite database.
                   Default: history/database/history.db
    --dry-run      Print what would be embedded without writing anything.
    --limit N      Stop after embedding N new messages (useful for testing).
    --batch N      Log progress every N messages (default: 50).
"""

import argparse
import asyncio
import logging
import os
import sqlite3
import sys
from datetime import datetime

# Run from the sandy root so .env and vector_memory are importable.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from vector_memory import VectorMemory

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# Default path relative to sandy root: history/database/history.db
_DEFAULT_DB = os.path.join("history", "database", "history.db")


def fetch_messages(db_path: str) -> list[dict]:
    """Return all messages from the Recall SQLite database, oldest first."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT
                id,
                author_name,
                author_id,
                server_id,
                content,
                timestamp
            FROM chat_messages
            ORDER BY timestamp ASC
            """
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


async def backfill(db_path: str, dry_run: bool, limit: int | None, batch: int) -> None:
    vm = VectorMemory()

    # Fetch the set of IDs already in Chroma so we can skip them.
    existing_ids: set[str] = set(vm._collection.get(include=[])["ids"])
    logger.info("Chroma already contains %d document(s)", len(existing_ids))

    messages = fetch_messages(db_path)
    logger.info("Recall DB contains %d message(s) total", len(messages))

    added = skipped = errors = 0

    for msg in messages:
        if limit is not None and added >= limit:
            logger.info("Reached --limit %d, stopping", limit)
            break

        msg_id  = str(msg["id"])
        content = (msg.get("content") or "").strip()

        # Skip already-embedded messages.
        if msg_id in existing_ids:
            skipped += 1
            continue

        # Skip empty or placeholder content.
        if not content or content == "(no text content)":
            skipped += 1
            continue

        if dry_run:
            logger.info(
                "[dry-run] Would embed id=%-8s  server=%-20s  author=%s",
                msg_id,
                msg.get("server_id", "?"),
                msg.get("author_name", "?"),
            )
            added += 1
            continue

        try:
            ts_raw = msg.get("timestamp", "")
            try:
                ts = datetime.fromisoformat(ts_raw)
            except Exception:
                ts = datetime.utcnow()

            await vm.add_message(
                message_id  = msg_id,
                content     = content,
                author_name = msg.get("author_name", "?"),
                server_id   = int(msg.get("server_id") or 0),
                timestamp   = ts,
            )
            added += 1

            if added % batch == 0:
                logger.info(
                    "Progress: %d embedded, %d skipped, %d errors",
                    added, skipped, errors,
                )

        except Exception as exc:
            logger.error("Failed to embed id=%s: %s", msg_id, exc)
            errors += 1

    logger.info(
        "Backfill complete — added=%d  skipped=%d  errors=%d",
        added, skipped, errors,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill ChromaDB from the Recall SQLite database"
    )
    parser.add_argument(
        "--db",
        default=_DEFAULT_DB,
        help=f"Path to Recall SQLite DB (default: {_DEFAULT_DB})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be embedded without writing",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Stop after embedding N new messages",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=50,
        metavar="N",
        help="Log progress every N messages (default: 50)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.db):
        logger.error("Database not found: %s", args.db)
        logger.error("Run from the sandy root directory, or pass --db explicitly.")
        sys.exit(1)

    asyncio.run(backfill(args.db, args.dry_run, args.limit, args.batch))


if __name__ == "__main__":
    main()
