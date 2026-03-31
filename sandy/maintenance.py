"""Maintenance utilities for Recall and vector memory.

Examples:
    python -m sandy.maintenance recall-find --query "steam" --test
    python -m sandy.maintenance delete-vector --discord-message-id 1482282320600891422 --test
    python -m sandy.maintenance purge-vector-from-recall --query "Vault of the Vanquished" --test --yes
"""

import argparse
import os
import sys
from textwrap import shorten

from dotenv import load_dotenv

from .paths import resolve_db_dir
from .recall import ChatDatabase
from .registry import Registry
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


def _build_registry(*, test_mode: bool) -> Registry:
    db_dir = resolve_db_dir(test_mode=test_mode)
    os.environ["DB_DIR"] = str(db_dir)
    return Registry()


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


def _print_registry_lookup_rows(rows) -> None:
    if not rows:
        print("No registry rows found.")
        return
    for row in rows:
        nickname = row["nickname"] if row["nickname"] else "-"
        print(
            f"user_id={row['user_id']} user_name={row['user_name']} "
            f"server_id={row['server_id']} server_name={row['server_name']} "
            f"nickname={nickname!r} voice_admin={row['voice_admin']}",
        )


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

    voice_admin = subparsers.add_parser(
        "set-voice-admin",
        help="Set or clear registry-backed voice admin for one user/server pair",
    )
    voice_admin.add_argument("--user-id", type=int, required=True)
    voice_admin.add_argument("--server-id", type=int, required=True)
    state = voice_admin.add_mutually_exclusive_group(required=True)
    state.add_argument("--enable", action="store_true", help="Grant voice admin")
    state.add_argument("--disable", action="store_true", help="Revoke voice admin")

    lookup = subparsers.add_parser(
        "lookup-registry",
        help="Fuzzy lookup users/servers/nicknames in the registry DB",
    )
    lookup.add_argument("--user", help="Case-insensitive substring match on users.user_name")
    lookup.add_argument("--server", help="Case-insensitive substring match on servers.server_name")
    lookup.add_argument("--nickname", help="Case-insensitive substring match on user_nicknames.nickname")
    lookup.add_argument("--limit", type=int, default=20)
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

    if args.command == "set-voice-admin":
        registry = _build_registry(test_mode=args.test)
        enabled = bool(args.enable and not args.disable)
        registry.set_voice_admin(
            user_id=args.user_id,
            server_id=args.server_id,
            is_admin=enabled,
        )
        verb = "enabled" if enabled else "disabled"
        print(
            f"Voice admin {verb} for user_id={args.user_id} server_id={args.server_id}.",
        )
        return 0

    if args.command == "lookup-registry":
        registry = _build_registry(test_mode=args.test)
        user_query = f"%{(args.user or '').strip().lower()}%"
        server_query = f"%{(args.server or '').strip().lower()}%"
        nickname_query = f"%{(args.nickname or '').strip().lower()}%"
        with registry._get_conn() as conn:
            rows = conn.execute(
                """
                SELECT
                    u.user_id,
                    u.user_name,
                    s.server_id,
                    s.server_name,
                    un.nickname,
                    un.voice_admin
                FROM user_nicknames un
                JOIN users u ON u.user_id = un.user_id
                JOIN servers s ON s.server_id = un.server_id
                WHERE (? = '%%' OR lower(u.user_name) LIKE ?)
                  AND (? = '%%' OR lower(s.server_name) LIKE ?)
                  AND (? = '%%' OR lower(COALESCE(un.nickname, '')) LIKE ?)
                ORDER BY s.server_name, u.user_name
                LIMIT ?
                """,
                (
                    user_query, user_query,
                    server_query, server_query,
                    nickname_query, nickname_query,
                    args.limit,
                ),
            ).fetchall()
        _print_registry_lookup_rows(rows)
        return 0

    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
