from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
from pathlib import Path

from tts_service.app import FasterCloneService, ServiceConfig


REPO_ROOT = Path(__file__).resolve().parents[2]
RECALL_DB = Path(os.getenv("SANDY_RECALL_DB", str(REPO_ROOT / "data" / "prod" / "recall.db")))
OUTPUT_DIR = REPO_ROOT / "tts_service" / "assets" / "candidate_renders"
MANIFEST_PATH = OUTPUT_DIR / "manifest.json"
TEXT_PATH = OUTPUT_DIR / "manifest.txt"

CURATED_IDS = [
    1087,
    1074,
    1071,
    1048,
    1042,
    1022,
    1014,
    982,
    956,
    934,
    889,
    879,
    809,
    785,
    1030,
    1028,
    990,
]


def _slug(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:48] or "candidate"


def _fetch_candidates(db_path: Path, ids: list[int]) -> list[dict[str, object]]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            f"""
            SELECT id, timestamp, channel_name, content
            FROM chat_messages
            WHERE id IN ({",".join("?" for _ in ids)})
            ORDER BY timestamp DESC
            """,
            ids,
        ).fetchall()
    finally:
        conn.close()

    by_id = {int(row["id"]): row for row in rows}
    ordered: list[dict[str, object]] = []
    for message_id in ids:
        row = by_id.get(message_id)
        if row is None:
            continue
        ordered.append(
            {
                "id": int(row["id"]),
                "timestamp": row["timestamp"],
                "channel_name": row["channel_name"],
                "text": row["content"].strip(),
            },
        )
    return ordered


def main() -> int:
    parser = argparse.ArgumentParser(description="Render Sandy voice audition candidates from Recall.")
    parser.add_argument("--limit", type=int, default=len(CURATED_IDS))
    args = parser.parse_args()

    candidates = _fetch_candidates(RECALL_DB, CURATED_IDS[: args.limit])
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    service = FasterCloneService(ServiceConfig())
    manifest: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []

    for index, candidate in enumerate(candidates, start=1):
        text = str(candidate["text"])
        print(f"[{index:02d}/{len(candidates):02d}] rendering recall_id={candidate['id']} text={text!r}")
        sys.stdout.flush()
        try:
            wav_bytes = service.synthesize_wav_bytes(text)
        except Exception as exc:
            print(f"  -> failed: {exc}")
            sys.stdout.flush()
            failures.append(
                {
                    "index": index,
                    "id": candidate["id"],
                    "timestamp": candidate["timestamp"],
                    "channel_name": candidate["channel_name"],
                    "text": text,
                    "error": str(exc),
                },
            )
            continue

        filename = f"{index:02d}-{int(candidate['id'])}-{_slug(text)}.wav"
        output_path = OUTPUT_DIR / filename
        output_path.write_bytes(wav_bytes)
        print(f"  -> wrote {output_path.name} ({len(wav_bytes)} bytes)")
        sys.stdout.flush()
        manifest.append(
            {
                "index": index,
                "id": candidate["id"],
                "timestamp": candidate["timestamp"],
                "channel_name": candidate["channel_name"],
                "filename": filename,
                "text": text,
            },
        )

    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "failures.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")
    lines = [
        f"{item['index']:02d}. {item['filename']}\n"
        f"    recall_id={item['id']} channel={item['channel_name']} timestamp={item['timestamp']}\n"
        f"    {item['text']}\n"
        for item in manifest
    ]
    TEXT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Rendered {len(manifest)} candidate reference wav(s) to {OUTPUT_DIR}")
    print(f"Skipped {len(failures)} candidate(s) that failed synthesis")
    print(f"Manifest: {MANIFEST_PATH}")
    print(f"Text list: {TEXT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
