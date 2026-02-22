#!/usr/bin/env python3
"""
Test script for the Recall API.
Uses realistic Discord snowflake IDs and the updated schema
(author_id/channel_id/server_id + human-readable name fields).
"""

import requests
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")
API_BASE = f"http://{os.getenv('RECALL_HOST', '127.0.0.1')}:{os.getenv('RECALL_PORT', '8000')}"

# ---------------------------------------------------------------------------
# Fake Discord snowflake IDs — realistic-looking but made up
# ---------------------------------------------------------------------------
SERVER_ID_MAIN   = 1359032772332621875
SERVER_NAME_MAIN = "Test Server Alpha"

SERVER_ID_ALT    = 1359032772332621900
SERVER_NAME_ALT  = "Test Server Beta"

CHANNEL_ID_GENERAL  = 1359032772332621878
CHANNEL_NAME_GENERAL = "general"

CHANNEL_ID_DEV   = 1359032772332621879
CHANNEL_NAME_DEV = "dev-talk"

USER_ID_ALICE    = 218896334130905001
USER_NAME_ALICE  = "alice"

USER_ID_BOB      = 218896334130905002
USER_NAME_BOB    = "bob_the_builder"

USER_ID_CHARLIE  = 218896334130905003
USER_NAME_CHARLIE = "charlie"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ok(label: str):
    print(f"  ✓ {label}")

def fail(label: str, detail: str = ""):
    print(f"  ✗ {label}" + (f": {detail}" if detail else ""))


def pre_cleanup():
    """Delete any messages left over from a previous failed test run."""
    try:
        r = requests.get(f"{API_BASE}/messages/?server_id={SERVER_ID_MAIN}&limit=1000")
        if r.status_code == 200:
            for msg in r.json():
                requests.delete(f"{API_BASE}/messages/{msg['id']}")
        r = requests.get(f"{API_BASE}/messages/?server_id={SERVER_ID_ALT}&limit=1000")
        if r.status_code == 200:
            for msg in r.json():
                requests.delete(f"{API_BASE}/messages/{msg['id']}")
    except requests.exceptions.ConnectionError:
        pass  # Server not running yet — reachability test will handle it


def run_tests():
    passed = 0
    failed = 0

    def check(condition: bool, label: str, detail: str = ""):
        nonlocal passed, failed
        if condition:
            ok(label)
            passed += 1
        else:
            fail(label, detail)
            failed += 1

    print(f"Recall API Test Suite")
    print(f"Target: {API_BASE}")
    print("=" * 60)

    pre_cleanup()

    # ------------------------------------------------------------------
    # 1. Server reachability
    # ------------------------------------------------------------------
    print("\n[1] Server health check")
    try:
        r = requests.get(f"{API_BASE}/")
        check(r.status_code == 200, "Server is reachable")
    except requests.exceptions.ConnectionError:
        fail("Server is reachable", "Connection refused — start the server with: python main.py")
        print("\nCannot continue without a running server. Exiting.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 2. Create test messages
    # ------------------------------------------------------------------
    print("\n[2] Create test messages")
    base_time = datetime.now()
    test_messages = [
        {
            "author_id":    USER_ID_ALICE,
            "author_name":  USER_NAME_ALICE,
            "channel_id":   CHANNEL_ID_GENERAL,
            "channel_name": CHANNEL_NAME_GENERAL,
            "server_id":    SERVER_ID_MAIN,
            "server_name":  SERVER_NAME_MAIN,
            "content":      "Hello everyone! This is my first message.",
            "timestamp":    (base_time - timedelta(minutes=10)).isoformat(),
            "tags":         ["greeting", "introduction"],
            "summary":      "Alice's introduction",
        },
        {
            "author_id":    USER_ID_BOB,
            "author_name":  USER_NAME_BOB,
            "channel_id":   CHANNEL_ID_GENERAL,
            "channel_name": CHANNEL_NAME_GENERAL,
            "server_id":    SERVER_ID_MAIN,
            "server_name":  SERVER_NAME_MAIN,
            "content":      "Hey Alice! Welcome to the chat.",
            "timestamp":    (base_time - timedelta(minutes=8)).isoformat(),
            "tags":         ["greeting", "response"],
            "summary":      "Bob welcomes Alice",
        },
        {
            "author_id":    USER_ID_CHARLIE,
            "author_name":  USER_NAME_CHARLIE,
            "channel_id":   CHANNEL_ID_DEV,
            "channel_name": CHANNEL_NAME_DEV,
            "server_id":    SERVER_ID_MAIN,
            "server_name":  SERVER_NAME_MAIN,
            "content":      "Has anyone seen the latest **project** updates? Check <#dev-updates>",
            "timestamp":    (base_time - timedelta(minutes=5)).isoformat(),
            "tags":         ["question", "project"],
            "summary":      "Charlie asks about project updates",
        },
        {
            "author_id":    USER_ID_ALICE,
            "author_name":  USER_NAME_ALICE,
            "channel_id":   CHANNEL_ID_DEV,
            "channel_name": CHANNEL_NAME_DEV,
            "server_id":    SERVER_ID_MAIN,
            "server_name":  SERVER_NAME_MAIN,
            "content":      "I haven't checked yet, but I can look into it.",
            "timestamp":    (base_time - timedelta(minutes=2)).isoformat(),
            "tags":         ["response", "project"],
            "summary":      "Alice offers to check project updates",
        },
        {
            # Message on a completely different server — tests server isolation
            "author_id":    USER_ID_BOB,
            "author_name":  USER_NAME_BOB,
            "channel_id":   CHANNEL_ID_GENERAL + 100,
            "channel_name": "general",
            "server_id":    SERVER_ID_ALT,
            "server_name":  SERVER_NAME_ALT,
            "content":      "This message is on a different server entirely.",
            "timestamp":    (base_time - timedelta(minutes=1)).isoformat(),
            "tags":         ["other-server"],
        },
    ]

    created_ids = []
    all_created = True
    for msg in test_messages:
        r = requests.post(f"{API_BASE}/messages/", json=msg)
        if r.status_code == 200:
            created_ids.append(r.json()["id"])
            ok(f"Created message ID {created_ids[-1]} ({msg['author_name']} in {msg['channel_name']})")
        else:
            fail(f"Create message for {msg['author_name']}", r.text)
            all_created = False

    check(all_created, f"All {len(test_messages)} messages created successfully")

    # ------------------------------------------------------------------
    # 3. Retrieve all messages
    # ------------------------------------------------------------------
    print("\n[3] Retrieve all messages")
    r = requests.get(f"{API_BASE}/messages/")
    check(r.status_code == 200, "GET /messages/ returns 200")
    if r.status_code == 200:
        msgs = r.json()
        check(len(msgs) >= len(test_messages), f"At least {len(test_messages)} messages returned (got {len(msgs)})")
        # Verify new field names are present
        if msgs:
            m = msgs[0]
            check("author_id"    in m, "Response has author_id field")
            check("author_name"  in m, "Response has author_name field")
            check("channel_id"   in m, "Response has channel_id field")
            check("channel_name" in m, "Response has channel_name field")
            check("server_id"    in m, "Response has server_id field")
            check("server_name"  in m, "Response has server_name field")
            check("author"       not in m, "Old 'author' field is gone (not in response)")
            check("server"       not in m, "Old 'server' field is gone (not in response)")
            check("channel"      not in m, "Old 'channel' field is gone (not in response)")

    # ------------------------------------------------------------------
    # 4. Filter by author_name (convenience)
    # ------------------------------------------------------------------
    print(f"\n[4] Filter by author name '{USER_NAME_ALICE}'")
    r = requests.get(f"{API_BASE}/messages/?author={USER_NAME_ALICE}")
    check(r.status_code == 200, "Request succeeded")
    if r.status_code == 200:
        alice_msgs = r.json()
        check(len(alice_msgs) == 2, f"Got 2 Alice messages (got {len(alice_msgs)})")
        check(all(m["author_id"] == USER_ID_ALICE for m in alice_msgs), "All returned messages have Alice's user_id")

    # ------------------------------------------------------------------
    # 5. Filter by author_id (precise)
    # ------------------------------------------------------------------
    print(f"\n[5] Filter by author_id {USER_ID_BOB}")
    r = requests.get(f"{API_BASE}/messages/?author_id={USER_ID_BOB}")
    check(r.status_code == 200, "Request succeeded")
    if r.status_code == 200:
        bob_msgs = r.json()
        # Bob is on two different servers — should get both messages (or just main server ones
        # depending on whether we also filter by server; here no server filter so both)
        check(len(bob_msgs) >= 1, f"Got at least 1 Bob message (got {len(bob_msgs)})")
        check(all(m["author_id"] == USER_ID_BOB for m in bob_msgs), "All messages have Bob's ID")

    # ------------------------------------------------------------------
    # 6. Filter by tag
    # ------------------------------------------------------------------
    print("\n[6] Filter by tag 'project'")
    r = requests.get(f"{API_BASE}/messages/?tag=project")
    check(r.status_code == 200, "Request succeeded")
    if r.status_code == 200:
        proj_msgs = r.json()
        check(len(proj_msgs) == 2, f"Got 2 'project' tagged messages (got {len(proj_msgs)})")
        check(all("project" in (m.get("tags") or []) for m in proj_msgs), "All returned messages actually have 'project' tag")

    # ------------------------------------------------------------------
    # 7. Filter by server_name (convenience)
    # ------------------------------------------------------------------
    print(f"\n[7] Filter by server name '{SERVER_NAME_MAIN}'")
    r = requests.get(f"{API_BASE}/messages/?server={SERVER_NAME_MAIN}")
    check(r.status_code == 200, "Request succeeded")
    if r.status_code == 200:
        main_msgs = r.json()
        check(len(main_msgs) == 4, f"Got 4 messages from main server (got {len(main_msgs)})")
        check(all(m["server_id"] == SERVER_ID_MAIN for m in main_msgs), "All messages have correct server_id")

    # ------------------------------------------------------------------
    # 8. Filter by server_id (precise)
    # ------------------------------------------------------------------
    print(f"\n[8] Filter by server_id {SERVER_ID_ALT}")
    r = requests.get(f"{API_BASE}/messages/?server_id={SERVER_ID_ALT}")
    check(r.status_code == 200, "Request succeeded")
    if r.status_code == 200:
        alt_msgs = r.json()
        check(len(alt_msgs) == 1, f"Got 1 message from alt server (got {len(alt_msgs)})")

    # ------------------------------------------------------------------
    # 9. Filter by channel_name (convenience)
    # ------------------------------------------------------------------
    print(f"\n[9] Filter by channel name '{CHANNEL_NAME_DEV}'")
    r = requests.get(f"{API_BASE}/messages/?channel={CHANNEL_NAME_DEV}")
    check(r.status_code == 200, "Request succeeded")
    if r.status_code == 200:
        dev_msgs = r.json()
        check(len(dev_msgs) == 2, f"Got 2 dev-talk messages (got {len(dev_msgs)})")

    # ------------------------------------------------------------------
    # 10. Filter by channel_id (precise)
    # ------------------------------------------------------------------
    print(f"\n[10] Filter by channel_id {CHANNEL_ID_GENERAL}")
    r = requests.get(f"{API_BASE}/messages/?channel_id={CHANNEL_ID_GENERAL}")
    check(r.status_code == 200, "Request succeeded")
    if r.status_code == 200:
        gen_msgs = r.json()
        check(len(gen_msgs) == 2, f"Got 2 general channel messages (got {len(gen_msgs)})")
        check(all(m["channel_id"] == CHANNEL_ID_GENERAL for m in gen_msgs), "All messages have correct channel_id")

    # ------------------------------------------------------------------
    # 11. Get specific message by ID
    # ------------------------------------------------------------------
    print("\n[11] Get specific message by ID")
    if created_ids:
        msg_id = created_ids[0]
        r = requests.get(f"{API_BASE}/messages/{msg_id}")
        check(r.status_code == 200, f"GET /messages/{msg_id} returns 200")
        if r.status_code == 200:
            m = r.json()
            check(m["author_id"]   == USER_ID_ALICE,      "author_id matches")
            check(m["author_name"] == USER_NAME_ALICE,    "author_name matches")
            check(m["channel_id"]  == CHANNEL_ID_GENERAL, "channel_id matches")
            check(m["server_id"]   == SERVER_ID_MAIN,     "server_id matches")
            check(m["server_name"] == SERVER_NAME_MAIN,   "server_name matches")
            check(set(m.get("tags") or []) == {"greeting", "introduction"}, "Tags correct")

    # ------------------------------------------------------------------
    # 12. Statistics
    # ------------------------------------------------------------------
    print("\n[12] Statistics endpoint")
    r = requests.get(f"{API_BASE}/stats/")
    check(r.status_code == 200, "GET /stats/ returns 200")
    if r.status_code == 200:
        stats = r.json()
        check("total_messages"  in stats, "Has total_messages")
        check("unique_authors"  in stats, "Has unique_authors")
        check("unique_servers"  in stats, "Has unique_servers")
        check("total_tags"      in stats, "Has total_tags")
        check(stats["unique_servers"] >= 2, f"At least 2 unique servers (got {stats.get('unique_servers')})")
        check(stats["total_tags"] >= 5,     f"At least 5 unique tags (got {stats.get('total_tags')})")
        print(f"       total_messages={stats['total_messages']}, unique_authors={stats['unique_authors']}, "
              f"unique_servers={stats['unique_servers']}, total_tags={stats['total_tags']}")

    # ------------------------------------------------------------------
    # 13. Time filtering: hours_ago
    # ------------------------------------------------------------------
    print("\n[13] Time filter: hours_ago=1")
    r = requests.get(f"{API_BASE}/messages/?hours_ago=1")
    check(r.status_code == 200, "Request succeeded")
    if r.status_code == 200:
        recent = r.json()
        check(len(recent) >= len(test_messages), f"All test messages within last hour (got {len(recent)})")

    # ------------------------------------------------------------------
    # 14. Time filtering: explicit since/until range
    # ------------------------------------------------------------------
    print("\n[14] Time filter: since/until range covering last 6 minutes")
    since_str = (base_time - timedelta(minutes=6)).isoformat()
    until_str = base_time.isoformat()
    r = requests.get(f"{API_BASE}/messages/?since={since_str}&until={until_str}")
    check(r.status_code == 200, "Request succeeded")
    if r.status_code == 200:
        ranged = r.json()
        # Messages timestamped 5, 2, and 1 minutes ago fall in this window
        check(len(ranged) == 3, f"Got 3 messages in 6-minute window (got {len(ranged)})")

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    print("\n[Cleanup] Deleting test messages...")
    cleanup_ok = True
    for msg_id in created_ids:
        r = requests.delete(f"{API_BASE}/messages/{msg_id}")
        if r.status_code == 200:
            ok(f"Deleted message ID {msg_id}")
        else:
            fail(f"Delete message ID {msg_id}", r.text)
            cleanup_ok = False
    if cleanup_ok:
        ok("All test messages cleaned up")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    total = passed + failed
    print("\n" + "=" * 60)
    if failed == 0:
        print(f"✓ ALL TESTS PASSED: {passed}/{total}")
        sys.exit(0)
    else:
        print(f"✗ {failed}/{total} FAILED, {passed} passed")
        sys.exit(1)


if __name__ == "__main__":
    run_tests()
