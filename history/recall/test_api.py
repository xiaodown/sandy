#!/usr/bin/env python3
"""
Test script for the Recall API.
Demonstrates basic functionality.

Note: tests are fully vibe coded
"""

import requests
import json
import sys
from datetime import datetime

import settings

# Configuration
API_BASE = f"http://{settings.host}:{settings.port}"

def test_api():
    """Test the basic API functionality."""
    print("Testing Recall API...")
    print(f"API Base: {API_BASE}")
    print("-" * 50)
    
    # Test tracking
    tests_passed = 0
    tests_failed = 0
    total_tests = 11  # Update this if you add/remove tests
    
    # Test 1: Check if server is running
    try:
        response = requests.get(f"{API_BASE}/")
        print("✓ Server is running")
        print(f"  Response: {response.json()}")
        tests_passed += 1
    except requests.exceptions.ConnectionError:
        print("✗ Server is not running")
        print("  Please start the server with: python main.py")
        tests_failed += 1
        print("\n" + "=" * 50)
        print(f"TESTS FAILED: {tests_failed}/{total_tests} tests failed")
        print("Cannot continue testing without server running")
        sys.exit(1)
    
    print()
    
    # Test 2: Create some test messages
    from datetime import datetime, timedelta
    base_time = datetime.now()
    test_messages = [
        {
            "author": "alice",
            "content": "Hello everyone! This is my first message.",
            "timestamp": (base_time - timedelta(minutes=10)).isoformat(),
            "server": "discord-main",
            "channel": "general",
            "tags": ["greeting", "introduction"],
            "summary": "Alice's introduction"
        },
        {
            "author": "bob",
            "content": "Hey Alice! Welcome to the chat.",
            "timestamp": (base_time - timedelta(minutes=8)).isoformat(),
            "server": "discord-main",
            "channel": "general",
            "tags": ["greeting", "response"],
            "summary": "Bob welcomes Alice"
        },
        {
            "author": "charlie",
            "content": "Has anyone seen the latest project updates?",
            "timestamp": (base_time - timedelta(minutes=5)).isoformat(),
            "server": "discord-main",
            "channel": "dev-talk",
            "tags": ["question", "project"],
            "summary": "Charlie asks about project updates"
        },
        {
            "author": "alice",
            "content": "I haven't checked yet, but I can look into it.",
            "timestamp": (base_time - timedelta(minutes=2)).isoformat(),
            "server": "discord-main",
            "channel": "dev-talk",
            "tags": ["response", "project"],
            "summary": "Alice offers to check project updates"
        }
    ]
    
    created_ids = []
    print("Creating test messages...")
    all_messages_created = True
    for msg in test_messages:
        response = requests.post(f"{API_BASE}/messages/", json=msg)
        if response.status_code == 200:
            created_msg = response.json()
            created_ids.append(created_msg["id"])
            print(f"✓ Created message ID {created_msg['id']} by {msg['author']}")
        else:
            print(f"✗ Failed to create message: {response.text}")
            all_messages_created = False
    
    if all_messages_created:
        tests_passed += 1
    else:
        tests_failed += 1
    
    print()
    
    # Test 3: Get all messages
    print("Retrieving all messages...")
    response = requests.get(f"{API_BASE}/messages/")
    if response.status_code == 200:
        messages = response.json()
        print(f"✓ Retrieved {len(messages)} messages")
        for msg in messages:
            print(f"  [{msg['id']}] {msg['author']}: {msg['content'][:50]}...")
        tests_passed += 1
    else:
        print(f"✗ Failed to get messages: {response.text}")
        tests_failed += 1
    
    print()
    
    # Test 4: Filter by author
    print("Filtering messages by author 'alice'...")
    response = requests.get(f"{API_BASE}/messages/?author=alice")
    if response.status_code == 200:
        alice_messages = response.json()
        print(f"✓ Found {len(alice_messages)} messages from Alice")
        for msg in alice_messages:
            print(f"  [{msg['id']}] {msg['content'][:50]}...")
        tests_passed += 1
    else:
        print(f"✗ Failed to filter messages: {response.text}")
        tests_failed += 1
    
    print()
    
    # Test 5: Filter by tag
    print("Filtering messages by tag 'project'...")
    response = requests.get(f"{API_BASE}/messages/?tag=project")
    if response.status_code == 200:
        project_messages = response.json()
        print(f"✓ Found {len(project_messages)} messages with 'project' tag")
        for msg in project_messages:
            print(f"  [{msg['id']}] {msg['author']}: {msg['content'][:50]}...")
        tests_passed += 1
    else:
        print(f"✗ Failed to filter by tag: {response.text}")
        tests_failed += 1
    
    print()
    
    # Test 6: Filter by server
    print("Filtering messages by server 'discord-main'...")
    response = requests.get(f"{API_BASE}/messages/?server=discord-main")
    if response.status_code == 200:
        server_messages = response.json()
        print(f"✓ Found {len(server_messages)} messages from discord-main server")
        for msg in server_messages:
            print(f"  [{msg['id']}] {msg['channel']}: {msg['author']}: {msg['content'][:50]}...")
        tests_passed += 1
    else:
        print(f"✗ Failed to filter by server: {response.text}")
        tests_failed += 1
    
    print()
    
    # Test 7: Filter by channel
    print("Filtering messages by channel 'dev-talk'...")
    response = requests.get(f"{API_BASE}/messages/?channel=dev-talk")
    if response.status_code == 200:
        channel_messages = response.json()
        print(f"✓ Found {len(channel_messages)} messages from dev-talk channel")
        for msg in channel_messages:
            print(f"  [{msg['id']}] {msg['author']}: {msg['content'][:50]}...")
        tests_passed += 1
    else:
        print(f"✗ Failed to filter by channel: {response.text}")
        tests_failed += 1
    
    print()
    
    # Test 8: Get specific message
    if created_ids:
        msg_id = created_ids[0]
        print(f"Getting specific message (ID {msg_id})...")
        response = requests.get(f"{API_BASE}/messages/{msg_id}")
        if response.status_code == 200:
            msg = response.json()
            print(f"✓ Retrieved message: {msg['author']}: {msg['content']}")
            print(f"  Server: {msg['server']}")
            print(f"  Channel: {msg['channel']}")
            print(f"  Tags: {msg.get('tags', 'None')}")
            print(f"  Summary: {msg.get('summary', 'None')}")
            tests_passed += 1
        else:
            print(f"✗ Failed to get specific message: {response.text}")
            tests_failed += 1
    else:
        print("✗ No messages created, skipping specific message test")
        tests_failed += 1
    
    print()
    
    # Test 9: Get statistics
    print("Getting database statistics...")
    response = requests.get(f"{API_BASE}/stats/")
    if response.status_code == 200:
        stats = response.json()
        print("✓ Statistics:")
        print(f"  Total messages: {stats['total_messages']}")
        print(f"  Unique authors: {stats['unique_authors']}")
        print(f"  Latest message: {stats['latest_message_time']}")
        tests_passed += 1
    else:
        print(f"✗ Failed to get stats: {response.text}")
        tests_failed += 1
    
    # Test 10: Time range filtering - last hour
    print("Testing time range filtering (last hour)...")
    response = requests.get(f"{API_BASE}/messages/?hours_ago=1")
    if response.status_code == 200:
        recent_messages = response.json()
        print(f"✓ Found {len(recent_messages)} messages from the last hour")
        for msg in recent_messages:
            print(f"  [{msg['id']}] {msg['author']}: {msg['content'][:50]}...")
        tests_passed += 1
    else:
        print(f"✗ Failed to filter by time range: {response.text}")
        tests_failed += 1
    
    print()
    
    # Test 11: Time range filtering - specific date range
    print("Testing specific date range filtering...")
    now = datetime.now()
    since_time = (now - timedelta(minutes=5)).isoformat()
    until_time = now.isoformat()
    
    response = requests.get(f"{API_BASE}/messages/?since={since_time}&until={until_time}")
    if response.status_code == 200:
        range_messages = response.json()
        print(f"✓ Found {len(range_messages)} messages in the specified range")
        print(f"  Range: {since_time} to {until_time}")
        tests_passed += 1
    else:
        print(f"✗ Failed to filter by date range: {response.text}")
        tests_failed += 1
    
    print()
    
    # Cleanup: Delete test messages we created
    if created_ids:
        print("Cleaning up test messages...")
        cleanup_success = True
        for msg_id in created_ids:
            response = requests.delete(f"{API_BASE}/messages/{msg_id}")
            if response.status_code == 200:
                print(f"✓ Deleted message ID {msg_id}")
            else:
                print(f"✗ Failed to delete message ID {msg_id}: {response.text}")
                cleanup_success = False
        
        if cleanup_success:
            print("✓ All test messages cleaned up successfully")
        else:
            print("⚠ Some test messages may not have been cleaned up")
    
    print()
    
    # Test summary
    print("=" * 50)
    if tests_failed == 0:
        print(f"✓ ALL TESTS PASSED: {tests_passed}/{total_tests} tests successful")
    else:
        print(f"✗ TESTS FAILED: {tests_failed}/{total_tests} tests failed, {tests_passed} passed")
    
    # Exit with appropriate code for CI/CD
    if tests_failed == 0:
        sys.exit(0)  # Success
    else:
        sys.exit(1)  # Failure


if __name__ == "__main__":
    test_api()