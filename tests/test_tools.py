from __future__ import annotations

from datetime import datetime, UTC
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from sandy import tools


@pytest.mark.asyncio
async def test_dispatch_injects_server_id_and_strips_hallucinated_ids(monkeypatch):
    handler = AsyncMock(return_value="ok")
    monkeypatch.setitem(tools._HANDLERS, "recall_recent", handler)

    result = await tools.dispatch(
        "recall_recent",
        {
            "hours_ago": 24,
            "channel_id": 999999,
            "author_id": 123456,
            "channel": "general",
        },
        server_id=42,
        server_name="Test Server",
    )

    assert result == "ok"
    handler.assert_awaited_once_with(
        {
            "hours_ago": 24,
            "channel": "general",
            "server_id": 42,
        }
    )


@pytest.mark.asyncio
async def test_dispatch_preserves_arguments_for_non_server_scoped_tools(monkeypatch):
    handler = AsyncMock(return_value="search results")
    monkeypatch.setitem(tools._HANDLERS, "search_web", handler)

    result = await tools.dispatch(
        "search_web",
        {"query": "sandy bot", "server_id": 999, "author_id": 123},
        server_id=42,
        server_name="Test Server",
    )

    assert result == "search results"
    handler.assert_awaited_once_with(
        {"query": "sandy bot", "server_id": 999, "author_id": 123}
    )


@pytest.mark.asyncio
async def test_dispatch_unknown_tool_returns_error():
    result = await tools.dispatch(
        "definitely_not_real",
        {},
        server_id=42,
        server_name="Test Server",
    )

    assert result == "Error: unknown tool 'definitely_not_real'."


@pytest.mark.asyncio
async def test_dispatch_wraps_handler_exception(monkeypatch):
    handler = AsyncMock(side_effect=RuntimeError("boom"))
    monkeypatch.setitem(tools._HANDLERS, "dice_roll", handler)

    result = await tools.dispatch(
        "dice_roll",
        {"dice": [{"sides": 20, "count": 1}]},
        server_id=42,
        server_name="Test Server",
    )

    assert result == "Error executing dice_roll: boom"


def test_format_messages_prefers_registry_nickname(monkeypatch):
    row = SimpleNamespace(
        timestamp=datetime(2026, 3, 13, 12, 0, tzinfo=UTC),
        author_id=111,
        server_id=42,
        author_name="OldName",
        channel_name="general",
        content="hello there",
        tags=["greeting"],
        summary="said hi",
    )
    monkeypatch.setattr(
        tools,
        "_registry",
        SimpleNamespace(get_user_info=lambda author_id, server_id: {"nickname": "CurrentNick"}),
    )

    formatted = tools._format_messages([row])

    assert "<CurrentNick>" in formatted
    assert "<OldName>" not in formatted
    assert "[tags: greeting]" in formatted
    assert "(summary: said hi)" in formatted
