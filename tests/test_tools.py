from __future__ import annotations

from datetime import datetime, UTC
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import httpx

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


@pytest.mark.asyncio
async def test_recall_query_translates_argument_names_and_drops_none(monkeypatch):
    calls = []

    def get_messages(**kwargs):
        calls.append(kwargs)
        return ["row"]

    monkeypatch.setattr(tools, "_recall_db", SimpleNamespace(get_messages=get_messages))

    result = await tools._recall_query(
        author="friend",
        query="pizza",
        channel="general",
        server_id=42,
        hours_ago=24,
    )

    assert result == ["row"]
    assert calls == [
        {
            "author_name": "friend",
            "channel_name": "general",
            "q": "pizza",
            "server_id": 42,
            "hours_ago": 24,
        }
    ]


@pytest.mark.asyncio
async def test_recall_query_returns_none_when_db_not_initialized(monkeypatch):
    monkeypatch.setattr(tools, "_recall_db", None)

    result = await tools._recall_query(server_id=42)

    assert result is None


@pytest.mark.asyncio
async def test_recall_recent_formats_success(monkeypatch):
    monkeypatch.setattr(tools, "_recall_query", AsyncMock(return_value=["row1", "row2"]))
    monkeypatch.setattr(tools, "_format_messages", lambda data: "formatted recall")

    result = await tools._handle_recall_recent({"hours_ago": 24})

    assert result == "2 message(s) retrieved:\n\nformatted recall"


@pytest.mark.asyncio
async def test_recall_from_user_formats_empty_and_error_cases(monkeypatch):
    monkeypatch.setattr(tools, "_recall_query", AsyncMock(return_value=[]))

    empty = await tools._handle_recall_from_user({"author": "friend"})

    monkeypatch.setattr(tools, "_recall_query", AsyncMock(return_value=None))
    error = await tools._handle_recall_from_user({"author": "friend"})

    assert empty == "No messages found from author: friend"
    assert error == "Error: could not reach the memory store."


class FakeSearchResponse:
    def __init__(self, payload, *, raise_error: Exception | None = None) -> None:
        self._payload = payload
        self._raise_error = raise_error

    def raise_for_status(self) -> None:
        if self._raise_error is not None:
            raise self._raise_error

    def json(self):
        return self._payload


class FakeAsyncClient:
    def __init__(self, response: FakeSearchResponse) -> None:
        self._response = response
        self.calls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False

    async def get(self, url, *, params=None, headers=None):
        self.calls.append({"url": url, "params": params, "headers": headers})
        return self._response


@pytest.mark.asyncio
async def test_search_web_handles_empty_query():
    result = await tools._handle_search_web({"query": "   "})

    assert result == "Error: no search query provided."


@pytest.mark.asyncio
async def test_search_web_formats_results_and_truncates_snippets(monkeypatch):
    long_snippet = "word " * 80
    fake_client = FakeAsyncClient(
        FakeSearchResponse(
            {
                "results": [
                    {
                        "title": "Result One",
                        "url": "https://example.com/1",
                        "content": long_snippet,
                    },
                    {
                        "title": "Result Two",
                        "url": "https://example.com/2",
                        "content": "short snippet",
                    },
                ]
            }
        )
    )
    monkeypatch.setattr(
        tools.httpx,
        "AsyncClient",
        lambda timeout: fake_client,
    )

    result = await tools._handle_search_web({"query": "sandy", "n_results": 1})

    assert "Web search results for 'sandy':" in result
    assert "1. Result One" in result
    assert "Result Two" not in result
    assert "https://example.com/1" in result
    assert "…" in result
    assert fake_client.calls[0]["params"]["q"] == "sandy"


@pytest.mark.asyncio
async def test_search_web_returns_error_on_http_failure(monkeypatch):
    fake_client = FakeAsyncClient(
        FakeSearchResponse({}, raise_error=httpx.HTTPStatusError(
            "bad gateway",
            request=httpx.Request("GET", "http://test/search"),
            response=httpx.Response(502),
        ))
    )
    monkeypatch.setattr(tools.httpx, "AsyncClient", lambda timeout: fake_client)

    result = await tools._handle_search_web({"query": "sandy"})

    assert "Error reaching web search:" in result


@pytest.mark.asyncio
async def test_search_web_returns_no_results_message(monkeypatch):
    fake_client = FakeAsyncClient(FakeSearchResponse({"results": []}))
    monkeypatch.setattr(tools.httpx, "AsyncClient", lambda timeout: fake_client)

    result = await tools._handle_search_web({"query": "sandy"})

    assert result == "No web results found for: sandy"


@pytest.mark.asyncio
async def test_dice_roll_clamps_out_of_range_values(monkeypatch):
    monkeypatch.setattr(tools.random, "randint", lambda a, b: b)

    result = await tools._handle_dice_roll(
        {"dice": [{"sides": 999, "count": 0}, {"sides": 0, "count": 12}]}
    )

    assert result.splitlines() == [
        "rolled 1 100-sided die: 100",
        "rolled 10 1-sided dice: 1 1 1 1 1 1 1 1 1 1",
    ]
