from sandy.llm import (
    BouncerResponse,
    OllamaInterface,
    _coerce_bouncer_tool_selection,
    _infer_steam_browse_category,
    _looks_like_direct_image_ask,
)
from sandy.config import LlmConfig


def test_infer_steam_category_prefers_explicit_latest_message():
    context = "\n".join(
        [
            "[2m ago] [alice] what's good on steam right now?",
            "[just now] [alice] what is on sale on steam?",
        ]
    )

    assert _infer_steam_browse_category(context) == "specials"


def test_infer_steam_category_uses_recent_history_for_followup():
    context = "\n".join(
        [
            "[30s ago] [alice] can you tell me what's coming soon on steam?",
            "[15s ago] [Sandy] sure, let me check.",
            "[just now] [alice] nah, can you check actual steam?",
        ]
    )

    assert _infer_steam_browse_category(context) == "upcoming"


def test_coerce_bouncer_search_web_to_steam_browse_for_storefront_turn():
    context = "\n".join(
        [
            "[20s ago] [alice] alright, check steam again.",
            "[just now] [alice] just want you to check steam again for things that are on sale.",
        ]
    )
    result = BouncerResponse(
        should_respond=True,
        reason="direct question",
        use_tool=True,
        recommended_tool="search_web",
        tool_parameters={"query": "Steam sales March 2023", "n_results": 7},
    )

    coerced = _coerce_bouncer_tool_selection(context, result)

    assert coerced.recommended_tool == "steam_browse"
    assert coerced.tool_parameters == {"category": "specials", "limit": 7}


def test_coerce_bouncer_leaves_specific_game_lookup_alone():
    context = "\n".join(
        [
            "[40s ago] [alice] can you tell me what's coming soon on steam?",
            "[20s ago] [Sandy] here's a few upcoming games.",
            "[just now] [alice] tell me more about Vault of the Vanquished",
        ]
    )
    result = BouncerResponse(
        should_respond=True,
        reason="specific game ask",
        use_tool=True,
        recommended_tool="search_web",
        tool_parameters={"query": "Vault of the Vanquished release date"},
    )

    coerced = _coerce_bouncer_tool_selection(context, result)

    assert coerced.recommended_tool == "search_web"
    assert coerced.tool_parameters == {"query": "Vault of the Vanquished release date"}


def test_looks_like_direct_image_ask_requires_sandy_and_picture_language():
    context = "\n".join(
        [
            "[20s ago] [alice] random chatter",
            "[just now] [alice] hey sandy, can you tell me what you think of this picture?",
        ]
    )

    assert _looks_like_direct_image_ask(context) is True

    # Without 'sandy' in the message, it should not match
    context_other = "\n".join(
        [
            "[20s ago] [alice] random chatter",
            "[just now] [alice] hey bob, can you tell me what you think of this picture?",
        ]
    )
    assert _looks_like_direct_image_ask(context_other) is False


def test_coerce_bouncer_no_respond_to_true_for_direct_image_ask():
    context = "\n".join(
        [
            "[20s ago] [alice] hi",
            "[just now] [alice] hey sandy, can you tell me what you think of this image?",
        ]
    )
    result = BouncerResponse(
        should_respond=False,
        reason="model got it wrong",
        use_tool=False,
        recommended_tool=None,
        tool_parameters=None,
    )

    coerced = _coerce_bouncer_tool_selection(context, result)

    assert coerced.should_respond is True
    assert "attached image or picture" in coerced.reason


async def test_ask_brain_can_use_vllm_chat_completions(monkeypatch):
    captured: dict[str, object] = {}

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "choices": [
                    {
                        "message": {"content": "vllm sandy reply"},
                        "finish_reason": "stop",
                    }
                ]
            }

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs) -> None:
            captured["timeout"] = kwargs.get("timeout")

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def post(self, url, *, json, headers):
            captured["url"] = url
            captured["json"] = json
            captured["headers"] = headers
            return FakeResponse()

    monkeypatch.setattr("sandy.llm.httpx.AsyncClient", FakeAsyncClient)

    llm = OllamaInterface(
        LlmConfig(
            brain_provider="vllm",
            brain_base_url="http://brain.test/v1",
            brain_api_key="secret-ish",
            brain_model="mistral-brain",
            brain_temperature=0.42,
            brain_num_predict=123,
            brain_num_ctx=4096,
            brain_reasoning_effort="none",
        )
    )

    response = await llm.ask_brain(
        [{"role": "user", "content": "hey sandy"}],
        server_name="test server",
        channel_name="bot-lab",
    )

    assert response is not None
    assert response.content == "vllm sandy reply"
    assert response.done_reason == "stop"
    assert response.eval_count is None
    assert captured["url"] == "http://brain.test/v1/chat/completions"
    payload = captured["json"]
    assert payload["model"] == "mistral-brain"
    assert payload["temperature"] == 0.42
    assert payload["max_tokens"] == 123
    assert payload["reasoning_effort"] == "none"
    assert payload["stream"] is False
    assert payload["messages"][0]["role"] == "system"
    assert payload["messages"][1] == {"role": "user", "content": "hey sandy"}
    assert captured["headers"]["Authorization"] == "Bearer secret-ish"
