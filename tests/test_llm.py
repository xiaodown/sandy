from sandy.llm import (
    BouncerResponse,
    _coerce_bouncer_tool_selection,
    _infer_steam_browse_category,
    _looks_like_direct_image_ask,
)


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


def test_looks_like_direct_image_ask_requires_bot_name_and_picture_language():
    context = "\n".join(
        [
            "[20s ago] [alice] random chatter",
            "[just now] [alice] hey sandy, can you tell me what you think of this picture?",
        ]
    )

    assert _looks_like_direct_image_ask(context, bot_name="Sandy") is True
    assert _looks_like_direct_image_ask(context, bot_name="OtherBot") is False


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

    coerced = _coerce_bouncer_tool_selection(context, result, bot_name="Sandy")

    assert coerced.should_respond is True
    assert "attached image or picture" in coerced.reason
