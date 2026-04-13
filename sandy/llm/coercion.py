"""Deterministic bouncer result coercion and heuristic overrides."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from ..logconf import get_logger

if TYPE_CHECKING:
    from .models import BouncerResponse

logger = get_logger(__name__)

_HISTORY_LINE_RE = re.compile(r"^\[[^\]]+\] \[[^\]]+\] (?P<content>.*)$")

_IMAGE_ASK_PATTERNS: tuple[str, ...] = (
    "this picture",
    "this image",
    "that picture",
    "that image",
    "look at this",
    "look at that",
    "what do you think of this",
    "what do you think of that",
)

_STEAM_CATEGORY_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("specials", ("on sale", "sales", "sale", "discount", "discounts")),
    ("upcoming", ("coming soon", "upcoming")),
    ("new_releases", ("new releases", "new release", "just came out", "fresh release", "fresh releases")),
    ("top_sellers", ("top sellers", "top seller", "best sellers", "best seller", "what's good", "whats good", "what's hot", "whats hot", "selling well")),
)


def _extract_history_messages(context: str) -> list[str]:
    """Return plain message text from Last10-formatted history lines."""
    messages: list[str] = []
    for line in context.splitlines():
        match = _HISTORY_LINE_RE.match(line.strip())
        if match:
            messages.append(match.group("content").strip())
    return messages


def _infer_steam_browse_category(context: str) -> str | None:
    """Infer the Steam storefront category implied by the latest turn."""
    messages = _extract_history_messages(context)
    if not messages:
        return None
    lowered_messages = [message.lower() for message in messages]
    latest = lowered_messages[-1]
    recent_window = lowered_messages[-4:]

    if not any("steam" in message for message in recent_window):
        return None

    for category, keywords in _STEAM_CATEGORY_KEYWORDS:
        if any(keyword in latest for keyword in keywords):
            return category

    # Follow-up turns like "check actual steam" need the most recent
    # storefront category from nearby history rather than a blind default.
    if "steam" in latest or "check actual" in latest or "check again" in latest:
        for message in reversed(lowered_messages[:-1]):
            for category, keywords in _STEAM_CATEGORY_KEYWORDS:
                if any(keyword in message for keyword in keywords):
                    return category

    if "steam" in latest:
        return "top_sellers"
    return None


def _looks_like_direct_image_ask(context: str) -> bool:
    messages = _extract_history_messages(context)
    if not messages:
        return False

    latest = messages[-1].lower()

    if "sandy" not in latest:
        return False
    if not any(pattern in latest for pattern in _IMAGE_ASK_PATTERNS):
        return False
    return "?" in latest or "think" in latest or "look" in latest


def _coerce_bouncer_tool_selection(
    context: str,
    result: "BouncerResponse",
) -> "BouncerResponse":
    """Apply deterministic tool overrides for obvious storefront asks."""
    if (
        not result.should_respond
        and _looks_like_direct_image_ask(context)
    ):
        logger.info("Bouncer image-ask override: forcing should_respond=True")
        result.should_respond = True
        result.reason = (
            "Deterministic override: latest message directly asks "
            "Sandy about an attached image or picture."
        )

    if not result.should_respond:
        return result
    if result.use_tool and result.recommended_tool not in {None, "search_web", "steam_browse"}:
        return result

    category = _infer_steam_browse_category(context)
    if category is None:
        return result

    if result.recommended_tool == "steam_browse":
        if result.tool_parameters is None:
            result.tool_parameters = {"category": category, "limit": 5}
        else:
            result.tool_parameters.setdefault("category", category)
            result.tool_parameters.setdefault("limit", 5)
        return result

    limit = 5
    if isinstance(result.tool_parameters, dict):
        raw_limit = result.tool_parameters.get("limit", result.tool_parameters.get("n_results"))
        if isinstance(raw_limit, int):
            limit = max(1, min(raw_limit, 10))

    logger.info(
        "Bouncer storefront override: forcing steam_browse(%s) instead of %s",
        category,
        result.recommended_tool or "none",
    )
    result.use_tool = True
    result.recommended_tool = "steam_browse"
    result.tool_parameters = {"category": category, "limit": limit}
    return result
