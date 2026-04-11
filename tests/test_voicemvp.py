from types import SimpleNamespace

from sandy.voice.capture import (
    _pcm_bytes_for_milliseconds,
    _slugify_capture_label,
)
from sandy.voice.manager import (
    resolve_target_channel,
)


def _channel(name: str):
    return SimpleNamespace(name=name)


def _guild(*names: str):
    return SimpleNamespace(voice_channels=[_channel(name) for name in names])


def test_resolve_target_channel_prefers_author_voice_when_no_query():
    author_voice = _channel("ops war room")

    result = resolve_target_channel(
        _guild("ops war room", "music"),
        query="",
        author_voice_channel=author_voice,
    )

    assert result is author_voice


def test_resolve_target_channel_matches_exact_name_case_insensitively():
    guild = _guild("General Voice", "Raid Night")

    result = resolve_target_channel(
        guild,
        query="general   voice",
        author_voice_channel=None,
    )

    assert result is guild.voice_channels[0]


def test_resolve_target_channel_prefers_exact_match_over_partial_ambiguity():
    guild = _guild("Gaming", "Gaming 2", "Movies")

    result = resolve_target_channel(
        guild,
        query="gaming",
        author_voice_channel=None,
    )

    assert result is guild.voice_channels[0]


def test_resolve_target_channel_returns_none_for_ambiguous_partial_match():
    guild = _guild("Gaming", "Gaming 2", "Movies")

    result = resolve_target_channel(
        guild,
        query="gam",
        author_voice_channel=None,
    )

    assert result is None


def test_slugify_capture_label_normalizes_spaces_and_case():
    assert _slugify_capture_label("  Xiao Down  ") == "xiao-down"


def test_slugify_capture_label_strips_punctuation():
    assert _slugify_capture_label("vscode changed?! my theme") == "vscode-changed-my-theme"


def test_pcm_bytes_for_milliseconds_matches_48khz_stereo_16bit():
    assert _pcm_bytes_for_milliseconds(250) == 48_000
