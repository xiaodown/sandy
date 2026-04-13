import io
import struct
import wave
from types import SimpleNamespace

import pytest

from sandy.voice.capture import (
    _pcm_bytes_for_milliseconds,
    _slugify_capture_label,
    _voice_recv_listener,
)
from sandy.voice.models import (
    resolve_target_channel,
)
from sandy.voice.tts import wav_bytes_to_audio_source


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


def test_voice_recv_listener_degrades_to_noop_when_extension_is_unavailable(monkeypatch):
    import sandy.voice.capture as capture_module

    monkeypatch.setattr(capture_module, "voice_recv", None)

    marker = object()

    @_voice_recv_listener()
    def listener():
        return marker

    assert listener() is marker


# ── wav_bytes_to_audio_source ────────────────────────────────────────────────

def _make_wav(
    *,
    channels: int = 1,
    sample_width: int = 2,
    sample_rate: int = 48_000,
    n_frames: int = 480,
) -> bytes:
    """Generate a minimal valid WAV file with the given params."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        # Write silence (zeros)
        wf.writeframes(b"\x00" * (n_frames * channels * sample_width))
    return buf.getvalue()


def test_wav_stereo_48k_passthrough():
    """Stereo 48 kHz 16-bit should pass through without conversion."""
    src = wav_bytes_to_audio_source(_make_wav(channels=2, sample_rate=48_000))
    data = src.stream.getvalue()
    assert isinstance(data, bytes)
    assert len(data) > 0


def test_wav_mono_converted_to_stereo():
    """Mono input should be converted to stereo (double the samples)."""
    n_frames = 480
    wav = _make_wav(channels=1, sample_rate=48_000, n_frames=n_frames)
    src = wav_bytes_to_audio_source(wav)
    data = src.stream.getvalue()
    # Stereo 16-bit: 2 channels × 2 bytes per sample × n_frames
    assert len(data) == n_frames * 2 * 2


def test_wav_sample_rate_converted_to_48k():
    """Non-48kHz sample rate should be resampled."""
    src = wav_bytes_to_audio_source(_make_wav(channels=2, sample_rate=24_000, n_frames=240))
    data = src.stream.getvalue()
    assert isinstance(data, bytes)
    assert len(data) > 0


def test_wav_mono_24k_converted():
    """Mono 24 kHz should convert to stereo 48 kHz."""
    src = wav_bytes_to_audio_source(_make_wav(channels=1, sample_rate=24_000, n_frames=240))
    data = src.stream.getvalue()
    assert isinstance(data, bytes)
    assert len(data) > 0


def test_wav_8bit_raises():
    """8-bit WAV should raise ValueError."""
    # wave module can't write 8-bit with setsampwidth(1) as 16-bit PCM,
    # but we can craft it manually or use 1-byte sample width
    with pytest.raises(ValueError, match="8-bit"):
        wav_bytes_to_audio_source(_make_wav(sample_width=1))


def test_wav_three_channels_raises():
    """3-channel WAV should raise ValueError."""
    with pytest.raises(ValueError, match="channel count"):
        wav_bytes_to_audio_source(_make_wav(channels=3))
