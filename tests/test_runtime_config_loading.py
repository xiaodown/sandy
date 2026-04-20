from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

from sandy.config import SandyConfig
from sandy.llm import _default_llm_config
from sandy.voice.manager import VoiceManager


def test_sandy_config_from_env_reads_current_environment(monkeypatch) -> None:
    monkeypatch.setenv("BRAIN_MODEL", "brain-from-env")
    monkeypatch.setenv("VISION_MODEL", "vision-from-env")

    cfg = SandyConfig.from_env()

    assert cfg.llm.brain_model == "brain-from-env"
    assert cfg.llm.vision_model == "vision-from-env"


def test_default_llm_config_reads_current_environment(monkeypatch) -> None:
    monkeypatch.setenv("BRAIN_MODEL", "brain-now")
    monkeypatch.setenv("BOUNCER_MODEL", "bounce-now")
    monkeypatch.setenv("TAGGER_MODEL", "tag-now")
    monkeypatch.setenv("SUMMARIZER_MODEL", "sum-now")
    monkeypatch.setenv("VISION_MODEL", "vision-now")
    monkeypatch.setenv("VISION_ROUTER_MODEL", "router-now")
    monkeypatch.setenv("OLLAMA_KEEP_ALIVE", "15m")

    cfg = _default_llm_config()

    assert cfg.brain_model == "brain-now"
    assert cfg.bouncer_model == "bounce-now"
    assert cfg.tagger_model == "tag-now"
    assert cfg.summarizer_model == "sum-now"
    assert cfg.vision_model == "vision-now"
    assert cfg.vision_router_model == "router-now"
    assert cfg.keep_alive == "15m"

def test_voice_manager_fallback_uses_runtime_config(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeTranscriber:
        def __init__(self, *, model_name: str, device: str, compute_type: str, language: str | None) -> None:
            captured["transcriber"] = {
                "model_name": model_name,
                "device": device,
                "compute_type": compute_type,
                "language": language,
            }

    class FakeTtsClient:
        def __init__(self, config) -> None:
            captured["tts"] = config

    monkeypatch.setattr("sandy.voice.manager.FasterWhisperTranscriber", FakeTranscriber)
    monkeypatch.setattr("sandy.voice.manager.TtsServiceClient", FakeTtsClient)
    monkeypatch.setenv("VOICE_STT_MODEL", "tiny.en")
    monkeypatch.setenv("VOICE_STT_DEVICE", "cuda:1")
    monkeypatch.setenv("VOICE_STT_COMPUTE_TYPE", "int8_float16")
    monkeypatch.setenv("VOICE_STT_LANGUAGE", "ja")
    monkeypatch.setenv("VOICE_TTS_SERVICE_URL", "http://tts.test:9999")
    monkeypatch.setenv("VOICE_TTS_SERVICE_TIMEOUT_SECONDS", "33")
    monkeypatch.setenv("VOICE_TTS_INSTRUCT", "dry")
    monkeypatch.setenv("VOICE_TTS_LANGUAGE", "Japanese")

    VoiceManager(
        registry=SimpleNamespace(is_voice_admin=lambda **_: True),
        runtime_state=SimpleNamespace(),
        llm=SimpleNamespace(),
        vector_memory=SimpleNamespace(),
    )

    assert captured["transcriber"] == {
        "model_name": "tiny.en",
        "device": "cuda:1",
        "compute_type": "int8_float16",
        "language": "ja",
    }
    tts_config = captured["tts"]
    assert tts_config.base_url == "http://tts.test:9999"
    assert tts_config.timeout_seconds == 33.0
    assert tts_config.default_instruct == "dry"
    assert tts_config.default_language == "Japanese"


def test_tts_service_config_reads_current_environment(monkeypatch) -> None:
    module_path = Path(__file__).resolve().parents[1] / "tts_service" / "tts_service" / "app.py"
    module_name = "test_tts_service_app"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None
    assert spec.loader is not None

    fake_tts = SimpleNamespace(FasterQwen3TTS=object)
    monkeypatch.setitem(sys.modules, "faster_qwen3_tts", fake_tts)

    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)

    monkeypatch.setenv("TTS_SERVICE_MODEL", "tts-model")
    monkeypatch.setenv("TTS_SERVICE_LANGUAGE", "French")
    monkeypatch.setenv("TTS_SERVICE_CLONE_XVECTOR_ONLY", "false")
    monkeypatch.setenv("TTS_SERVICE_DEVICE", "cuda:1")
    monkeypatch.setenv("TTS_SERVICE_DTYPE", "float16")
    monkeypatch.setenv("TTS_SERVICE_MAX_SEQ_LEN", "4096")
    monkeypatch.setenv("TTS_SERVICE_DO_SAMPLE", "true")

    cfg = module.ServiceConfig.from_env()

    assert cfg.model_name == "tts-model"
    assert cfg.language == "French"
    assert cfg.clone_xvec_only is False
    assert cfg.device == "cuda:1"
    assert cfg.dtype == "float16"
    assert cfg.max_seq_len == 4096
    assert cfg.do_sample is True
