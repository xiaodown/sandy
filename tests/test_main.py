from __future__ import annotations

import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


def _load_local_main_module() -> types.ModuleType:
    path = Path(__file__).resolve().parents[1] / "sandy" / "__main__.py"
    module = types.ModuleType("sandy_main_local")
    module.__file__ = str(path)
    module.__package__ = "sandy"
    source = path.read_text(encoding="utf-8")
    exec(compile(source, str(path), "exec"), module.__dict__)
    return module


@pytest.mark.asyncio
async def test_main_prewarm_uses_live_pipeline_after_setup(monkeypatch) -> None:
    main_module = _load_local_main_module()
    warm_model = AsyncMock(return_value=True)
    live_pipeline = SimpleNamespace(llm=SimpleNamespace(warm_model=warm_model))
    started_tokens: list[str] = []
    api_service_calls: list[SimpleNamespace] = []

    report = SimpleNamespace(hard_failures=[])
    monkeypatch.setattr(main_module, "collect_health_report", AsyncMock(return_value=report))
    monkeypatch.setattr(main_module, "log_startup_report", lambda *_args, **_kwargs: None)

    fake_config = SimpleNamespace(
        api=SimpleNamespace(enabled=True, host="127.0.0.1", port=8765),
        prewarm_enabled=True,
        prewarm_model_name="test-prewarm-model",
    )
    fake_config_module = types.ModuleType("sandy.config")
    fake_config_module.SandyConfig = SimpleNamespace(
        from_env=lambda *, test_mode: fake_config,
    )
    monkeypatch.setitem(sys.modules, "sandy.config", fake_config_module)

    async def fake_start(token: str) -> None:
        started_tokens.append(token)

    fake_bot_module = types.ModuleType("sandy.bot")
    fake_bot_module.pipeline = None
    fake_bot_module.runtime_state = SimpleNamespace()
    fake_bot_module.DISCORD_API_KEY = "discord-token"
    fake_bot_module.logger = SimpleNamespace(
        info=lambda *args, **kwargs: None,
        warning=lambda *args, **kwargs: None,
        error=lambda *args, **kwargs: None,
    )
    fake_bot_module.bot = SimpleNamespace(start=fake_start, close=AsyncMock())
    fake_bot_module.shutdown_background_work = AsyncMock()

    def fake_setup(_config) -> None:
        fake_bot_module.pipeline = live_pipeline

    fake_bot_module.setup = fake_setup
    monkeypatch.setitem(sys.modules, "sandy.bot", fake_bot_module)

    class FakeApiService:
        def __init__(self, *, pipeline, runtime_state, test_mode) -> None:
            api_service_calls.append(
                SimpleNamespace(
                    pipeline=pipeline,
                    runtime_state=runtime_state,
                    test_mode=test_mode,
                )
            )

    class FakeApiServer:
        def __init__(self, _service, *, host, port) -> None:
            self.address = (host, port)

        def start(self) -> None:
            return None

        def shutdown(self) -> None:
            return None

    fake_api_module = types.ModuleType("sandy.api")
    fake_api_module.ApiService = FakeApiService
    fake_api_module.ApiServer = FakeApiServer
    monkeypatch.setitem(sys.modules, "sandy.api", fake_api_module)

    result = await main_module._main(SimpleNamespace(test=False))

    assert result == 0
    assert fake_bot_module.pipeline is live_pipeline
    assert api_service_calls[0].pipeline is live_pipeline
    warm_model.assert_awaited_once_with("test-prewarm-model")
    assert started_tokens == ["discord-token"]
    fake_bot_module.shutdown_background_work.assert_awaited_once()
    fake_bot_module.bot.close.assert_awaited_once()
