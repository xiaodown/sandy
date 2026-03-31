from pathlib import Path

from sandy.registry import Registry


def test_registry_voice_admin_round_trip(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DB_DIR", str(tmp_path))
    registry = Registry()

    assert registry.is_voice_admin(user_id=123, server_id=456) is False

    registry.set_voice_admin(user_id=123, server_id=456, is_admin=True)
    assert registry.is_voice_admin(user_id=123, server_id=456) is True

    registry.set_voice_admin(user_id=123, server_id=456, is_admin=False)
    assert registry.is_voice_admin(user_id=123, server_id=456) is False
