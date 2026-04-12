from sandy.maintenance import main
from sandy.registry import Registry


def test_set_voice_admin_cli_round_trip(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("DB_DIR", str(tmp_path / "prod"))
    monkeypatch.setenv("TEST_DB_DIR", str(tmp_path / "test"))

    rc = main([
        "--test",
        "set-voice-admin",
        "--user-id", "123",
        "--server-id", "456",
        "--enable",
    ])
    assert rc == 0
    monkeypatch.setenv("DB_DIR", str(tmp_path / "test"))
    assert Registry().is_voice_admin(user_id=123, server_id=456) is True

    rc = main([
        "--test",
        "set-voice-admin",
        "--user-id", "123",
        "--server-id", "456",
        "--disable",
    ])
    assert rc == 0
    assert Registry().is_voice_admin(user_id=123, server_id=456) is False


def test_lookup_registry_cli_finds_fuzzy_matches(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.setenv("DB_DIR", str(tmp_path / "prod"))
    monkeypatch.setenv("TEST_DB_DIR", str(tmp_path / "test"))
    monkeypatch.setenv("DB_DIR", str(tmp_path / "test"))
    registry = Registry()
    registry.set_voice_admin(user_id=123, server_id=456, is_admin=True)
    with registry._get_conn() as conn:
        conn.execute("UPDATE users SET user_name = ? WHERE user_id = ?", ("xiaodown", 123))
        conn.execute("UPDATE servers SET server_name = ? WHERE server_id = ?", ("Snack Bandits Co.", 456))
        conn.execute(
            "UPDATE user_nicknames SET nickname = ? WHERE user_id = ? AND server_id = ?",
            ("Will i am creator of SANDY", 123, 456),
        )
        conn.commit()

    rc = main([
        "--test",
        "lookup-registry",
        "--user", "xiao",
        "--server", "snack",
    ])
    assert rc == 0
    out = capsys.readouterr().out
    assert "user_id=123" in out
    assert "server_id=456" in out
    assert "voice_admin=1" in out
