from __future__ import annotations

from pathlib import Path

from sandy import paths


def test_resolve_runtime_path_anchors_relative_paths_to_repo_root(monkeypatch) -> None:
    monkeypatch.chdir(paths.project_root() / "sandy")

    resolved = paths.resolve_runtime_path("data/prod/")

    assert resolved == paths.project_root() / "data" / "prod"


def test_resolve_db_dir_uses_repo_root_for_relative_env_paths(monkeypatch) -> None:
    monkeypatch.chdir(paths.project_root() / "sandy")
    monkeypatch.setenv("DB_DIR", "data/prod/")
    monkeypatch.setenv("TEST_DB_DIR", "data/test/")

    assert paths.resolve_db_dir(test_mode=False) == paths.project_root() / "data" / "prod"
    assert paths.resolve_db_dir(test_mode=True) == paths.project_root() / "data" / "test"


def test_resolve_runtime_path_preserves_absolute_paths(tmp_path: Path) -> None:
    assert paths.resolve_runtime_path(tmp_path) == tmp_path
