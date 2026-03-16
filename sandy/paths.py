from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    """Return the repository root, not the package directory."""
    return Path(__file__).resolve().parent.parent


def resolve_runtime_path(raw_path: str | os.PathLike[str]) -> Path:
    """Resolve relative runtime paths against the repo root.

    Sandy historically treated DB_DIR like "data/prod/" as cwd-relative, which
    meant launching from the package directory silently wrote into sandy/data/.
    Runtime state should not depend on whatever directory the operator happened
    to be standing in.
    """
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    return project_root() / path


def resolve_db_dir(*, test_mode: bool) -> Path:
    env_name = "TEST_DB_DIR" if test_mode else "DB_DIR"
    default = "data/test/" if test_mode else "data/prod/"
    return resolve_runtime_path(os.getenv(env_name, default))
