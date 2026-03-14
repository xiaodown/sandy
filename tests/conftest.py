import os
from pathlib import Path


_TEST_DB_DIR = Path(__file__).resolve().parent / ".testdata"

os.environ.setdefault("DISCORD_API_KEY", "test-token")
os.environ.setdefault("DB_DIR", str(_TEST_DB_DIR))
os.environ.setdefault("TEST_DB_DIR", str(_TEST_DB_DIR))
os.environ.setdefault("PREWARM_MODEL", "False")
