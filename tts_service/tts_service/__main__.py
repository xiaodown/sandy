from __future__ import annotations

import logging
import os

import uvicorn

from .app import create_app


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )
    host = os.getenv("TTS_SERVICE_HOST", "127.0.0.1")
    port = int(os.getenv("TTS_SERVICE_PORT", "8777"))
    uvicorn.run(create_app(), host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
