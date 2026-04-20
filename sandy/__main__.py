"""Entry point for ``python -m sandy``.

Usage:
    python -m sandy          # prod mode (DB_DIR from .env, primary Discord key)
    python -m sandy --test   # test mode (TEST_DB_DIR from .env, test Discord key)
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
from collections.abc import Sequence

from dotenv import load_dotenv

from .health import collect_health_report, log_startup_report

startup_logger = logging.getLogger("sandy.startup")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sandy — Discord personality bot")
    parser.add_argument(
        "--test", action="store_true",
        help="Run in test mode (TEST_DB_DIR database, DISCORD_API_KEY_TEST token)",
    )
    return parser


def _prepare_runtime(argv: Sequence[str] | None = None) -> argparse.Namespace:
    # Parse args BEFORE importing .bot — bot.py reads DB_DIR and DISCORD_API_KEY
    # at module level, so we need the env vars set first.
    load_dotenv()
    args = _build_parser().parse_args(argv)

    if args.test:
        os.environ["DB_DIR"] = os.getenv("TEST_DB_DIR", "data/test/")
        # Use the dedicated test token if available, otherwise keep whatever
        # DISCORD_API_KEY is already set to (backwards-compatible).
        test_key = os.getenv("DISCORD_API_KEY_TEST")
        if test_key:
            os.environ["DISCORD_API_KEY"] = test_key
    else:
        # Default to prod unless .env already set something else.
        os.environ.setdefault("DB_DIR", "data/prod/")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )
    return args


async def _main(args: argparse.Namespace) -> int:
    mode = "TEST" if args.test else "PROD"
    startup_logger.info("Starting Sandy in %s mode (DB_DIR=%s)", mode, os.environ.get("DB_DIR"))
    report = await collect_health_report(test_mode=args.test)
    log_startup_report(report, startup_logger)
    if report.hard_failures:
        startup_logger.error("Aborting startup due to %d hard failure(s)", len(report.hard_failures))
        return 1

    # Build central config from env vars (already loaded by load_dotenv + arg handling).
    from .config import SandyConfig
    config = SandyConfig.from_env(test_mode=args.test)

    # Import the Discord stack only after startup prerequisites are satisfied.
    from . import bot as bot_module  # noqa: E402

    bot_module.setup(config)

    from .api import ApiServer, ApiService

    api_server: ApiServer | None = None
    try:
        if config.api.enabled:
            api_server = ApiServer(
                ApiService(
                    pipeline=bot_module.pipeline,
                    runtime_state=bot_module.runtime_state,
                    test_mode=args.test,
                ),
                host=config.api.host,
                port=config.api.port,
            )
            api_server.start()
            host, port = api_server.address
            bot_module.logger.info("Observability API listening on http://%s:%d", host, port)

        if config.prewarm_enabled and config.prewarm_model_name:
            bot_module.logger.info("Beginning pre-warming of model %s before Discord connect", config.prewarm_model_name)
            if await bot_module.pipeline.llm.warm_model(config.prewarm_model_name):
                bot_module.logger.info("Pre-warming model %s complete", config.prewarm_model_name)
            else:
                bot_module.logger.warning("Pre-warming of model %s failed", config.prewarm_model_name)
        await bot_module.bot.start(bot_module.DISCORD_API_KEY)
    except Exception as exc:
        bot_module.logger.error("Fatal: %s", exc)
        return 1
    finally:
        if api_server is not None:
            api_server.shutdown()
        await bot_module.shutdown_background_work()
        await bot_module.bot.close()
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _prepare_runtime(argv)
    return asyncio.run(_main(args))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        startup_logger.info("^c caught - stand by, shutting down cleanly...")
