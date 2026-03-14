"""Entry point for ``python -m sandy``.

Usage:
    python -m sandy          # prod mode (DB_DIR from .env, primary Discord key)
    python -m sandy --test   # test mode (TEST_DB_DIR from .env, test Discord key)
"""

import argparse
import asyncio
import logging
import os

from dotenv import load_dotenv

from .health import collect_health_report, log_startup_report

# Parse args BEFORE importing .bot — bot.py reads DB_DIR and DISCORD_API_KEY
# at module level, so we need the env vars set first.
load_dotenv()

parser = argparse.ArgumentParser(description="Sandy — Discord personality bot")
parser.add_argument(
    "--test", action="store_true",
    help="Run in test mode (TEST_DB_DIR database, DISCORD_API_KEY_TEST token)",
)
args = parser.parse_args()

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
startup_logger = logging.getLogger("sandy.startup")


async def _main() -> int:
    mode = "TEST" if args.test else "PROD"
    startup_logger.info("Starting Sandy in %s mode (DB_DIR=%s)", mode, os.environ.get("DB_DIR"))
    report = await collect_health_report(test_mode=args.test)
    log_startup_report(report, startup_logger)
    if report.hard_failures:
        startup_logger.error("Aborting startup due to %d hard failure(s)", len(report.hard_failures))
        return 1

    # Import the Discord stack only after startup prerequisites are satisfied.
    from .bot import bot, DISCORD_API_KEY, logger, pipeline, shutdown_background_work  # noqa: E402

    try:
        prewarm_enabled = os.getenv("PREWARM_MODEL") == "True"
        prewarm_model_name = os.getenv("PREWARM_MODEL_NAME")
        if prewarm_enabled and prewarm_model_name:
            logger.info("Beginning pre-warming of model %s before Discord connect", prewarm_model_name)
            if await pipeline.llm.warm_model(prewarm_model_name):
                logger.info("Pre-warming model %s complete", prewarm_model_name)
            else:
                logger.warning("Pre-warming of model %s failed", prewarm_model_name)
        await bot.start(DISCORD_API_KEY)
    except Exception as exc:
        logger.error("Fatal: %s", exc)
        return 1
    finally:
        await shutdown_background_work()
        await bot.close()
    return 0


try:
    raise SystemExit(asyncio.run(_main()))
except KeyboardInterrupt:
    startup_logger.info("^c caught - stand by, shutting down cleanly...")
