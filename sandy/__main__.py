"""Entry point for ``python -m sandy``."""

import asyncio

from .bot import bot, DISCORD_API_KEY, logger


async def _main() -> None:
    try:
        await bot.start(DISCORD_API_KEY)
    except Exception as exc:
        logger.error("Fatal: %s", exc)
    finally:
        await bot.close()


try:
    asyncio.run(_main())
except KeyboardInterrupt:
    logger.info("^c caught - stand by, shutting down cleanly...")
