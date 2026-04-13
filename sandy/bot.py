"""
Sandy's Discord bot glue.

This module owns Discord client lifecycle, event registration, and orderly
shutdown. The actual message pipeline lives in pipeline.py.
"""

import asyncio
import logging
import os
from collections.abc import Awaitable

import discord
from dotenv import load_dotenv

from .logconf import get_logger
from .pipeline import SandyPipeline, build_pipeline
from .runtime_state import RuntimeState

load_dotenv()

logger = get_logger("sandy.bot")

intents = discord.Intents.default()
intents.members = True
intents.message_content = True
bot = discord.Client(intents=intents)
DISCORD_API_KEY = os.getenv("DISCORD_API_KEY")

logging.getLogger("discord").setLevel(logging.WARNING)


class BackgroundTaskSupervisor:
    """Track background tasks so failures are logged and shutdown is orderly."""

    def __init__(self) -> None:
        self._tasks: set[asyncio.Task] = set()

    def create_task(self, awaitable: Awaitable[object], *, name: str) -> asyncio.Task:
        task = asyncio.create_task(awaitable, name=name)
        self._tasks.add(task)
        task.add_done_callback(self._on_done)
        return task

    def _on_done(self, task: asyncio.Task) -> None:
        self._tasks.discard(task)
        try:
            task.result()
        except asyncio.CancelledError:
            logger.info("Background task cancelled: %s", task.get_name())
        except Exception:
            logger.exception("Background task failed: %s", task.get_name())

    async def shutdown(self) -> None:
        if not self._tasks:
            return

        pending = tuple(self._tasks)
        logger.info("Waiting for %d background task(s) to finish", len(pending))
        await asyncio.gather(*pending, return_exceptions=True)


background_tasks = BackgroundTaskSupervisor()
runtime_state = RuntimeState()
pipeline = build_pipeline(background_tasks=background_tasks, runtime_state=runtime_state)


def _refresh_discord_runtime_state() -> None:
    runtime_state.set_discord_connected(True, user_name=bot.user.name if bot.user else None)
    runtime_state.set_discord_servers([guild.name for guild in bot.guilds])


@bot.event
async def on_connect():
    """Event handler for when Discord transport connects or reconnects."""
    _refresh_discord_runtime_state()


@bot.event
async def on_ready():
    """Event handler for when the bot is ready."""
    _refresh_discord_runtime_state()
    await pipeline.on_ready(bot)


@bot.event
async def on_disconnect():
    """Event handler for Discord disconnects."""
    runtime_state.set_discord_connected(False)


@bot.event
async def on_resumed():
    """Event handler for when Discord resumes after a transient disconnect."""
    _refresh_discord_runtime_state()


@bot.event
async def on_message(message: discord.Message):
    """Event handler for incoming messages."""
    if message.guild is None:
        # Bot doesn't respond to DMs
        return
    if await pipeline.handle_control_message(message, bot_user=bot.user):
        return
    await pipeline.handle_message(message, bot_user=bot.user)


@bot.event
async def on_voice_state_update(member, before, after):
    """Event handler for voice roster changes relevant to an active session."""
    pipeline.handle_voice_state_update(member, before, after, bot_user=bot.user)


async def shutdown_background_work() -> None:
    """Flush queued background work before process exit."""
    await pipeline.shutdown()
    await background_tasks.shutdown()
