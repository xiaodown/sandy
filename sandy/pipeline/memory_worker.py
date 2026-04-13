"""Serialize deferred memory work behind a small in-process queue."""

import asyncio

import discord

from ..logconf import get_logger
from ..runtime_state import RuntimeState

logger = get_logger("sandy.bot")


class MemoryWorker:
    """Serialize deferred memory work behind a small in-process queue."""

    _SENTINEL = object()

    def __init__(self, handler, *, runtime_state: RuntimeState | None = None) -> None:
        self._handler = handler
        self._runtime_state = runtime_state
        self._queue: asyncio.Queue[object] = asyncio.Queue()
        self._closed = False

    async def run(self) -> None:
        logger.info("Memory worker started")
        while True:
            item = await self._queue.get()
            try:
                if item is self._SENTINEL:
                    logger.info("Memory worker stopping")
                    return

                message, image_descriptions = item
                if self._runtime_state is not None:
                    self._runtime_state.memory_processing_started(
                        message_id=getattr(message, "id", None),
                    )
                try:
                    await self._handler(message, image_descriptions=image_descriptions)
                except Exception:
                    logger.exception("Memory worker handler failed for message %s", getattr(message, "id", "?"))
                finally:
                    if self._runtime_state is not None:
                        self._runtime_state.memory_processing_finished(
                            message_id=getattr(message, "id", None),
                        )
            finally:
                self._queue.task_done()

    async def enqueue(
        self,
        message: discord.Message,
        image_descriptions: list[str] | None = None,
    ) -> None:
        if self._closed:
            raise RuntimeError("Memory worker is closed")
        await self._queue.put((message, image_descriptions))
        if self._runtime_state is not None:
            self._runtime_state.memory_enqueued()

    async def shutdown(self) -> None:
        if self._closed:
            return

        self._closed = True
        await self._queue.join()
        await self._queue.put(self._SENTINEL)
