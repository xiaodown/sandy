from __future__ import annotations

import asyncio

import pytest

from sandy.bot import BackgroundTaskSupervisor, MemoryWorker


@pytest.mark.asyncio
async def test_background_supervisor_waits_for_task_completion():
    supervisor = BackgroundTaskSupervisor()
    steps: list[str] = []
    release = asyncio.Event()

    async def job():
        steps.append("started")
        await release.wait()
        steps.append("finished")

    supervisor.create_task(job(), name="test-job")
    await asyncio.sleep(0)
    assert steps == ["started"]

    shutdown_task = asyncio.create_task(supervisor.shutdown())
    await asyncio.sleep(0)
    assert not shutdown_task.done()

    release.set()
    await shutdown_task

    assert steps == ["started", "finished"]
    assert supervisor._tasks == set()


@pytest.mark.asyncio
async def test_memory_worker_processes_queue_then_stops_cleanly():
    calls: list[tuple[int, list[str] | None]] = []

    async def handler(message, image_descriptions=None):
        calls.append((message.id, image_descriptions))

    worker = MemoryWorker(handler)
    run_task = asyncio.create_task(worker.run())

    await worker.enqueue(type("Message", (), {"id": 1})(), image_descriptions=["cat"])
    await worker.enqueue(type("Message", (), {"id": 2})())
    await worker.shutdown()
    await run_task

    assert calls == [(1, ["cat"]), (2, None)]


@pytest.mark.asyncio
async def test_memory_worker_rejects_enqueue_after_shutdown():
    worker = MemoryWorker(lambda *_args, **_kwargs: None)

    await worker.shutdown()

    with pytest.raises(RuntimeError, match="Memory worker is closed"):
        await worker.enqueue(type("Message", (), {"id": 1})())
