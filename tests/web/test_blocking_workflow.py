import asyncio
import threading

from starlette.concurrency import run_in_threadpool

from backend.web_v2.blocking_workflow import MAX_CONCURRENT_WORKFLOWS, offload_workflow


def test_busy_workflows_leave_metadata_threads_available_and_cancel_waiters():
    occupied, release = threading.Event(), threading.Event()
    lock = threading.Lock()
    active = set()
    started = []

    @offload_workflow
    def workload(identity: int):
        with lock:
            active.add(identity)
            started.append(identity)
            if len(active) == MAX_CONCURRENT_WORKFLOWS:
                occupied.set()
        try:
            assert release.wait(3), "Fixture timed out"
            return identity
        finally:
            with lock:
                active.remove(identity)

    async def exercise():
        jobs = [asyncio.create_task(workload(i)) for i in range(MAX_CONCURRENT_WORKFLOWS + 1)]
        try:
            assert await asyncio.to_thread(occupied.wait, 2)
            # Metadata uses a different limiter and must remain available even
            # while every long-workflow slot is occupied.
            assert await asyncio.wait_for(run_in_threadpool(lambda: "metadata ready"), 0.3) == "metadata ready"
            assert len(started) == MAX_CONCURRENT_WORKFLOWS
            jobs[-1].cancel()
            await asyncio.gather(jobs[-1], return_exceptions=True)
        finally:
            release.set()
            results = await asyncio.gather(*jobs, return_exceptions=True)
        assert results[:-1] == list(range(MAX_CONCURRENT_WORKFLOWS))
        assert isinstance(results[-1], asyncio.CancelledError)
        assert not active and len(started) == MAX_CONCURRENT_WORKFLOWS

    asyncio.run(exercise())
    # A subsequent app/event loop must not reuse a limiter bound to a closed loop.
    assert asyncio.run(workload(99)) == 99
