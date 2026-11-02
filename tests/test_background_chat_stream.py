import asyncio
import pytest
from backend.web_v2.chat_api.background_stream import BackgroundStreamResponse, _RUNNING


def test_disconnecting_subscriber_keeps_turn_and_lock_until_completion():
    async def exercise():
        lock = asyncio.Lock()
        await lock.acquire()
        proceed = asyncio.Event()
        completed = []
        async def source():
            try:
                yield "first"
                await proceed.wait()
                completed.append("persisted")
                yield "last"
            finally:
                lock.release()
        response = BackgroundStreamResponse(source())
        assert await anext(response.body_iterator) == "first"
        await response.body_iterator.aclose()
        assert lock.locked() and response.producer in _RUNNING
        proceed.set()
        await asyncio.wait_for(response.producer, 1)
        assert completed == ["persisted"] and not lock.locked()
    asyncio.run(exercise())


def test_early_http_disconnect_and_full_buffer_never_block_background_turn():
    async def exercise():
        produced = []
        async def source():
            for i in range(1000):
                produced.append(i)
                yield str(i)
        response = BackgroundStreamResponse(source())
        # Let the bounded queue fill with no subscriber.
        for _ in range(1000):
            if response.queue.full(): break
            await asyncio.sleep(0)
        assert response.queue.full() and response.queue.qsize() == 128
        async def receive(): return {"type": "http.disconnect"}
        async def send(message):
            await asyncio.sleep(10)
        await response({"type":"http", "asgi":{"spec_version":"2.0"}}, receive, send)
        await asyncio.wait_for(response.producer, 1)
        assert len(produced) == 1000
    asyncio.run(exercise())


def test_connected_stream_preserves_frames_and_explicit_cancel_still_finalizes():
    async def exercise():
        stopped = asyncio.Event()
        released = asyncio.Event()
        async def source():
            try:
                yield "start"
                await stopped.wait()
                yield "stopped"
            finally: released.set()
        response = BackgroundStreamResponse(source())
        assert await anext(response.body_iterator) == "start"
        stopped.set()
        assert [x async for x in response.body_iterator] == ["stopped"]
        await response.producer
        assert released.is_set()
    asyncio.run(exercise())


def test_failed_producer_closes_stream_without_leaking_lock():
    async def exercise():
        finalized=[]
        async def source():
            try:
                yield "start"
                raise ValueError("synthetic")
            finally: finalized.append(True)
        response=BackgroundStreamResponse(source())
        frames=[x async for x in response.body_iterator]
        await response.producer
        assert "event: error" in frames[-1] and finalized == [True]
    asyncio.run(exercise())


def test_old_cancellation_is_settled_but_active_turn_keeps_stopping(monkeypatch):
    from backend.web_v2.chat_api import sessions
    async def exercise():
        lock = asyncio.Lock()
        monkeypatch.setattr(sessions, "_SESSION_LOCKS", {"active": lock})
        assert sessions.visible_run_status("old", "stopping") == "stopped"
        await lock.acquire()
        assert sessions.visible_run_status("active", "stopping") == "stopping"
        assert sessions.visible_run_status("active", "started") == "started"
        lock.release()
        assert sessions.visible_run_status("active", "stopping") == "stopped"
    asyncio.run(exercise())
