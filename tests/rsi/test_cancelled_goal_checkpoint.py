import asyncio

import pytest
from test_science_goal import assessment, create, outcome

from agent.science_goal import run_goal


def cancelled(*, calls=1, status="failed", marker=True):
    return {
        "status": status,
        "summary": "Cancelled fixture; no models accepted",
        "execution_cancelled": marker,
        "usage": {"model_calls": calls, "tool_calls": 1, "wall_seconds": 0.0},
    }


@pytest.mark.asyncio
async def test_timeout_keeps_completed_failure_checkpoint_then_continues(tmp_path):
    store = create(tmp_path, wall_seconds=2)

    async def cycle(index, *args):
        if index == 0:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                return cancelled()
        (store.root / "accepted.json").write_text("{}")
        (store.root / "reused.json").write_text("{}")
        return outcome()

    state = await run_goal(
        store, cycle, lambda s: assessment(store, s), cycle_wall_seconds=0.02, stop_on_uncheckpointed_failure=True
    )
    assert state["status"] == "completed" and len(state["cycles"]) == 2
    assert state["cycles"][0]["status"] == "failed" and state["cycles"][0]["execution_cancelled"]
    assert state["usage"]["model_calls"] == 2 and state["usage"]["wall_seconds"] > 0.02
    assert any(
        e["event"] == "cancelled_cycle_checkpoint_saved" and e["trigger_error_type"] == "TimeoutError"
        for e in state["events"]
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("status,marker", [("completed", True), ("failed", False)])
async def test_timeout_cannot_recover_an_unmarked_or_successful_result(tmp_path, status, marker):
    store = create(tmp_path)

    async def cycle(*args):
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            return cancelled(status=status, marker=marker)

    state = await run_goal(
        store, cycle, lambda s: assessment(store, s), cycle_wall_seconds=0.02, stop_on_uncheckpointed_failure=True
    )
    assert state["status"] == "interrupted" and len(state["cycles"]) == 1
    assert state["usage"]["model_calls"] == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("overrun", ["model_calls", "total_wall"])
async def test_cancelled_checkpoint_never_grants_extra_total_budget(tmp_path, overrun):
    store = create(tmp_path, wall_seconds=0.04 if overrun == "total_wall" else 2)

    async def cycle(*args):
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            if overrun == "total_wall":
                await asyncio.sleep(0.04)
            return cancelled(calls=4 if overrun == "model_calls" else 1)

    state = await run_goal(
        store, cycle, lambda s: assessment(store, s), cycle_wall_seconds=0.02, stop_on_uncheckpointed_failure=True
    )
    assert state["status"] == "budget_exhausted" and len(state["cycles"]) == 1
    if overrun == "model_calls":
        assert state["usage"]["model_calls"] == 4
    else:
        assert state["usage"]["wall_seconds"] > 0.04


@pytest.mark.asyncio
async def test_pause_with_a_known_failure_checkpoint_stays_paused(tmp_path):
    store = create(tmp_path)
    started = asyncio.Event()

    async def cycle(*args):
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            return cancelled()

    running = asyncio.create_task(
        run_goal(
            store, cycle, lambda s: assessment(store, s), pause_poll_seconds=0.005, stop_on_uncheckpointed_failure=True
        )
    )
    await started.wait()
    store.pause()
    state = await running
    assert state["status"] == "paused" and state["usage"]["model_calls"] == 1
    assert len(state["cycles"]) == 1 and state["inflight"] is None
