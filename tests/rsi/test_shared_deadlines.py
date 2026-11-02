"""Offline timeout fixtures: no model endpoint and no autonomous-model evidence."""

import asyncio
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from test_science_goal import assessment, create, outcome
from test_shared_model_program import task_pair

from agent.harness import Budget, ModelBinding, Profile
from agent.rsi.engine.storage import atomic_json
from agent.science_goal import GoalSpec, run_goal

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/rsi"))
import run_shared_science_goal as shared


@pytest.mark.asyncio
async def test_shared_host_timeout_saves_a_complete_failed_outcome(tmp_path, monkeypatch):
    profile = Profile(
        "science_expert",
        "langgraph",
        ModelBinding("test", "no_api", "https://example.org/v1"),
        Budget(wall_seconds=4, request_seconds=2, model_calls=8, tool_calls=8),
    )
    goal = GoalSpec(
        "Retain a checkpoint after a bounded host timeout",
        shared.CRITERIA,
        max_cycles=1,
        wall_seconds=20,
        model_calls=8,
        tool_calls=8,
    )
    store = shared.prepare(tmp_path / "goal", task_pair(tmp_path), profile, goal)
    driver = shared.SharedScientistGoal(store, None)
    observed = []

    async def timed_host(bounded, task, runtimes, *, credential):
        observed.append(bounded)
        assert credential is None
        try:
            async with asyncio.timeout(bounded.budget.wall_seconds):
                await asyncio.sleep(20)
        except TimeoutError:
            pass
        result = {
            "status": "timeout",
            "host_wall_seconds": bounded.budget.wall_seconds,
            "audit": {"requests": {"model_calls": 1}, "events": {"tool_calls": 0}},
        }
        atomic_json(task.output / "run.json", result)
        return result

    monkeypatch.setattr(shared, "execute", timed_host)
    monkeypatch.setattr(shared, "audit_events", lambda *a: {"passed": False, "reason": "timed_out_fixture"})
    state = await run_goal(store, driver.cycle, driver.evaluate)
    saved = shared.read(store.root / "episodes/cycle_000/outcome.json")
    assert saved["status"] == "failed" and state["cycles"][0]["phase"] == "develop"
    assert saved["usage"]["model_calls"] == state["usage"]["model_calls"] == 1
    assert saved["progress"] == {"targets": {}} and saved["target_outcomes"] == {}
    assert state["status"] == "budget_exhausted" and state["inflight"] is None
    assert 0 < observed[0].budget.wall_seconds < profile.budget.wall_seconds
    assert observed[0].model == profile.model and observed[0].harness == profile.harness


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [TimeoutError, ValueError])
async def test_uncheckpointed_failure_stops_shared_policy_without_retry(tmp_path, error):
    store = create(tmp_path, max_cycles=4)
    calls = []

    async def broken(index, *args):
        calls.append(index)
        raise error("do not retain provider response bodies")

    state = await run_goal(store, broken, lambda s: assessment(store, s), stop_on_uncheckpointed_failure=True)
    assert state["status"] == "interrupted" and calls == [0]
    assert state["usage"]["model_calls"] == 3 and state["usage"]["tool_calls"] == 2
    assert state["inflight"] is None and "usage" not in state["cycles"][0]
    assert state["cycles"][0]["error_type"] == error.__name__
    assert state["events"][-1] == {"event": "uncheckpointed_cycle_failure", "index": 0, "error_type": error.__name__}
    assert "provider response" not in (store.root / "goal.json").read_text()


@pytest.mark.asyncio
async def test_checkpointed_failure_can_be_repaired_on_next_cycle(tmp_path):
    store = create(tmp_path, max_cycles=3)
    calls = []

    async def cycle(index, *args):
        calls.append(index)
        if index == 0:
            return {**outcome(), "status": "failed"}
        for name in ["accepted", "reused"]:
            (store.root / (name + ".json")).write_text("{}")
        return outcome()

    state = await run_goal(store, cycle, lambda s: assessment(store, s), stop_on_uncheckpointed_failure=True)
    assert state["status"] == "completed" and calls == [0, 1]
    assert state["usage"]["model_calls"] == 2 and state["cycles"][0]["status"] == "failed"


@pytest.mark.parametrize("value", [1, None, "yes", []])
@pytest.mark.asyncio
async def test_failure_policy_requires_boolean(tmp_path, value):
    store = create(tmp_path)
    with pytest.raises(ValueError, match="checkpoint failure policy"):
        await run_goal(store, None, None, stop_on_uncheckpointed_failure=value)


@pytest.mark.parametrize("remaining,expected", [(100, 90), (50, 45), (1000, 108)])
def test_finalization_time_is_reserved_inside_the_existing_budget(remaining, expected):
    profile = Profile(
        "science_expert",
        "langgraph",
        ModelBinding("test", "no_api", "https://example.org/v1"),
        Budget(wall_seconds=120, request_seconds=100, model_calls=8, tool_calls=8),
    )
    bounded = shared.execution_profile(profile, remaining)
    assert bounded.budget.wall_seconds == pytest.approx(expected)
    assert bounded.budget.request_seconds == min(100, expected)
    assert bounded.budget.model_calls == 8 and bounded.budget.tool_calls == 8
    assert replace(bounded, budget=profile.budget) == profile


def test_finalization_reserve_has_a_fixed_upper_bound():
    profile = Profile(
        "science_expert",
        "langgraph",
        ModelBinding("test", "no_api", "https://example.org/v1"),
        Budget(wall_seconds=900),
    )
    assert shared.execution_profile(profile, 900).budget.wall_seconds == 870


@pytest.mark.parametrize("remaining", [0, -1, float("inf"), float("nan")])
def test_exhausted_or_invalid_preparation_budget_does_not_call_a_model(remaining):
    profile = Profile("science_expert", "langgraph", ModelBinding("test", "no_api", "https://example.org/v1"))
    with pytest.raises((ValueError, TimeoutError)):
        shared.execution_profile(profile, remaining)
