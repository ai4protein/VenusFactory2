"""Provider outages retain actual costs and wait without burning more episodes."""

import json

import pytest

from agent.harness import Budget, ModelBinding, Profile
from agent.rsi.provider_access import access_wait
from agent.science_goal import GoalSpec, GoalStore, run_goal


def trace(root, status, *, completed=False, recovered=False, unpaired=False):
    execution = root / "episodes/cycle_000/execution"
    execution.mkdir(parents=True)
    (execution / "run.json").write_text(json.dumps({"status": "completed" if completed else "runtime_error"}))
    rows = [] if unpaired else [{"event": "request", "call": 0}]
    rows += [{"event": "response", "call": 0, "status": status, "completed": status == 200}]
    if recovered:
        rows += [{"event": "request", "call": 1}, {"event": "response", "call": 1, "status": 200, "completed": True}]
    (execution / "model_requests.jsonl").write_text("\n".join(json.dumps(x) for x in rows) + "\n")
    return execution


@pytest.mark.parametrize("status", [401, 402, 403])
def test_access_denial_is_bound_to_host_records(tmp_path, status):
    execution = trace(tmp_path, status)
    got = access_wait(execution, tmp_path)
    assert got["needs_input"] is True
    assert got["dependency_failure"]["http_status"] == status
    assert got["dependency_failure"]["automatic_retry"] is False
    assert got["dependency_failure"]["model_fallback"] is False
    assert len(got["dependency_failure"]["request_sha256"]) == 64
    assert "usage" not in got and "status" not in got


@pytest.mark.parametrize(
    "settings",
    [
        {"status": 200},
        {"status": 500},
        {"status": 429},
        {"status": 403, "completed": True},
        {"status": 403, "recovered": True},
        {"status": 403, "unpaired": True},
    ],
)
def test_success_recovery_or_unpaired_error_does_not_trigger_access_wait(tmp_path, settings):
    assert access_wait(trace(tmp_path, **settings), tmp_path) == {}


def test_missing_trace_and_model_prose_are_not_provider_denial(tmp_path):
    assert access_wait(tmp_path, tmp_path) == {}
    execution = trace(tmp_path, 200)
    path = execution / "model_requests.jsonl"
    with path.open("a") as f:
        f.write(json.dumps({"event": "assistant", "text": "HTTP 403", "status": 403}) + "\n")
    assert access_wait(execution, tmp_path) == {}


@pytest.mark.asyncio
async def test_wait_retains_one_failure_and_cost_then_resume_continues(tmp_path):
    profile = Profile(
        "science_expert",
        "langgraph",
        ModelBinding("fixture", "fixture", "https://example.org/v1"),
        Budget(wall_seconds=2, model_calls=3, tool_calls=2),
    )
    store = GoalStore.create(
        tmp_path / "goal",
        GoalSpec("Synthetic outage recovery", ("accepted",), max_cycles=4, model_calls=12, tool_calls=8),
        profile,
    )
    execution = trace(store.root, 403)
    calls = []

    async def cycle(index, *args):
        calls.append(index)
        return {
            "status": "failed",
            "phase": "develop",
            "usage": {"model_calls": 1, "tool_calls": 0, "wall_seconds": 0.0},
            **access_wait(execution, store.root),
        }

    def evaluate(state):
        return {"criteria": {"accepted": {"passed": False, "evidence": []}}}

    state = await run_goal(store, cycle, evaluate)
    assert state["status"] == "waiting_for_input" and calls == [0]
    assert state["usage"]["model_calls"] == 1 and len(state["cycles"]) == 1
    assert state["cycles"][0]["status"] == "failed"
    before = (store.root / "cycle_000.json").read_bytes()
    await run_goal(store, cycle, evaluate)
    assert calls == [0]
    store.resume()
    # Resume permits a new episode while retaining the first failure and cost.
    state = await run_goal(store, cycle, evaluate)
    assert calls == [0, 1] and state["usage"]["model_calls"] == 2
    assert (store.root / "cycle_000.json").read_bytes() == before
