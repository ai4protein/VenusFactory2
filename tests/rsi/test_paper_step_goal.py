"""Persistent control-flow tests: synthetic model records, real isolated code."""

import asyncio
import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest
from test_paper_step_tools import CODE, simulated_session
from test_paper_step_tools import lifecycle as lifecycle

from agent.harness import Budget, ModelBinding, Profile
from agent.rsi import paper_step_goal as goal
from agent.rsi.paper_step_sessions import prepare_phase, read
from agent.rsi.storage import atomic_json
from agent.science_goal import GoalSpec, run_goal


@pytest.fixture
def system(lifecycle, tmp_path):
    definition = deepcopy(lifecycle.definition)
    definition["cases"]["reuse"].append({"id": "reuse_second", "inputs": [2, 5, 12], "expected": {"sum": 19}})
    profile = Profile(
        "science_agent",
        "kimi-code",
        ModelBinding("test_fixture", "simulated", "https://example.org/v1"),
        Budget(tool_calls=8),
    )
    spec = GoalSpec(
        "Implement and independently verify numerical paper step, then reuse",
        goal.CRITERIA,
        max_cycles=6,
        model_calls=96,
        tool_calls=32,
        wall_seconds=7200,
    )
    store = goal.create_step_goal(tmp_path / "goal", definition, profile, spec, condition="persistent_agent_ablation")
    driver = goal.PaperStepGoal(store, "unused-test-credential", {})
    return store, driver, profile


def execute_simulation(calls, *, corrupt=None, codes=None):
    async def execute(folder, profile, runtimes, credential):
        calls.append(folder)
        declaration = read(folder / "phase.json")
        task = simulated_session(
            folder,
            phase=declaration["phase"],
            capability_id=declaration["capability_id"],
            session_id="simulated-" + folder.name,
            profile=profile,
            code=(codes or [CODE])[min(len(calls) - 1, len(codes or [CODE]) - 1)],
        )
        if corrupt:
            corrupt(task)
        return read(task.output / "run.json")

    return execute


@pytest.mark.asyncio
async def test_goal_requires_every_fresh_input_after_real_host_worker_registration(system, monkeypatch):
    store, driver, profile = system
    calls = []
    original = execute_simulation(calls)

    async def execute(*args, **kwargs):
        if calls:
            before = driver.evaluate(store.read())["criteria"]
            assert before["accepted_step_registered"]["passed"]
            assert not before["fresh_inputs_reused"]["passed"]
        return await original(*args, **kwargs)

    monkeypatch.setattr(goal, "execute_phase", execute)
    state = await run_goal(
        store, driver.cycle, driver.evaluate, cycle_wall_seconds=driver.protocol["cycle_wall_seconds"]
    )
    assert state["status"] == "completed", state
    assert len(calls) == 3 and len(state["cycles"]) == 3
    assert state["usage"]["model_calls"] == 3 and state["usage"]["tool_calls"] == 9
    assert all(c["passed"] for c in state["assessment"]["criteria"].values())
    assert set(state["progress"]["reuse"]) == {"reuse", "reuse_second"}
    for folder in calls:
        assert read(folder / "host_worker_exit.json")["returncode"] == 0
        assert read(folder / "host_acceptance.json")["source_provenance"]["provider"] == "test_fixture"
    assert not driver.lifecycle.capability(state["progress"]["capability_id"])["agent_authorship_verified"]
    assert goal.PaperStepGoal(store, None, {}).evaluate(state) == state["assessment"]


@pytest.mark.asyncio
@pytest.mark.parametrize("missing_trace", [False, True])
async def test_access_failure_stops_and_missing_usage_keeps_reservation(system, monkeypatch, missing_trace):
    store, driver, profile = system
    calls = []

    async def execute(folder, *args, **kwargs):
        calls.append(folder)
        out = folder / "runtime"
        out.mkdir()
        result = {"status": "runtime_error", "audit": {"passed": False}}
        atomic_json(out / "run.json", result)
        if not missing_trace:
            (out / "events.jsonl").write_text("")
        (out / "model_requests.jsonl").write_text(
            json.dumps({"event": "request", "call": 1})
            + "\n"
            + json.dumps({"event": "response", "call": 1, "status": 403, "completed": False})
            + "\n"
        )
        return result

    monkeypatch.setattr(goal, "execute_phase", execute)
    state = await run_goal(
        store, driver.cycle, driver.evaluate, cycle_wall_seconds=driver.protocol["cycle_wall_seconds"]
    )
    assert state["status"] == "waiting_for_input" and len(calls) == 1
    assert state["usage"]["model_calls"] == (24 if missing_trace else 1)
    assert state["usage"]["tool_calls"] == (8 if missing_trace else 0)
    state = await run_goal(
        store, driver.cycle, driver.evaluate, cycle_wall_seconds=driver.protocol["cycle_wall_seconds"]
    )
    assert len(calls) == 1
    if missing_trace:
        store.resume()
        state = await run_goal(
            store, driver.cycle, driver.evaluate, cycle_wall_seconds=driver.protocol["cycle_wall_seconds"]
        )
        assert state["status"] == "waiting_for_input" and len(calls) == 1
        assert state["cycles"][-1]["usage"]["model_calls"] == 0


@pytest.mark.asyncio
async def test_completed_source_with_incomplete_session_is_held_without_repeating(system, monkeypatch):
    store, driver, profile = system
    calls = []

    def corrupt(task):
        value = read(task.output / "run.json")
        value["status"] = "runtime_error"
        value["audit"]["passed"] = False
        atomic_json(task.output / "run.json", value)

    monkeypatch.setattr(goal, "execute_phase", execute_simulation(calls, corrupt=corrupt))
    state = await run_goal(
        store, driver.cycle, driver.evaluate, cycle_wall_seconds=driver.protocol["cycle_wall_seconds"]
    )
    assert state["status"] == "waiting_for_input" and state["progress"]["hold"]
    assert not state["progress"].get("capability_id")
    store.resume()
    state = await run_goal(
        store, driver.cycle, driver.evaluate, cycle_wall_seconds=driver.protocol["cycle_wall_seconds"]
    )
    assert len(calls) == 1 and state["status"] == "waiting_for_input"
    assert state["usage"]["model_calls"] == 1 and state["usage"]["tool_calls"] == 3


@pytest.mark.asyncio
async def test_orphaned_source_checkpoint_stops_before_any_model_call(system, monkeypatch):
    store, driver, profile = system
    folder = prepare_phase(driver.lifecycle, store.root / "episodes/cycle_009", "develop")
    simulated_session(folder, phase="develop", profile=profile)
    calls = []
    monkeypatch.setattr(goal, "execute_phase", execute_simulation(calls))
    state = await run_goal(
        store, driver.cycle, driver.evaluate, cycle_wall_seconds=driver.protocol["cycle_wall_seconds"]
    )
    assert state["status"] == "waiting_for_input" and not calls
    assert state["usage"]["model_calls"] == 0 and state["progress"]["hold"]


@pytest.mark.asyncio
async def test_rejected_candidates_are_retained_and_ceiling_prevents_new_model_calls(system, monkeypatch):
    store, driver, profile = system
    calls = []
    wrong = "def reproduce(inputs):\n    return {'sum':3}\n"
    monkeypatch.setattr(goal, "execute_phase", execute_simulation(calls, codes=[wrong]))
    state = await run_goal(
        store, driver.cycle, driver.evaluate, cycle_wall_seconds=driver.protocol["cycle_wall_seconds"]
    )
    assert state["status"] == "waiting_for_input" and len(calls) == 2
    assert [c["status"] for c in state["cycles"]] == [
        "candidate_rejected",
        "candidate_rejected",
        "candidate_budget_exhausted",
    ]
    assert not state["progress"].get("capability_id")
    for folder in calls:
        assert not read(folder / "host_acceptance.json")["assessment"]["passed"]
    assert len(list((driver.lifecycle.root / "attempts").glob("*/reservation.json"))) == 2


@pytest.mark.asyncio
async def test_failed_host_worker_retains_candidate_and_does_not_recompute(system, monkeypatch):
    store, driver, profile = system
    calls = []
    monkeypatch.setattr(goal, "execute_phase", execute_simulation(calls))

    async def fail(*args, **kwargs):
        raise RuntimeError("fixture worker failure")

    monkeypatch.setattr(goal, "host_acceptance_worker", fail)
    state = await run_goal(
        store, driver.cycle, driver.evaluate, cycle_wall_seconds=driver.protocol["cycle_wall_seconds"]
    )
    assert state["status"] == "waiting_for_input" and state["progress"]["hold"]
    store.resume()
    state = await run_goal(
        store, driver.cycle, driver.evaluate, cycle_wall_seconds=driver.protocol["cycle_wall_seconds"]
    )
    assert len(calls) == 1 and state["status"] == "waiting_for_input"


@pytest.mark.parametrize("change", ["protocol", "definition", "reference"])
def test_changed_admission_is_rejected_before_execution(system, change):
    store, driver, profile = system
    if change == "protocol":
        value = read(store.root / "protocol.json")
        value["implementation"] = {}
        atomic_json(store.root / "protocol.json", value)
    elif change == "definition":
        value = read(store.root / "step_definition.json")
        value["tolerance"]["atol"] = 0.01
        atomic_json(store.root / "step_definition.json", value)
    else:
        Path(next(iter(driver.definition["references"]))).write_text("changed reference")
    with pytest.raises(ValueError):
        goal.PaperStepGoal(store, None, {})


@pytest.mark.asyncio
async def test_cancelling_host_acceptance_reaps_process_and_retains_termination(system, monkeypatch, tmp_path):
    store, driver, profile = system
    folder = tmp_path / "worker"
    folder.mkdir()
    original = asyncio.create_subprocess_exec
    processes = []

    async def spawn(*args, **kwargs):
        assert set(kwargs["env"]) == {"PATH", "LANG", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"}
        process = await original(sys.executable, "-c", "import time; time.sleep(60)", **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    task = asyncio.create_task(goal.host_acceptance_worker(store, folder, "develop"))
    for _ in range(500):
        if (folder / "host_worker.json").exists():
            break
        await asyncio.sleep(0.01)
    assert processes
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert processes[0].returncode == -9
    assert read(folder / "host_worker_termination.json") == {"returncode": -9, "interrupted": True}
