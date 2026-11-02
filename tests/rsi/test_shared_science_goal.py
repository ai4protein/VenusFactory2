"""Persistent host integration using real MCP/numerics and a synthetic executor.

No model endpoint is called. These fixtures are not autonomous model evidence.
"""

import json
import sys
from pathlib import Path

import pytest
from test_shared_model_program import CODE, client_for, task_pair

from agent.harness import Budget, ModelBinding, Profile
from agent.rsi.protein_model import digest
from agent.rsi.scientist_model_library import validate_model_goal
from agent.rsi.storage import atomic_json, sha256_file
from agent.rsi.tool_contracts import validate_generated_result
from agent.science_goal import GoalSpec, GoalStore, run_goal

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/rsi"))
import run_shared_science_goal as shared


def read(path):
    return json.loads(path.read_text())


def prepared(tmp_path, *, reject_second=False, max_cycles=6, history_policy=None):
    tasks = task_pair(tmp_path)
    for task in tasks:
        task["plugin_id"] = "generated_context.v2"
        task["scope"]["interface_version"] = "generated_context.v2"
        future = Path(task["data"]["future"]["path"])
        rows = read(future)
        for row in rows:
            row["target"] = 1234567.0
        atomic_json(future, rows)
        task["data"]["future"]["sha256"] = sha256_file(future)
    if reject_second:
        path = Path(tasks[1]["data"]["validation"]["path"])
        rows = read(path)
        for row in rows:
            row["target"] *= -1
        atomic_json(path, rows)
        tasks[1]["data"]["validation"]["sha256"] = sha256_file(path)
    profile = Profile(
        "science_expert",
        "langgraph",
        ModelBinding("test", "synthetic_no_api", "https://example.org/v1"),
        Budget(model_calls=8, tool_calls=8, wall_seconds=120, request_seconds=60),
    )
    goal = GoalSpec(
        "Independently qualify and freshly reuse both synthetic models",
        shared.CRITERIA,
        max_cycles=max_cycles,
        model_calls=48,
        tool_calls=48,
        wall_seconds=900,
    )
    options = {} if history_policy is None else {"shared_training_history": history_policy}
    store = shared.prepare(
        tmp_path / "goal", tasks, profile, goal, training_baseline_feedback="group_baselines.v2", **options
    )
    return store, shared.SharedScientistGoal(store, None)


def synthetic_executor(monkeypatch):
    calls = []

    async def execute(profile, task, runtimes, *, credential):
        assert credential is None
        declaration = read(task.capability / "task.json")
        ids = [x["id"] for x in declaration["targets"]]
        calls.append({"phase": task.phase, "targets": ids, "prompt": task.prompt})
        events = []

        async def call(client, name, args):
            identity = str(len(events))
            events.append(
                {
                    "kind": "tool_call_start",
                    "tool_call_id": identity,
                    "tool_name": "mcp__venusfactory__" + name,
                    "args": args,
                }
            )
            result = (await client.call_tool(name, args)).data
            validate_generated_result(name, result, workspace=task.workspace, capability=task.capability)
            events.append(
                {"kind": "tool_result", "tool_call_id": identity, "is_error": False, "output": json.dumps(result)}
            )
            return result

        async with client_for(task.workspace, task.capability) as client:
            await call(client, "inspect_model_code_task", {})
            if task.phase == "develop":
                assert "discover_generated_models" not in task.tools
                trial = await call(
                    client, "evaluate_model_code", {"source_code": CODE, "rationale": "synthetic executor fixture"}
                )
                await call(
                    client,
                    "submit_model_code",
                    {"source_sha256": trial["source_sha256"], "rationale": "independent target fits"},
                )
            else:
                await call(client, "discover_generated_models", {})
                await call(client, "predict_with_generated_model", {"model_id": "all"})
        run = {
            "status": "completed",
            "phase": task.phase,
            "session_id": "synthetic_" + task.output.parent.name,
            "host_wall_seconds": 0.0,
            "audit": {"requests": {"model_calls": 1}, "events": {"tool_calls": len(events) // 2}},
        }
        atomic_json(task.output / "run.json", run)
        (task.output / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
        return run

    monkeypatch.setattr(shared, "execute", execute)
    monkeypatch.setattr(shared, "audit_events", lambda *args: {"passed": True, "synthetic_executor": True})
    return calls


@pytest.mark.asyncio
async def test_persistent_shared_goal_accepts_all_then_reuses_in_new_session(tmp_path, monkeypatch):
    store, driver = prepared(tmp_path)
    calls = synthetic_executor(monkeypatch)
    state = await run_goal(store, driver.cycle, driver.evaluate)
    assert state["status"] == "completed"
    assert [c["phase"] for c in calls] == ["develop", "reuse"]
    assert all(c["targets"] == ["synthetic_0", "synthetic_1"] for c in calls)
    assert state["usage"]["model_calls"] == 2  # One shared executor call per episode, not per protein.
    assert len(state["progress"]["targets"]) == 2
    assert all(c["passed"] for c in driver.evaluate(state)["criteria"].values())
    assert all("1234567" not in c["prompt"] for c in calls)
    assert not list(store.root.glob("future*"))
    for task_id, record in state["progress"]["targets"].items():
        assert record["development_episode"] != record["reuse_episode"]
        assert task_id in record["controller"] and task_id in record["reuse_receipt"]
    # A shared-origin verifier must still reject this synthetic executor at its native proof gate.
    with pytest.raises(ValueError, match="publication auditor|Fixture execution"):
        validate_model_goal(store.root, state["progress"]["targets"]["synthetic_0"]["model_id"])
    resumed = shared.SharedScientistGoal(GoalStore(store.root), None)
    assert resumed.evaluate(state) == driver.evaluate(state)


@pytest.mark.asyncio
async def test_partial_acceptance_reuses_success_then_repairs_remaining_targets(tmp_path, monkeypatch):
    store, driver = prepared(tmp_path, reject_second=True, max_cycles=4)
    calls = synthetic_executor(monkeypatch)
    state = await run_goal(store, driver.cycle, driver.evaluate)
    assert state["status"] == "budget_exhausted"
    assert [(x["phase"], x["targets"]) for x in calls] == [
        ("develop", ["synthetic_0", "synthetic_1"]),
        ("reuse", ["synthetic_0"]),
        ("develop", ["synthetic_1"]),
        ("develop", ["synthetic_1"]),
    ]
    assert state["usage"]["model_calls"] == 4
    assert set(state["progress"]["targets"]) == {"synthetic_0"}
    assert "reuse_receipt" in state["progress"]["targets"]["synthetic_0"]
    assert state["cycles"][0]["target_outcomes"]["synthetic_1"]["status"] == "candidate_rejected"
    history = read(store.root / "episodes/cycle_002/shared_training_history.json")
    assert len(history["attempts"]) == 1 and history["attempts"][0]["source_code"] == CODE
    assert [r["task_id"] for r in history["attempts"][0]["results"]] == ["synthetic_1"]
    assert all("1234567" not in c["prompt"] for c in calls)
    assert not any(c["passed"] for c in state["assessment"]["criteria"].values())
    # Changing source text in a prior tool request cannot inject unexecuted code into the next repair.
    path = store.root / "episodes/cycle_002/execution/events.jsonl"
    events = [json.loads(x) for x in path.read_text().splitlines()]
    for event in events:
        if event.get("tool_name", "").endswith("evaluate_model_code"):
            event["args"]["source_code"] += "\n# unexecuted change\n"
    path.write_text("".join(json.dumps(x) + "\n" for x in events))
    with pytest.raises(ValueError, match="source differs"):
        driver.training_history(state, [driver.by_id["synthetic_1"]])


@pytest.mark.asyncio
async def test_pause_resume_keeps_accepted_models_without_retraining(tmp_path, monkeypatch):
    store, driver = prepared(tmp_path)
    calls = synthetic_executor(monkeypatch)

    async def pause_after_development(*args):
        result = await driver.cycle(*args)
        store.pause()
        return result

    state = await run_goal(store, pause_after_development, driver.evaluate)
    assert state["status"] == "paused" and len(calls) == 1
    assert len(state["progress"]["targets"]) == 2
    reopened = shared.SharedScientistGoal(GoalStore(store.root), None)
    store.resume()
    state = await run_goal(store, reopened.cycle, reopened.evaluate)
    assert state["status"] == "completed" and [c["phase"] for c in calls] == ["develop", "reuse"]


@pytest.mark.parametrize("change", ["cohort", "feedback", "implementation", "target_order", "criterion"])
def test_reopen_rejects_changed_frozen_goal(tmp_path, change):
    store, _ = prepared(tmp_path)
    if change == "cohort":
        path = store.root / "targets/synthetic_0/data/training.json"
        value = read(path)
        value[0]["target"] += 1
        atomic_json(path, value)
    elif change == "criterion":
        state = store.read()
        state["definition"]["goal"]["criteria"] = ["self_reported_success"]
        state["definition_sha256"] = digest(state["definition"])
        atomic_json(store.root / "goal.json", state)
    else:
        path = store.root / "protocol.json"
        value = read(path)
        if change == "feedback":
            value["target_training_baseline_feedback"] = "none"
        elif change == "implementation":
            value["implementation"].pop(next(iter(value["implementation"])))
        else:
            value["targets"].reverse()
        atomic_json(path, value)
    with pytest.raises(ValueError):
        shared.SharedScientistGoal(store, None)
