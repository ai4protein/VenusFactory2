import importlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from agent.harness import Budget, ModelBinding, Profile
from agent.harness.contracts import digest, file_hash
from agent.science_goal import GoalSpec, GoalStore, goal_cycle_prompt, run_goal


def base():
    return Profile(
        "science_expert",
        "langgraph",
        ModelBinding("test", "model", "https://example.org/v1"),
        Budget(wall_seconds=2, model_calls=4, tool_calls=2),
    )


@pytest.mark.asyncio
async def test_three_conditions_share_budget_and_criteria_but_preserve_interaction_semantics(tmp_path):
    outcomes = []
    contexts = {}
    for condition in ["science_goal", "persistent_agent_ablation", "scripted_agent_turns"]:
        profile = base() if condition == "science_goal" else replace(base(), mode="science_agent")
        store = GoalStore.create(
            tmp_path / condition,
            GoalSpec(
                "Develop then reuse", ("accepted", "reused"), max_cycles=3, model_calls=6, tool_calls=4, wall_seconds=10
            ),
            profile,
            condition=condition,
        )
        prompts = []

        async def cycle(index, p, state, context, prompts=prompts, store=store):
            prompts.append(context)
            (store.root / f"proof_{index}").write_text("actual result")
            return {
                "status": "completed",
                "summary": "checkpoint",
                "usage": {"model_calls": 2, "tool_calls": 1, "wall_seconds": 0.01},
                "progress": {"count": index + 1},
            }

        def evaluate(state, store=store):
            checks = {}
            for i, key in enumerate(["accepted", "reused"]):
                path = store.root / f"proof_{i}"
                checks[key] = {
                    "passed": path.exists(),
                    "evidence": [{"path": path.name, "sha256": file_hash(path)}] if path.exists() else [],
                }
            return {"criteria": checks}

        result = await run_goal(store, cycle, evaluate)
        outcomes.append(
            (result["status"], result["usage"]["model_calls"], result["usage"]["tool_calls"], len(result["cycles"]))
        )
        contexts[condition] = prompts
    assert outcomes == [("completed", 4, 2, 2)] * 3
    for key in contexts:
        assert '"count": 1' in contexts[key][1]
        assert "Use PI" in contexts[key][0] if key == "science_goal" else "Use PI" not in contexts[key][0]
    assert "Current conversational user turn" in contexts["scripted_agent_turns"][1]
    assert "Persistent Science goal" in contexts["science_goal"][1]


def test_experimental_condition_requires_explicit_matching_admission(tmp_path):
    agent = replace(base(), mode="science_agent")
    with pytest.raises(ValueError):
        GoalStore.create(tmp_path / "implicit", GoalSpec("Goal", ("done",)), agent)
    for condition, profile in [("invented", base()), ("science_goal", agent), ("persistent_agent_ablation", base())]:
        with pytest.raises(ValueError):
            GoalStore.create(tmp_path / condition, GoalSpec("Goal", ("done",)), profile, condition=condition)
    store = GoalStore.create(
        tmp_path / "valid", GoalSpec("Goal", ("done",)), agent, condition="persistent_agent_ablation"
    )
    data = store.read()
    data["definition"]["condition"] = "science_goal"
    (store.root / "goal.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match="changed"):
        store.read()


def test_old_science_definition_keeps_completion_contract(tmp_path):
    store = GoalStore.create(tmp_path / "legacy", GoalSpec("Old explicit goal", ("done",)), base())
    state = store.read()
    state["definition"].pop("condition")
    state["definition_sha256"] = digest(state["definition"])
    (store.root / "goal.json").write_text(json.dumps(state))
    assert "Persistent Science goal" in goal_cycle_prompt(store.read())


def test_allocation_schedule_is_complete_and_rotates_positions(monkeypatch):
    root = Path(__file__).resolve().parents[2]
    monkeypatch.syspath_prepend(str(root / "scripts/rsi"))
    runner = importlib.import_module("run_goal_comparison")
    rows = runner.schedule(runner.CASES, 5)
    assert len(rows) == len({r["id"] for r in rows}) == 30
    for case in runner.CASES:
        for repeat in range(5):
            assert {r["condition"] for r in rows if r["case"] == case and r["repeat"] == repeat} == set(
                runner.CONDITIONS
            )
    first = [r["condition"] for r in rows if r["case"] == runner.CASES[0] and r["repeat"] == 0]
    second = [r["condition"] for r in rows if r["case"] == runner.CASES[0] and r["repeat"] == 1]
    assert first != second


def test_prepare_builds_all_paired_allocations_before_any_execution(tmp_path, monkeypatch):
    from types import SimpleNamespace

    root = Path(__file__).resolve().parents[2]
    monkeypatch.syspath_prepend(str(root / "scripts/rsi"))
    runner = importlib.import_module("run_goal_comparison")
    domain = importlib.import_module("run_science_goal")
    scope = {"protein": "fixture", "assay": "fixture assay", "target": "value", "unit": "test", "sequence_length": 1}
    rows = [[{"id": str(i), "sequence": aa, "target": float(i)}] for i, aa in enumerate(["A", "C", "D"])]
    monkeypatch.setattr(
        domain, "prepare_case", lambda case, cache: (scope, *rows, {"source": "fixture"}, "fixed test partitions")
    )
    monkeypatch.setattr(
        runner, "builtin_registry", lambda: SimpleNamespace(preflight=lambda profile, runtimes: {"ready": True})
    )
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps(base().public()))
    out = tmp_path / "campaign"
    runner.create(
        SimpleNamespace(
            profile=profile, output=out, cases=["alde_parpgb"], repeats=1, concurrency=1, source_cache=tmp_path
        )
    )
    protocol = runner.verify_protocol(out)
    assert len(protocol["design"]["schedule"]) == 3
    assert json.loads((out / "report.json").read_text())["status"] == "prepared"
    hashes = []
    for row in protocol["design"]["schedule"]:
        arm = out / "arms" / row["id"]
        state = GoalStore(arm).read()
        assert not state["cycles"] and state["usage"]["model_calls"] == 0
        assert state["definition"]["profile"]["mode"] == runner.MODES[row["condition"]]
        hashes.append(json.loads((arm / "protocol.json").read_text())["data_hashes"])
    assert hashes == [hashes[0]] * 3
