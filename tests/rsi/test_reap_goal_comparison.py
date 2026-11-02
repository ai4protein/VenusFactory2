"""Goal comparison keeps failures, selects without future labels and enforces ordering."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_generated_features import CODE, ROWS, SCOPE

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/rsi"))
from scripts.rsi.cases import run_reap_goal_comparison as comparison
from scripts.rsi.cases.run_reap_generated_matrix import policy

from agent.harness import Budget, ModelBinding, Profile
from agent.harness.contracts import file_hash
from agent.harness.host import write_json
from agent.rsi.engine.model_lifecycle import ModelLifecycle
from agent.rsi.engine.model_plugins import get_model_plugin
from agent.science_goal import GoalSpec, GoalStore


def test_balanced_goal_allocations_and_invalid_seeds():
    rows = comparison.schedule(comparison.CASES, [4, 5, 6])
    assert len(rows) == len({r["id"] for r in rows}) == 18
    for family in comparison.CASES:
        for seed in [4, 5, 6]:
            assert {r["condition"] for r in rows if r["case"] == family and r["seed"] == seed} == set(
                comparison.CONDITIONS
            )
    for seeds in [[4, 4, 5], [1, 2, 3], [4, 5], [4, 5, True]]:
        with pytest.raises(ValueError):
            comparison.schedule(comparison.CASES, seeds)


def candidate(root, index, constant):
    folder = root / "episodes" / f"cycle_{index:03d}"
    folder.mkdir(parents=True)
    training, validation = ROWS[:60], ROWS[60:80]
    controller = ModelLifecycle(folder / "controller", policy(SCOPE, training, validation))
    source = (
        CODE
        if constant is None
        else CODE.replace(
            "return (np.asarray(features) @ np.asarray(model['weights'])).tolist()",
            f"return [{constant!r} for row in features]",
        )
    )
    model = get_model_plugin(comparison.PLUGIN).fit(training, SCOPE, {"source_code": source, "seed": 19})
    write_json(folder / "candidate.json", model)
    mid = controller.submit(folder / "candidate.json", session_id=f"fixture-{index}", mode="science_agent")
    assessment = controller.assess(mid, training=training, validation=validation)
    write_json(folder / "acceptance.json", assessment)
    return {"index": index}, controller, mid


def test_development_selection_retains_rejected_source_and_accepted_incumbent(tmp_path):
    first, _, _ = candidate(tmp_path, 0, 1000.0)
    second, controller, mid = candidate(tmp_path, 1, 100.0)
    state = {"cycles": [first, second], "progress": {}}
    store = SimpleNamespace(root=tmp_path)
    selected = comparison.select_model(store, state)
    assert selected["model_id"] == mid and selected["cycle"] == 1
    assert selected["selection_reason"] == "best_development_submission" and selected["accepted"] is False
    write_json(tmp_path / "future_metrics.json", {"spearman": 1.0, "preferred_cycle": 0})
    assert comparison.select_model(store, state) == selected
    third, controller, mid = candidate(tmp_path, 2, None)
    state["cycles"].append(third)
    selected = comparison.select_model(store, state)
    assert selected["accepted"] and selected["selection_reason"] == "first_independently_accepted"
    state["progress"]["accepted_model"] = {"model_id": mid, "controller": str(controller.root.relative_to(tmp_path))}
    assert comparison.select_model(store, state)["selection_reason"] == "goal_incumbent"
    current = controller._state()
    current["published"] = []
    write_json(controller.root / "state.json", current)
    with pytest.raises(ValueError, match="publication"):
        comparison.select_model(store, state)


def goals(root):
    allocations = comparison.schedule(comparison.CASES, [4, 5, 6])
    profile = Profile(
        "science_agent",
        "langgraph",
        ModelBinding("fixture", "fixture", "https://example.org/v1"),
        Budget(wall_seconds=60, model_calls=4, tool_calls=4),
    )
    for row in allocations:
        store = GoalStore.create(
            root / "arms" / row["id"],
            GoalSpec(
                "fixture",
                ("accepted_model", "fresh_session_reuse"),
                max_cycles=1,
                model_calls=4,
                tool_calls=4,
                wall_seconds=60,
            ),
            profile,
            condition="scripted_agent_turns",
        )
        store.mutate(lambda s: s.update(status="budget_exhausted"))
        write_json(
            store.root / "comparison_result.json",
            {
                "status": "budget_exhausted",
                "implementation_unchanged": True,
                "selected_model": None,
                "external_scripted_turns": 0,
                "internal_scripted_gates": 0,
                "human_interventions": 0,
            },
        )
    return allocations


def test_all_failed_allocations_remain_in_future_denominator(tmp_path):
    rows = goals(tmp_path)
    comparison.freeze_goal_barrier(tmp_path, rows)
    result = comparison.score_all(tmp_path, rows)
    assert len(result["rows"]) == len(result["paired_contrasts"]) == 18
    assert all(
        r["ranking_utility"] == -1 and r["spearman"] is None and r["utility_is_failure_floor"] for r in result["rows"]
    )
    assert all(r["ranking_utility_difference"] == 0 for r in result["paired_contrasts"])


@pytest.mark.parametrize("mutation", ["nonterminal", "prescored", "duplicated"])
def test_terminal_barrier_rejects_invalid_allocations(tmp_path, mutation):
    rows = goals(tmp_path)
    folder = tmp_path / "arms" / rows[0]["id"]
    if mutation == "nonterminal":
        GoalStore(folder).mutate(lambda s: s.update(status="active"))
    elif mutation == "prescored":
        write_json(folder / "selected_future_metrics.json", {})
    else:
        rows[-1] = rows[0]
    with pytest.raises(ValueError):
        comparison.freeze_goal_barrier(tmp_path, rows)
    assert not (tmp_path / "development_barrier.json").exists()


def test_scoring_rejects_goal_mutation_after_barrier(tmp_path):
    rows = goals(tmp_path)
    comparison.freeze_goal_barrier(tmp_path, rows)
    GoalStore(tmp_path / "arms" / rows[-1]["id"]).mutate(lambda s: s.update(status="active"))
    with pytest.raises(ValueError, match="changed after"):
        comparison.score_all(tmp_path, rows)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status,actual_start,allowed",
    [("running", None, False), ("running", "different-process", False), ("campaign_completed", None, True)],
)
async def test_dependency_never_restarts_dead_or_reused_process(tmp_path, monkeypatch, status, actual_start, allowed):
    dep = tmp_path / "previous"
    dep.mkdir()
    write_json(dep / "protocol.json", {})
    write_json(dep / "report.json", {"status": status})
    protocol = {
        "design": {
            "start_after": str(dep),
            "dependency_protocol_sha256": file_hash(dep / "protocol.json"),
            "dependency_pid": 17,
            "dependency_process_start": "original-process",
        }
    }
    monkeypatch.setattr(comparison, "process_start", lambda pid: actual_start)
    if allowed:
        await comparison.wait_for_dependency(tmp_path, protocol)
        assert comparison.read(tmp_path / "dependency_receipt.json")["finished_before_this_campaign"]
    else:
        with pytest.raises(ValueError, match="no longer alive"):
            await comparison.wait_for_dependency(tmp_path, protocol)
        assert comparison.read(tmp_path / "report.json")["status"] == "dependency_not_completed"


def test_observed_failed_requests_are_counted(tmp_path):
    folder = tmp_path / "episodes/cycle_000/execution"
    folder.mkdir(parents=True)
    (folder / "model_requests.jsonl").write_text(
        "\n".join(
            json.dumps(r)
            for r in [{"event": "request"}, {"event": "error"}, {"event": "request"}, {"event": "response"}]
        )
    )
    (folder / "events.jsonl").write_text(
        "\n".join(json.dumps(r) for r in [{"kind": "tool_call_start"}, {"kind": "tool_call_error"}])
    )
    assert comparison.measured_calls(tmp_path) == {"observed_model_calls": 2, "observed_tool_calls": 1}
