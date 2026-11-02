"""Real classifier lifecycle tests for adaptive feedback and failure-inclusive scoring."""

import json
import sys
from collections import Counter
from pathlib import Path

import pytest
from test_generated_classifier import CODE, ROWS, SCOPE
from test_nucb_science_goal import setup_goal

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/rsi"))
from scripts.rsi.cases import run_nucb_goal_comparison as comparison
import run_science_goal as base
from scripts.rsi.cases.run_nucb_generated_matrix import policy
from scripts.rsi.cases.run_nucb_science_goal import PLUGIN, NucbModelGoal

from agent.harness import Budget, ModelBinding, Profile
from agent.harness.host import write_json
from agent.rsi.engine.model_lifecycle import ModelLifecycle
from agent.rsi.engine.model_plugins import get_model_plugin
from agent.science_goal import GoalSpec, GoalStore, run_goal

BAD_CODE = "def fit(sequences, targets, seed):\n    return {}\ndef predict(model, sequences):\n    return [[1.0, 0.0] for s in sequences]\n"


def enable_feedback(store, value):
    path = store.root / "protocol.json"
    protocol = json.loads(path.read_text())
    protocol["development_feedback"] = value
    write_json(path, protocol)


def candidate(store, index, code=BAD_CODE):
    folder = store.root / "episodes" / f"cycle_{index:03d}"
    folder.mkdir(parents=True)
    lc = ModelLifecycle(folder / "controller", policy(SCOPE, ROWS[:720], ROWS[720:840]))
    model = get_model_plugin(PLUGIN).fit(ROWS[:720], SCOPE, {"source_code": code, "seed": 19})
    write_json(folder / "candidate.json", model)
    mid = lc.submit(folder / "candidate.json", session_id=f"fixture-{index}", mode="science_agent")
    assessment = lc.assess(mid, training=ROWS[:720], validation=ROWS[720:840])
    write_json(folder / "acceptance.json", assessment)
    return {"index": index, "phase": "develop"}, lc, mid


def test_factorial_schedule_has_all_18_allocations():
    rows = comparison.schedule()
    assert len(rows) == len({r["id"] for r in rows}) == 18
    assert Counter((r["condition"], r["feedback"]) for r in rows) == {
        (c, f): 3 for c in comparison.common.CONDITIONS for f in comparison.FEEDBACK
    }
    assert len({tuple(r["condition"] + "_" + r["feedback"] for r in rows if r["repeat"] == i) for i in range(3)}) == 3


@pytest.mark.parametrize("feedback", ["none", "source_and_metrics"])
@pytest.mark.asyncio
async def test_rejection_revision_reuse_and_exact_feedback_visibility(tmp_path, monkeypatch, feedback):
    store = setup_goal(tmp_path)
    enable_feedback(store, feedback)
    plugin = get_model_plugin(PLUGIN)
    phases = []
    seen = []

    async def execute(profile, task, runtimes, credential):
        phases.append(task.phase)
        session = f"fixture-{len(phases)}"
        if task.phase == "develop":
            if len(phases) == 2:
                assert ("DEVELOPMENT acceptance feedback" in task.prompt) == (feedback == "source_and_metrics")
                assert BAD_CODE not in task.prompt  # JSON-escaped source, not raw multiline code.
                assert ("return [[1.0, 0.0]" in task.prompt) == (feedback == "source_and_metrics")
                seen.append(task.prompt)
            train = json.loads((task.workspace / "training.json").read_text())
            assert train == ROWS[:720]
            model = plugin.fit(train, SCOPE, {"source_code": BAD_CODE if len(phases) == 1 else CODE, "seed": 19})
            write_json(task.workspace / "candidate.json", model)
        else:
            inputs = json.loads((task.workspace / "inputs.json").read_text())
            assert all(set(x) == {"id", "sequence"} for x in inputs)
            lc, mid = goal.lifecycle(store.read())
            write_json(task.workspace / "predictions.json", plugin.predict(lc._model(mid), inputs, SCOPE))
        return {
            "status": "completed",
            "session_id": session,
            "host_wall_seconds": 0.01,
            "audit": {"passed": True, "requests": {"model_calls": 1}, "events": {"tool_calls": 1}},
        }

    monkeypatch.setattr(base, "execute", execute)
    monkeypatch.setattr(base, "audit_events", lambda *args, **kwargs: {"passed": True})
    goal = NucbModelGoal(store, "fixture")
    state = await run_goal(store, goal.cycle, goal.evaluate)
    assert state["status"] == "completed" and phases == ["develop", "develop", "reuse"]
    assert state["cycles"][0]["status"] == "candidate_rejected"
    assert state["cycles"][1]["source_bytes_changed_from_feedback"] == (feedback == "source_and_metrics")
    packet = json.loads((store.root / "episodes/cycle_001/development_feedback.json").read_text())
    if feedback == "source_and_metrics":
        assert len(packet) == 1 and packet[0]["source_code"] == BAD_CODE
        assert "predictions_sha256" not in packet[0]["development_acceptance_metrics"]
        assert not any(k in packet[0] for k in ["model_state", "targets", "future"])
    else:
        assert packet == []
    assert not (store.root / "future_metrics.json").exists()
    assert comparison.publish(store, goal, state) == comparison.publish(store, goal, state)


@pytest.mark.parametrize("tamper", ["assessment", "source"])
def test_feedback_rejects_mutated_assessment_or_registered_source(tmp_path, tamper):
    store = setup_goal(tmp_path)
    enable_feedback(store, "source_and_metrics")
    cycle, lc, mid = candidate(store, 0)
    state = store.read()
    state["cycles"] = [cycle]
    goal = NucbModelGoal(store, "fixture")
    assert len(goal.development_feedback(state)) == 1
    if tamper == "assessment":
        path = store.root / "episodes/cycle_000/acceptance.json"
        d = json.loads(path.read_text())
        d["metrics"]["brier"] = 0
        write_json(path, d)
    else:
        path = lc.root / "artifacts/versions" / mid / "model.json"
        d = json.loads(path.read_text())
        d["recipe"]["source_code"] = CODE
        # Deliberately override read-only permissions only on this temporary test fixture.
        path.parent.chmod(0o755)
        path.chmod(0o644)
        write_json(path, d)
    with pytest.raises(ValueError):
        goal.development_feedback(state)


def test_classifier_selection_uses_g3_even_for_rejected_source(tmp_path):
    store = setup_goal(tmp_path)
    cycle, lc, mid = candidate(store, 0)
    state = store.read()
    state["cycles"] = [cycle]
    chosen = comparison.select_model(store, state)
    assert chosen["model_id"] == mid and not chosen["accepted"] and chosen["development_skill"] < 0
    assert chosen["selection_reason"] == "best_development_submission"


def failed_goals(root):
    rows = comparison.schedule()
    profile = Profile(
        "science_agent",
        "langgraph",
        ModelBinding("fixture", "fixture", "https://example.org/v1"),
        Budget(wall_seconds=60, model_calls=4, tool_calls=4),
    )
    for row in rows:
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
    return rows


def test_all_failed_goals_retained_and_postbarrier_changes_rejected(tmp_path):
    rows = failed_goals(tmp_path)
    comparison.common.freeze_goal_barrier(tmp_path, rows)
    report = comparison.score_all(tmp_path, rows)
    assert len(report["rows"]) == 18 and len(report["paired_contrasts"]) == 21
    assert all(
        r["uses_prior_fallback"] and r["all_allocation_brier_skill"] == 0 and r["raw_metrics"] is None
        for r in report["rows"]
    )
    first = GoalStore(tmp_path / "arms" / rows[0]["id"])
    first.mutate(lambda s: s.update(status="active"))
    with pytest.raises(ValueError, match="after the global"):
        comparison.score_all(tmp_path, rows)


def test_rejected_classifier_negative_future_skill_is_not_clipped(tmp_path):
    rows = failed_goals(tmp_path)
    store = GoalStore(tmp_path / "arms" / rows[0]["id"])
    cycle, lc, mid = candidate(store, 0)
    store.mutate(lambda s: s.update(cycles=[cycle]))
    state = store.read()
    selected = comparison.select_model(store, state)
    path = store.root / "comparison_result.json"
    record = json.loads(path.read_text())
    record["selected_model"] = selected
    write_json(path, record)
    write_json(store.root / "protocol.json", {"scope": SCOPE})
    (store.root / "data").mkdir()
    write_json(store.root / "data/training.json", ROWS[:720])
    write_json(store.root / "data/future.json", ROWS[840:900])
    comparison.common.freeze_goal_barrier(tmp_path, rows)
    report = comparison.score_all(tmp_path, rows)
    arm = report["rows"][0]
    assert arm["raw_metrics"]["rows"] == 60 and arm["all_allocation_brier_skill"] < 0 and not arm["uses_prior_fallback"]
    assert not arm["complete_objective"] and not arm["selected_model"]["accepted"]
