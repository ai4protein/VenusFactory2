"""Reject missing failures, altered continuations and changed scoring barriers."""

import json
import sys
from pathlib import Path

import pytest
from test_reap_goal_comparison import candidate, goals

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/rsi"))
from scripts.rsi.cases import audit_reap_goal_comparison as auditor
from scripts.rsi.cases.run_reap_goal_comparison import freeze_goal_barrier

from agent.harness import Budget, ModelBinding, Profile
from agent.harness.contracts import digest
from agent.harness.host import write_json
from agent.science_goal import GoalSpec, GoalStore, run_goal


def barrier(root):
    rows = goals(root)
    sha = freeze_goal_barrier(root, rows)
    write_json(root / "report.json", {"development_barrier_sha256": sha})
    return rows


def test_audit_keeps_all_eighteen_failed_goals(tmp_path):
    rows = barrier(tmp_path)
    assert auditor.verify_barrier(tmp_path, rows)["count"] == 18


@pytest.mark.parametrize("mutation", ["goal", "result", "denominator", "premature_future"])
def test_changed_barrier_or_outcome_is_rejected(tmp_path, mutation):
    rows = barrier(tmp_path)
    folder = tmp_path / "arms" / rows[0]["id"]
    if mutation == "goal":
        GoalStore(folder).mutate(lambda s: s.update(status="active"))
    elif mutation == "result":
        p = folder / "comparison_result.json"
        record = auditor.read(p)
        record["status"] = "completed"
        write_json(p, record)
    elif mutation == "denominator":
        rows.pop()
    else:
        p = tmp_path / "development_barrier.json"
        record = auditor.read(p)
        record["future_metrics_present"] = True
        write_json(p, record)
    with pytest.raises(ValueError):
        auditor.verify_barrier(tmp_path, rows)


def test_incomplete_campaign_cannot_be_terminal_audited(tmp_path):
    write_json(tmp_path / "report.json", {"status": "waiting_for_dependency"})
    write_json(tmp_path / "protocol.json", {})
    with pytest.raises(ValueError, match="Complete 18"):
        auditor.audit(tmp_path, tmp_path, tmp_path.parent / "audit")


def test_replayed_rejected_selection_ignores_future_outcomes(tmp_path):
    cycles = []
    episodes = []
    for index, constant in enumerate([100.0, 1000.0]):
        cycle, controller, mid = candidate(tmp_path, index, constant)
        cycles.append(cycle)
        assessment = auditor.read(controller.root.parent / "acceptance.json")
        episodes.append(
            {"cycle": index, "training_replayed": True, "accepted": False, "acceptance_metrics": assessment["metrics"]}
        )
    state = {"cycles": cycles, "progress": {}}
    write_json(tmp_path / "selected_future_metrics.json", {"preferred_cycle": 1, "spearman": 1})
    selected = auditor.selection_from_replayed_assessments(tmp_path, state, {"episodes": episodes})
    assert selected["cycle"] == 0 and not selected["accepted"]
    assert selected["selection_reason"] == "best_development_submission"


@pytest.mark.asyncio
async def test_accounting_rejects_fabricated_external_user_turn(tmp_path):
    profile = Profile(
        "science_agent",
        "langgraph",
        ModelBinding("fixture", "fixture", "https://example.org/v1"),
        Budget(wall_seconds=30, model_calls=4, tool_calls=4),
    )
    store = GoalStore.create(
        tmp_path / "goal",
        GoalSpec(
            "fixture",
            ("accepted_model", "fresh_session_reuse"),
            max_cycles=1,
            model_calls=4,
            tool_calls=4,
            wall_seconds=30,
        ),
        profile,
        condition="scripted_agent_turns",
    )

    async def cycle(index, p, state, context):
        (store.root / "turns").mkdir()
        write_json(
            store.root / "turns/turn_000.json",
            {
                "index": index,
                "condition": "scripted_agent_turns",
                "checkpoint": state,
                "context": context,
                "context_sha256": digest(context),
                "external_user_turn": True,
                "human_intervention": False,
                "episode_profile": p.public(),
            },
        )
        execution = store.root / "episodes/cycle_000/execution"
        execution.mkdir(parents=True)
        write_json(execution / "run.json", {"status": "runtime_error", "human_interventions": 0})
        write_json(execution / "profile.lock.json", {"profile": p.public()})
        (execution / "model_requests.jsonl").write_text(json.dumps({"event": "request"}) + "\n")
        (execution / "events.jsonl").write_text(json.dumps({"kind": "tool_call_start"}) + "\n")
        return {
            "status": "failed",
            "progress": {},
            "phase": "develop",
            "usage": {"model_calls": 1, "tool_calls": 1, "wall_seconds": 0},
        }

    def evaluate(state):
        return {"criteria": {k: {"passed": False, "evidence": []} for k in ["accepted_model", "fresh_session_reuse"]}}

    state = await run_goal(store, cycle, evaluate)
    result = {
        "episodes": 1,
        "usage": state["usage"],
        "external_scripted_turns": 1,
        "internal_scripted_gates": 0,
        "human_interventions": 0,
        "runtime_failures": [{"episode": 0, "status": "runtime_error"}],
    }
    row = {"condition": "scripted_agent_turns"}
    assert auditor.turn_accounting(store.root, row, state, result)["observed_model_calls"] == 1
    p = store.root / "turns/turn_000.json"
    turn = auditor.read(p)
    turn["external_user_turn"] = False
    write_json(p, turn)
    with pytest.raises(ValueError, match="User-turn"):
        auditor.turn_accounting(store.root, row, state, result)
