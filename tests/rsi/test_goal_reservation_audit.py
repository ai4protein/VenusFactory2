"""Actual outer cancellations preserve budget without fabricating successful usage."""

import asyncio
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/rsi"))
from scripts.rsi.evaluation.audit_goal_reservations import retained_timeout_reservation
from scripts.rsi.cases.audit_nucb_goal_comparison import verify_revision_receipts
from scripts.rsi.cases.audit_reap_goal_comparison import turn_accounting

from agent.harness import Budget, ModelBinding, Profile
from agent.harness.contracts import digest
from agent.harness.host import write_json
from agent.science_goal import GoalSpec, GoalStore, run_goal


async def cancelled_goal(path, cycles=1):
    profile = Profile(
        "science_agent",
        "langgraph",
        ModelBinding("fixture", "fixture", "https://example.org/v1"),
        Budget(wall_seconds=0.08, model_calls=4, tool_calls=3),
    )
    store = GoalStore.create(
        path,
        GoalSpec(
            "timeout fixture",
            ("accepted_model", "fresh_session_reuse"),
            max_cycles=cycles,
            model_calls=4 * cycles,
            tool_calls=3 * cycles,
            wall_seconds=2,
        ),
        profile,
        condition="persistent_agent_ablation",
    )

    async def cycle(index, bounded, state, context):
        (store.root / "turns").mkdir(exist_ok=True)
        write_json(
            store.root / "turns" / f"turn_{index:03d}.json",
            {
                "index": index,
                "condition": "persistent_agent_ablation",
                "checkpoint": state,
                "context": context,
                "context_sha256": digest(context),
                "external_user_turn": index == 0,
                "human_intervention": False,
                "episode_profile": bounded.public(),
            },
        )
        execution = store.root / "episodes" / f"cycle_{index:03d}" / "execution"
        execution.mkdir(parents=True)
        lock = {"profile": bounded.public()}
        lock["sha256"] = digest(lock)
        write_json(execution / "profile.lock.json", lock)
        write_json(execution / "run.json", {"status": "running", "human_interventions": 0})
        (execution / "model_requests.jsonl").write_text(json.dumps({"event": "request", "call": 0}) + "\n")
        await asyncio.sleep(5)
        raise AssertionError("Controller should cancel this episode")

    def evaluate(state):
        return {
            "criteria": {key: {"passed": False, "evidence": []} for key in ["accepted_model", "fresh_session_reuse"]}
        }

    state = await run_goal(store, cycle, evaluate)
    assert state["status"] == "budget_exhausted" and len(state["cycles"]) == cycles
    result = {
        "episodes": cycles,
        "usage": state["usage"],
        "external_scripted_turns": 1,
        "internal_scripted_gates": 0,
        "human_interventions": 0,
        "runtime_failures": [{"episode": i, "status": "running"} for i in range(cycles)],
    }
    return store, state, result


@pytest.mark.asyncio
@pytest.mark.parametrize("cycles", [1, 2])
async def test_real_timeouts_keep_actual_calls_separate_from_full_reservations(tmp_path, cycles):
    store, state, result = await cancelled_goal(tmp_path / "goal", cycles)
    for cycle in state["cycles"]:
        reservation = retained_timeout_reservation(store.root, state, cycle)
        assert reservation["model_calls"] == 4 and reservation["tool_calls"] == 3
        assert cycle["status"] == "failed" and "usage" not in cycle
    counts = turn_accounting(store.root, {"condition": "persistent_agent_ablation"}, state, result)
    assert counts["observed_model_calls"] == cycles and counts["observed_tool_calls"] == 0
    assert state["usage"]["model_calls"] == 4 * cycles and state["usage"]["tool_calls"] == 3 * cycles
    assert verify_revision_receipts(store.root, state) == 0
    assert not list(store.root.glob("episodes/*/outcome.json"))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mutation",
    [
        "successful_status",
        "other_error",
        "revision_claim",
        "finalized_outcome",
        "reservation",
        "profile_digest",
        "checkpoint_history",
        "actual_overrun",
        "erased_failure",
    ],
)
async def test_timeout_exception_rejects_fabricated_receipts_or_success(tmp_path, mutation):
    store, state, result = await cancelled_goal(tmp_path / "goal")
    cycle = state["cycles"][0]
    turn_path = store.root / "turns/turn_000.json"
    lock_path = store.root / "episodes/cycle_000/execution/profile.lock.json"
    if mutation == "successful_status":
        cycle["status"] = "completed"
    elif mutation == "other_error":
        cycle["error_type"] = "UnknownError"
    elif mutation == "revision_claim":
        cycle["source_bytes_changed_from_feedback"] = True
    elif mutation == "finalized_outcome":
        write_json(store.root / "episodes/cycle_000/outcome.json", {"status": "completed"})
    elif mutation in {"reservation", "checkpoint_history"}:
        value = json.loads(turn_path.read_text())
        if mutation == "reservation":
            value["checkpoint"]["inflight"]["reserved"]["model_calls"] = 1
        else:
            value["checkpoint"]["cycles"] = [{"status": "completed"}]
        write_json(turn_path, value)
    elif mutation == "profile_digest":
        value = json.loads(lock_path.read_text())
        value["sha256"] = "corrupt"
        write_json(lock_path, value)
    elif mutation == "actual_overrun":
        (lock_path.parent / "model_requests.jsonl").write_text(
            "".join(json.dumps({"event": "request", "call": i}) + "\n" for i in range(5))
        )
    else:
        result["runtime_failures"] = []
    with pytest.raises(ValueError):
        turn_accounting(store.root, {"condition": "persistent_agent_ablation"}, state, result)


def test_finalized_cycles_do_not_gain_timeout_exemption(tmp_path):
    cycle = {"index": 0, "status": "failed", "usage": {"model_calls": 1, "tool_calls": 0, "wall_seconds": 2}}
    assert retained_timeout_reservation(tmp_path, {"cycles": [cycle]}, cycle) is None
