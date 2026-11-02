"""Matched policy tests cover total cost, rejected evidence and deferred scoring."""

import copy
import json
import sys
from pathlib import Path

import pytest
from test_improver_policy import history, revised
from test_reap_science_goal import setup_goal

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/rsi"))
import run_reap_goal_comparison as goals
import run_reap_improver_comparison as paired
from run_reap_improver_goal import PolicyReapGoal, policy_action
from run_reap_science_goal import ReapModelGoal

from agent.harness import Budget, ModelBinding, Profile
from agent.harness.contracts import digest, file_hash
from agent.harness.events import audit_requests, audit_trace
from agent.harness.host import write_json
from agent.rsi.improver_policy import BASE_POLICY, candidate, identity, revision_packet
from agent.science_goal import GoalSpec, GoalStore

CAP = {"model_calls": 8, "tool_calls": 4, "wall_seconds": 420}


def test_allocation_matches_batch_budget_without_sixfold_revision_charge():
    rows = paired.schedule(list(goals.CASES), [7, 8, 9], CAP)
    assert len(rows) == len({r["id"] for r in rows}) == 18
    for key in CAP:
        totals = {c: sum(r["goal_budget"][key] for r in rows if r["condition"] == c) for c in paired.CONDITIONS}
        assert totals["fixed_policy"] == totals["revised_policy"]
        assert totals["fixed_policy_total_budget"] == pytest.approx(totals["revised_policy"] + CAP[key])
    for case in goals.CASES:
        for seed in [7, 8, 9]:
            block = [r for r in rows if r["case"] == case and r["seed"] == seed]
            assert {r["condition"] for r in block} == set(paired.CONDITIONS)
            assert len({r["goal_budget"]["max_cycles"] for r in block}) == 1
    assert len({r["condition"] for r in rows[::3]}) == 3


@pytest.mark.parametrize("seeds", [[7, 7, 9], [4, 5, 6], [7, 8], [7, 8, True]])
def test_no_silent_partition_substitution(seeds):
    with pytest.raises(ValueError):
        paired.schedule(list(goals.CASES), seeds, CAP)


def test_history_identical_across_policy_conditions_and_tamper_rejected(tmp_path):
    payload = history()
    for i, policy in enumerate([BASE_POLICY, revised()]):
        store = setup_goal(tmp_path / str(i))
        path = store.root / "improver_policy.json"
        write_json(path, policy)
        write_json(store.root / "shared_development_history.json", payload)
        protocol = json.loads((store.root / "protocol.json").read_text())
        action = policy_action(ReapModelGoal(store, "fixture").development_action(), policy, payload)
        protocol["improver"] = {"policy_id": identity(policy), "file_sha256": file_hash(path),
                                "development_action_sha256": digest(action), "shared_history_sha256": identity(payload)}
        write_json(store.root / "protocol.json", protocol)
        plugin = PolicyReapGoal(store, "fixture")
        assert json.dumps(payload, sort_keys=True) in plugin.development_action()
        assert plugin.shared_history == payload
        altered = copy.deepcopy(payload)
        altered[0]["error_type"] = "TamperedFailure"
        write_json(store.root / "shared_development_history.json", altered)
        with pytest.raises(ValueError, match="history changed"):
            PolicyReapGoal(store, "fixture")
    contaminated = copy.deepcopy(payload)
    contaminated[0]["future_spearman"] = 1.0
    with pytest.raises(ValueError, match="Unexpected development history"):
        policy_action("fixture task", BASE_POLICY, contaminated)


def revision_fixture(root, monkeypatch):
    packet = revision_packet(BASE_POLICY, history())
    profile = Profile("science_agent", "langgraph", ModelBinding("fixture", "fixture", "https://example.org/v1"),
                      Budget(wall_seconds=420, model_calls=8, tool_calls=4))
    protocol = {"profile": profile.public()}
    proposed = candidate(revised(), "Fixture rationale about improving complexity selection.", packet)
    write_json(root / "protocol.json", protocol)
    (root / "workspace").mkdir()
    write_json(root / "workspace/improver_candidate.json", proposed)
    execution = root / "execution"
    execution.mkdir()
    events = [{"kind": "tool_call_start", "tool_call_id": "revision-fixture",
               "tool_name": "mcp__venusfactory__submit_improver_policy",
               "args": {"policy_json": json.dumps(proposed["policy"]), "rationale": proposed["rationale"]}},
              {"kind": "tool_result", "tool_call_id": "revision-fixture", "is_error": False, "output": "fixture"}]
    (execution / "events.jsonl").write_text("\n".join(json.dumps(e) for e in events) + "\n")
    (execution / "model_requests.jsonl").write_text(
        json.dumps({"event": "request", "call": 1}) + "\n" +
        json.dumps({"event": "response", "call": 1, "status": 200, "completed": True}) + "\n")
    audit = {"passed": True, "requests": audit_requests(execution / "model_requests.jsonl"),
             "events": audit_trace(execution / "events.jsonl", paired.REVISION_TOOLS, 4)}
    report = {"status": "candidate_submitted", "protocol_sha256": digest(protocol), "promoted": False,
              "recursive_effectiveness_verified": False, "candidate_sha256": identity(proposed),
              "policy_id": proposed["policy_id"], "parent_id": proposed["parent_id"],
              "run": {"status": "completed", "host_wall_seconds": 2.5, "audit": audit},
              "scientific_audit": {"passed": True}}
    write_json(root / "report.json", report)
    monkeypatch.setattr(paired, "revision_context", lambda _: (protocol, packet))
    return proposed, report, events


def test_revision_requires_physical_records_and_exact_tool_submitted_candidate(tmp_path, monkeypatch):
    proposed, report, events = revision_fixture(tmp_path, monkeypatch)
    actual, receipt = paired.revision_evidence(tmp_path)
    assert actual == proposed and receipt["observed_model_calls"] == receipt["observed_tool_calls"] == 1
    assert receipt["charged_once_per_campaign"]
    events[0]["args"]["rationale"] = "A different model-submitted explanation must not match this candidate."
    (tmp_path / "execution/events.jsonl").write_text("\n".join(json.dumps(e) for e in events) + "\n")
    with pytest.raises(ValueError, match="recorded model tool call"):
        paired.revision_evidence(tmp_path)
    report["status"] = "prepared"
    write_json(tmp_path / "report.json", report)
    with pytest.raises(ValueError, match="No submitted policy"):
        paired.revision_evidence(tmp_path)


def test_missing_physical_request_cannot_be_replaced_by_success_flag(tmp_path, monkeypatch):
    revision_fixture(tmp_path, monkeypatch)
    (tmp_path / "execution/model_requests.jsonl").write_text("")
    with pytest.raises(ValueError, match="execution audit failed"):
        paired.revision_evidence(tmp_path)


def test_failed_allocations_remain_in_contrasts_and_revision_cost_counted_once():
    rows = [{**r, "complete_objective": r["condition"] != "revised_policy",
             "ranking_utility": -1.0 if r["condition"] == "revised_policy" else 0.5,
             "observed_model_calls": 3, "observed_tool_calls": 2, "charged_usage": {"wall_seconds": 10}}
            for r in paired.schedule(list(goals.CASES), [7, 8, 9], CAP)]
    comparisons = paired.contrasts(rows)
    assert len(comparisons) == 18
    assert all(r["ranking_utility_difference"] == -1.5 for r in comparisons if r["left"] == "revised_policy")
    costs = paired.cost_summary(rows, {"observed_model_calls": 5, "observed_tool_calls": 2, "host_wall_seconds": 20})
    evolved = next(c for c in costs if c["condition"] == "revised_policy")
    assert evolved["complete_objectives"] == 0 and evolved["total_observed_model_calls"] == 23
    assert sum(c["revision_model_calls"] for c in costs) == 5
    with pytest.raises(ValueError, match="all 18"):
        paired.contrasts(rows[:-1])
    with pytest.raises(ValueError, match="All allocated"):
        paired.cost_summary(rows[:-1], {})


def test_real_global_barrier_retains_eighteen_no_model_failures(tmp_path):
    rows = paired.schedule(list(goals.CASES), [7, 8, 9], CAP)
    profile = Profile("science_expert", "langgraph", ModelBinding("fixture", "fixture", "https://example.org/v1"),
                      Budget(wall_seconds=60, model_calls=4, tool_calls=4))
    for row in rows:
        store = GoalStore.create(tmp_path / "arms" / row["id"], GoalSpec("fixture", ("accepted_model", "fresh_session_reuse")), profile)
        store.mutate(lambda s: s.update(status="budget_exhausted"))
        write_json(store.root / "comparison_result.json", {"status": "budget_exhausted", "implementation_unchanged": True,
                   "selected_model": None, "external_scripted_turns": 0, "internal_scripted_gates": 0, "human_interventions": 0})
    pending = GoalStore(tmp_path / "arms" / rows[-1]["id"])
    pending.mutate(lambda s: s.update(status="active"))
    with pytest.raises(ValueError, match="terminal"):
        goals.freeze_goal_barrier(tmp_path, rows)
    assert not (tmp_path / "development_barrier.json").exists()
    pending.mutate(lambda s: s.update(status="budget_exhausted"))
    goals.freeze_goal_barrier(tmp_path, rows)
    result = goals.score_all(tmp_path, rows, contrasts=paired.contrasts)
    assert len(result["rows"]) == len(result["paired_contrasts"]) == 18
    assert all(r["utility_is_failure_floor"] and r["spearman"] is None and r["ranking_utility"] == -1.0 for r in result["rows"])


def test_missing_provider_usage_is_unknown_and_revision_tokens_are_separate(tmp_path):
    path = tmp_path / "requests.jsonl"
    entries = [{"event": "request", "call": 1},
               {"event": "response", "call": 1, "usage": {"prompt_tokens": 20, "completion_tokens": 10}},
               {"event": "request", "call": 2}, {"event": "response", "call": 2}]
    path.write_text("\n".join(json.dumps(r) for r in entries))
    usage = paired.token_usage([path])
    assert usage["prompt_tokens"] == 20 and usage["completion_tokens"] == 10
    assert usage["responses"] == 2 and usage["responses_with_usage"] == 1 and not usage["complete"]
    assert not paired.token_usage([tmp_path / "missing.jsonl"])["complete"]
