"""Tamper tests for policy origin, matched total budgets and failure accounting."""

import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/rsi"))
import audit_improver_revision as revision_audit
import audit_reap_improver_comparison as audit
import run_reap_improver_comparison as runner
from test_improver_policy import history, revised

from agent.harness import Budget, ModelBinding, Profile
from agent.harness.contracts import digest, file_hash
from agent.harness.host import write_json
from agent.rsi.improver_policy import BASE_POLICY, candidate, identity, revision_packet


@pytest.fixture
def design():
    profile = Profile(
        "science_expert",
        "langgraph",
        ModelBinding("fixture", "fixture", "https://example.org/v1"),
        Budget(wall_seconds=900, model_calls=24, tool_calls=12),
    ).public()
    return {
        "schedule": runner.schedule(
            list(runner.goals.CASES), [7, 8, 9], {"model_calls": 8, "tool_calls": 4, "wall_seconds": 420}
        ),
        "revision_budget": {"model_calls": 8, "tool_calls": 4, "wall_seconds": 420},
        "profiles": {c: copy.deepcopy(profile) for c in runner.CONDITIONS},
        "goal_condition": "science_goal",
        "runtime_concurrency": 1,
        "start_after": None,
        "partitions": {f + ":" + str(s): {} for f in runner.goals.CASES for s in [7, 8, 9]},
    }


def test_independent_predeclared_design_matches_single_revision_total_cost(design):
    totals = audit.check_design(design)
    assert totals["fixed_policy_total_budget"] == {"model_calls": 584, "tool_calls": 292, "wall_seconds": 16620}
    assert totals["fixed_policy"] == totals["revised_policy"]


@pytest.mark.parametrize("damage", ["drop", "duplicate", "swap", "cap", "mode", "history_partition"])
def test_design_rejects_unbalanced_or_nonmatched_controls(design, damage):
    if damage == "drop":
        design["schedule"].pop()
    elif damage == "duplicate":
        design["schedule"][-1] = design["schedule"][0]
    elif damage == "swap":
        design["schedule"][0], design["schedule"][1] = design["schedule"][1], design["schedule"][0]
    elif damage == "cap":
        design["schedule"][0]["goal_budget"]["model_calls"] += 1
    elif damage == "mode":
        design["profiles"]["revised_policy"]["mode"] = "science_agent"
    else:
        design["partitions"].pop("SaSrtA:9")
    with pytest.raises(ValueError):
        audit.check_design(design)


def revision_fixture(root):
    packet = revision_packet(BASE_POLICY, history())
    proposed = candidate(revised(), "Fixture rationale with a falsifiable modelling hypothesis.", packet)
    protocol = {"fixture": True}
    write_json(
        root / "report.json",
        {
            "status": "candidate_submitted",
            "protocol_sha256": digest(protocol),
            "promoted": False,
            "recursive_effectiveness_verified": False,
            "candidate_sha256": identity(proposed),
            "policy_id": proposed["policy_id"],
            "parent_id": proposed["parent_id"],
        },
    )
    (root / "workspace").mkdir()
    (root / "execution").mkdir()
    write_json(root / "workspace/improver_candidate.json", proposed)
    args = {"policy_json": json.dumps(proposed["policy"]), "rationale": proposed["rationale"]}
    events = [
        {
            "kind": "tool_call_start",
            "tool_call_id": "inspect",
            "tool_name": "mcp__venusfactory__inspect_improver_context",
            "args": {},
        },
        {"kind": "tool_result", "tool_call_id": "inspect", "is_error": False, "output": json.dumps(packet)},
        {
            "kind": "tool_call_start",
            "tool_call_id": "submit",
            "tool_name": "mcp__venusfactory__submit_improver_policy",
            "args": args,
        },
        {"kind": "tool_result", "tool_call_id": "submit", "is_error": False, "output": "{}"},
    ]
    write_json(
        root / "workspace/improver_attempt.json",
        {
            "proposal_text_sha256": identity(args["policy_json"]),
            "status": "candidate_submitted",
            "candidate_sha256": identity(proposed),
        },
    )
    write_events(root, events)
    return protocol, packet, proposed, events


def write_events(root, events):
    (root / "execution/events.jsonl").write_text("\n".join(json.dumps(e) for e in events) + "\n")


def test_model_submitted_candidate_and_exact_context_required(tmp_path):
    protocol, packet, proposed, events = revision_fixture(tmp_path)
    assert revision_audit.submitted_candidate(tmp_path, protocol, packet) == proposed
    events[1]["output"] = json.dumps({**packet, "future_score": 0.9})
    write_events(tmp_path, events)
    with pytest.raises(ValueError, match="context observation"):
        revision_audit.submitted_candidate(tmp_path, protocol, packet)


@pytest.mark.parametrize(
    "damage",
    [
        "rationale",
        "second_submit",
        "receipt",
        "failed_context",
        "context_after",
        "late_context_result",
        "false_promotion",
    ],
)
def test_candidate_cannot_be_substituted_or_resubmitted(tmp_path, damage):
    protocol, packet, _, events = revision_fixture(tmp_path)
    if damage == "rationale":
        events[2]["args"]["rationale"] = "Substituted operator narrative instead of model output."
    elif damage == "second_submit":
        events.append(copy.deepcopy(events[2]))
    elif damage == "receipt":
        write_json(tmp_path / "workspace/improver_attempt.json", {"status": "candidate_submitted"})
    elif damage == "failed_context":
        events[1]["is_error"] = True
    elif damage == "context_after":
        events = events[2:] + events[:2]
    elif damage == "late_context_result":
        events = [events[0], events[2], events[1], events[3]]
    else:
        report = json.loads((tmp_path / "report.json").read_text())
        report["promoted"] = True
        write_json(tmp_path / "report.json", report)
    write_events(tmp_path, events)
    with pytest.raises(ValueError):
        revision_audit.submitted_candidate(tmp_path, protocol, packet)


def test_historical_projection_omits_reuse_and_final_outcomes(tmp_path):
    protocol = {"model_plugin_id": "generated_features.v1", "split": {"family": "fixture"}}
    state = {
        "status": "budget_exhausted",
        "inflight": None,
        "cycles": [
            {"index": 0, "phase": "develop", "status": "failed", "error_type": "FixtureError", "future_spearman": 0.9},
            {"index": 1, "phase": "reuse", "status": "failed", "future_spearman": 0.9},
        ],
    }
    write_json(tmp_path / "protocol.json", protocol)
    write_json(tmp_path / "goal.json", state)
    refs = [
        {
            "path": str(tmp_path),
            "goal_sha256": file_hash(tmp_path / "goal.json"),
            "protocol_sha256": file_hash(tmp_path / "protocol.json"),
        }
    ]
    projected = revision_audit.history_projection(refs)
    assert len(projected) == 1 and projected[0]["error_type"] == "FixtureError"
    assert "future_spearman" not in projected[0]
    state["inflight"] = {"index": 2}
    write_json(tmp_path / "goal.json", state)
    refs[0]["goal_sha256"] = file_hash(tmp_path / "goal.json")
    with pytest.raises(ValueError, match="terminal"):
        revision_audit.history_projection(refs)


def test_cost_replay_keeps_failure_denominators_missing_usage_and_single_revision(design):
    rows = [
        {
            **row,
            "complete_objective": False,
            "ranking_utility": -1.0,
            "observed_model_calls": 3,
            "observed_tool_calls": 2,
            "charged_usage": {"wall_seconds": 10},
            "token_usage": {"prompt_tokens": 20, "completion_tokens": 10, "complete": False},
        }
        for row in design["schedule"]
    ]
    receipt = {
        "observed_model_calls": 5,
        "observed_tool_calls": 2,
        "host_wall_seconds": 20,
        "token_usage": {"prompt_tokens": 11, "completion_tokens": 7, "complete": True},
    }
    costs = runner.cost_summary(rows, receipt)
    for cost in costs:
        revised_condition = cost["condition"] == "revised_policy"
        cost.update(
            reported_prompt_tokens=120 + (11 if revised_condition else 0),
            reported_completion_tokens=60 + (7 if revised_condition else 0),
            token_usage_complete=False,
        )
    audit.check_costs(rows, receipt, costs)
    assert audit.contrasts(rows) == runner.contrasts(rows)
    for field, value in [("revision_model_calls", 30), ("token_usage_complete", True), ("allocations", 5)]:
        corrupt = copy.deepcopy(costs)
        next(c for c in corrupt if c["condition"] == "revised_policy")[field] = value
        with pytest.raises(ValueError):
            audit.check_costs(rows, receipt, corrupt)
    with pytest.raises(ValueError):
        audit.check_costs(rows[:-1], receipt, costs)


def test_partial_token_usage_cannot_claim_zero_unknown_cost(tmp_path):
    path = tmp_path / "requests.jsonl"
    rows = [
        {"event": "request", "call": 1},
        {"event": "response", "call": 1, "usage": {"prompt_tokens": 10, "completion_tokens": 4}},
        {"event": "request", "call": 2},
        {"event": "response", "call": 2},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows))
    result = revision_audit.usage([path])
    assert result["prompt_tokens"] == 10 and result["responses_with_usage"] == 1 and not result["complete"]
    assert result == runner.token_usage([path])


@pytest.mark.parametrize("damage", [None, "history", "cost", "outcome", "barrier"])
def test_terminal_audit_retains_eighteen_missing_model_failures(tmp_path, monkeypatch, design, damage):
    from agent.science_goal import GoalSpec, GoalStore

    root = tmp_path / "fixture_campaign"
    root.mkdir()
    shared = history()
    proposed = candidate(
        revised(), "Fixture revised strategy for the full audit path.", revision_packet(BASE_POLICY, shared)
    )
    design.update(revision=str(tmp_path / "fixture_revision"), parent_id=identity(BASE_POLICY))
    for key in design["partitions"]:
        family, seed = key.split(":")
        design["partitions"][key] = {
            "scope": {},
            "source": {},
            "split": {"family": family, "seed": int(seed)},
            "data_hashes": {},
        }
    record = {"design": design, "design_sha256": digest(design), "implementation": {}}
    record = json.loads(json.dumps(record))
    design = record["design"]
    write_json(root / "plan.json", record)
    write_json(root / "parent_policy.json", BASE_POLICY)
    write_json(root / "candidate_policy.json", proposed["policy"])
    write_json(root / "shared_development_history.json", shared)
    receipt = {
        "history_sha256": identity(shared),
        "observed_model_calls": 2,
        "observed_tool_calls": 2,
        "host_wall_seconds": 1.0,
        "token_usage": {"prompt_tokens": 30, "completion_tokens": 5, "complete": True},
    }
    write_json(root / "revision_receipt.json", receipt)
    protocols = {}
    definitions = {}
    for row in design["schedule"]:
        folder = root / "arms" / row["id"]
        profile = Profile.from_dict(design["profiles"][row["condition"]])
        store = GoalStore.create(
            folder, GoalSpec("Fixture", ("accepted_model", "fresh_session_reuse"), **row["goal_budget"]), profile
        )
        store.mutate(lambda state: state.update(status="budget_exhausted"))
        state = store.read()
        definition = state["definition_sha256"]
        chosen = proposed["policy"] if row["condition"] == "revised_policy" else BASE_POLICY
        arm = {
            **design["partitions"][row["case"] + ":" + str(row["seed"])],
            "comparison_allocation": row,
            "improver": {"policy_id": identity(chosen), "shared_history_sha256": identity(shared)},
        }
        write_json(folder / "protocol.json", arm)
        write_json(folder / "improver_policy.json", chosen)
        write_json(folder / "shared_development_history.json", shared)
        protocols[row["id"]] = file_hash(folder / "protocol.json")
        definitions[row["id"]] = definition
        write_json(
            folder / "comparison_result.json",
            {
                "allocation": row,
                "status": "budget_exhausted",
                "implementation_unchanged": True,
                "episodes": 0,
                "usage": state["usage"],
                "criteria": None,
                "selected_model": None,
                "external_scripted_turns": 0,
                "internal_scripted_gates": 0,
                "human_interventions": 0,
                "runtime_failures": [],
            },
        )
    protocol = {
        **record,
        "plan_sha256": file_hash(root / "plan.json"),
        "candidate_policy_id": proposed["policy_id"],
        "revision_receipt_sha256": file_hash(root / "revision_receipt.json"),
        "arm_protocols": protocols,
        "goal_definitions": definitions,
    }
    write_json(root / "protocol.json", protocol)
    monkeypatch.setattr(runner, "verify_campaign", lambda _: protocol)
    runner.goals.freeze_goal_barrier(root, design["schedule"])
    analysis = runner.score_offspring(root, design["schedule"])
    if damage == "cost":
        analysis["costs"][1]["revision_model_calls"] = 12
    if damage == "outcome":
        analysis["rows"][0]["ranking_utility"] = 0.9
    write_json(root / "analysis.json", analysis)
    write_json(
        root / "report.json",
        {
            "status": "campaign_completed",
            "all_allocations_attempted": True,
            "design_sha256": record["design_sha256"],
            "analysis_sha256": file_hash(root / "analysis.json"),
            "development_barrier_sha256": file_hash(root / "development_barrier.json"),
            "allocations": [{"id": r["id"], "status": "budget_exhausted", "exit_code": 0} for r in design["schedule"]],
            "reported_complete_objectives": 0,
        },
    )
    pre = tmp_path / "fixture_preflight.json"
    write_json(
        pre,
        {
            "status": "plan_verified_revision_pending",
            "plan_sha256": file_hash(root / "plan.json"),
            "revision_executed": False,
            "offspring_executed": False,
        },
    )
    monkeypatch.setattr(audit, "check_plan", lambda *_: (record, {}))
    monkeypatch.setattr(audit, "audit_revision", lambda _: {"receipt": receipt, "candidate": proposed})
    monkeypatch.setattr(audit, "audit_goal", lambda *_: {"episodes": []})
    if damage == "history":
        write_json(root / "arms" / design["schedule"][0]["id"] / "shared_development_history.json", [])
    if damage == "barrier":
        folder = root / "arms" / design["schedule"][-1]["id"]
        GoalStore(folder).mutate(lambda state: state.update(status="active"))
    if damage:
        reason = {
            "history": "Unequal policy or shared history",
            "cost": "token cost differs",
            "outcome": "Outcomes or contrasts differ",
            "barrier": "Terminal outcome changed",
        }[damage]
        with pytest.raises(ValueError, match=reason):
            audit.audit(root, tmp_path / "fixture_implementation", tmp_path / "audit", pre)
    else:
        result = audit.audit(root, tmp_path / "fixture_implementation", tmp_path / "audit", pre)
        assert result["allocations_checked"] == 18 and len(result["goals"]) == 18
        assert all(r["spearman"] is None and r["ranking_utility"] == -1 for r in analysis["rows"])
