"""Independent matched-policy plan and terminal offspring audit; never calls LLMs."""

from __future__ import annotations

# Direct-script entry point; implementation imports remain package-qualified.
import sys as _entry_sys
from pathlib import Path as _EntryPath
_entry_root = _EntryPath(__file__).resolve().parents[3]
_entry_sys.path[:0] = [str(_entry_root), str(_entry_root / 'src'), str(_entry_root / 'scripts/rsi')]


import argparse
import json
import shutil
from pathlib import Path

from scripts.rsi.evaluation.audit_improver_revision import audit as audit_revision
from scripts.rsi.evaluation.audit_improver_revision import context, policy, usage
from scripts.rsi.cases.audit_reap_generated import compare_metrics, independent_metrics, read, require
from scripts.rsi.cases.audit_reap_goal_comparison import selection_from_replayed_assessments, turn_accounting, verify_barrier
from scripts.rsi.cases.audit_reap_science_goal import audit as audit_goal

from agent.harness.contracts import digest, file_hash
from agent.rsi.cases.reap_model_case import prepare_reap_model_case
from agent.science_goal import GoalStore

CONDITIONS = ("fixed_policy", "revised_policy", "fixed_policy_total_budget")
FAMILIES = ("P450_BM3_FL62", "SaSrtA")


def check_design(design):
    rows, cap = design["schedule"], design["revision_budget"]
    require(len(rows) == len({r["id"] for r in rows}) == 18, "Exactly 18 unique policy allocations required")
    require(set(design["profiles"]) == set(CONDITIONS), "Unexpected policy conditions")
    profiles = list(design["profiles"].values())
    require(
        all(p == profiles[0] for p in profiles)
        and profiles[0]["mode"] == "science_expert"
        and design["goal_condition"] == "science_goal",
        "Common Scientist harness/profile changed",
    )
    require(design["runtime_concurrency"] == 1 and design["start_after"] is None, "Execution order changed")
    require(
        set(cap) == {"model_calls", "tool_calls", "wall_seconds"}
        and cap == {"model_calls": 8, "tool_calls": 4, "wall_seconds": 420},
        "Frozen revision cap differs",
    )
    expected = []
    block = 0
    for repeat, seed in enumerate([7, 8, 9]):
        for family in FAMILIES if repeat % 2 == 0 else FAMILIES[::-1]:
            order = CONDITIONS[block % 3 :] + CONDITIONS[: block % 3]
            for condition in order:
                budget = {"max_cycles": 6, "model_calls": 96, "tool_calls": 48, "wall_seconds": 2700}
                if condition == CONDITIONS[2]:
                    budget["model_calls"] += 1 + (block < 2)
                    budget["tool_calls"] += int(block < 4)
                    budget["wall_seconds"] += 70
                expected.append(
                    {
                        "id": f"{family}_seed{seed:02d}_{condition}",
                        "case": family,
                        "seed": seed,
                        "repeat": repeat,
                        "condition": condition,
                        "goal_budget": budget,
                    }
                )
            block += 1
    require(rows == expected, "Allocation, pairing, order or distributed total budget changed")
    require(
        set(design["partitions"]) == {f + ":" + str(s) for f in FAMILIES for s in [7, 8, 9]},
        "Partition denominator differs",
    )
    totals = {
        c: {k: sum(row["goal_budget"][k] for row in rows if row["condition"] == c) for k in cap} for c in CONDITIONS
    }
    require(
        all(
            totals[CONDITIONS[0]][k] == totals[CONDITIONS[1]][k]
            and totals[CONDITIONS[2]][k] == totals[CONDITIONS[1]][k] + cap[k]
            for k in cap
        ),
        "Revision allowance charged more than once or omitted",
    )
    return totals


def check_plan(root, implementation_root):
    record = read(root / "plan.json")
    design = record["design"]
    require(digest(design) == record["design_sha256"], "Plan identity changed")
    totals = check_design(design)
    for name, sha in record["implementation"].items():
        require(
            file_hash(root / "implementation" / name) == sha == file_hash(implementation_root / name),
            "Frozen comparison implementation changed",
        )
    revision = Path(design["revision"])
    require(file_hash(revision / "protocol.json") == design["revision_protocol_sha256"], "Planned revision changed")
    rp, packet = context(revision)
    require(
        design["parent_id"] == packet["parent_id"]
        and design["history_sha256"] == packet["history_sha256"]
        and read(root / "parent_policy.json") == packet["parent_policy"]
        and read(root / "shared_development_history.json") == packet["development_history"],
        "Shared starting state differs",
    )
    for k in ["provider", "model", "base_url", "protocol"]:
        require(design["profiles"][CONDITIONS[0]]["model"][k] == rp["profile"]["model"][k], "Revision backbone changed")
    require(design["profiles"][CONDITIONS[0]]["harness"] == rp["profile"]["harness"], "Revision harness changed")
    require(
        design["revision_budget"] == {k: rp["profile"]["budget"][k] for k in design["revision_budget"]},
        "Revision cap binding differs",
    )
    for key, expected in design["partitions"].items():
        family, seed = key.split(":")
        source = design["sources"]
        scope, train, val, future, origins, split = prepare_reap_model_case(
            source["admission"], source["features"], source["baseline"], family, int(seed)
        )
        actual = {
            "scope": scope,
            "source": origins,
            "split": split,
            "data_hashes": {n: digest(v) for n, v in [("training", train), ("validation", val), ("future", future)]},
        }
        require(actual == expected, "Paired public-source partition changed")
    return record, totals


def preflight(root, implementation_root):
    record, totals = check_plan(root, implementation_root)
    report = read(root / "report.json")
    revision = Path(record["design"]["revision"])
    require(
        report["status"] == "planned_waiting_for_revision"
        and report["allocations"] == []
        and report["plan_sha256"] == file_hash(root / "plan.json")
        and report["planned"] == 18,
        "Only an unstarted comparison can receive preflight status",
    )
    require(
        read(revision / "report.json")["status"] == "prepared"
        and not (revision / "execution").exists()
        and not (root / "arms").exists()
        and not (root / "analysis.json").exists(),
        "Preflight must precede revision and offspring execution",
    )
    return {
        "status": "plan_verified_revision_pending",
        "plan_sha256": file_hash(root / "plan.json"),
        "design_sha256": record["design_sha256"],
        "revision_report_sha256": file_hash(revision / "report.json"),
        "allocations_verified": 18,
        "batch_budget_caps": totals,
        "revision_executed": False,
        "offspring_executed": False,
        "performance_verified": False,
        "scope": "Offline design/source/partition audit only; no candidate or performance result exists.",
    }


def contrasts(rows):
    table = {(r["case"], r["seed"], r["condition"]): r for r in rows}
    require(len(table) == len(rows) == 18, "Missing allocated outcome")
    out = []
    for family, seed in sorted({key[:2] for key in table}):
        for left, right in [
            (CONDITIONS[1], CONDITIONS[0]),
            (CONDITIONS[1], CONDITIONS[2]),
            (CONDITIONS[2], CONDITIONS[0]),
        ]:
            a, b = table[(family, seed, left)], table[(family, seed, right)]
            out.append(
                {
                    "case": family,
                    "seed": seed,
                    "left": left,
                    "right": right,
                    "complete_objective_difference": int(a["complete_objective"]) - int(b["complete_objective"]),
                    "ranking_utility_difference": a["ranking_utility"] - b["ranking_utility"],
                    "downstream_model_calls_difference": a["observed_model_calls"] - b["observed_model_calls"],
                }
            )
    return out


def check_costs(rows, receipt, costs):
    require(
        len(rows) == 18 and len(costs) == 3 and {c["condition"] for c in costs} == set(CONDITIONS),
        "Cost denominator differs",
    )
    for c in costs:
        group = [r for r in rows if r["condition"] == c["condition"]]
        require(len(group) == 6, "Missing failed allocation cost")
        revised = c["condition"] == CONDITIONS[1]
        expected = {
            "allocations": 6,
            "complete_objectives": sum(r["complete_objective"] for r in group),
            "revision_model_calls": receipt["observed_model_calls"] if revised else 0,
            "revision_tool_calls": receipt["observed_tool_calls"] if revised else 0,
            "revision_elapsed_seconds": receipt["host_wall_seconds"] if revised else 0,
            "downstream_model_calls": sum(r["observed_model_calls"] for r in group),
            "downstream_tool_calls": sum(r["observed_tool_calls"] for r in group),
            "total_charged_goal_wall_seconds": sum(r["charged_usage"]["wall_seconds"] for r in group),
        }
        for kind in ["model", "tool"]:
            expected["total_observed_" + kind + "_calls"] = (
                expected["downstream_" + kind + "_calls"] + expected["revision_" + kind + "_calls"]
            )
        usages = [r["token_usage"] for r in group] + ([receipt["token_usage"]] if revised else [])
        expected.update(
            reported_prompt_tokens=sum(u["prompt_tokens"] for u in usages),
            reported_completion_tokens=sum(u["completion_tokens"] for u in usages),
            token_usage_complete=all(u["complete"] for u in usages),
        )
        require(all(c.get(k) == v for k, v in expected.items()), "Revision, failed-outcome or token cost differs")


def audit(root, implementation_root, output, preflight_path):
    record, _ = check_plan(root, implementation_root)
    pre = read(preflight_path)
    require(
        pre["status"] == "plan_verified_revision_pending"
        and pre["plan_sha256"] == file_hash(root / "plan.json")
        and pre["revision_executed"] is False
        and pre["offspring_executed"] is False,
        "Missing bound pre-execution audit",
    )
    protocol, report = read(root / "protocol.json"), read(root / "report.json")
    design = record["design"]
    require(
        report["status"] == "campaign_completed" and report["all_allocations_attempted"],
        "Complete terminal campaign required",
    )
    require(
        protocol["design"] == design
        and protocol["design_sha256"] == record["design_sha256"] == report["design_sha256"]
        and protocol["plan_sha256"] == file_hash(root / "plan.json")
        and protocol["implementation"] == record["implementation"],
        "Execution differs from predeclared plan",
    )
    require(file_hash(root / "analysis.json") == report["analysis_sha256"], "Analysis binding changed")
    revision = audit_revision(Path(design["revision"]))
    receipt = revision["receipt"]
    proposed = revision["candidate"]
    require(
        receipt == read(root / "revision_receipt.json")
        and file_hash(root / "revision_receipt.json") == protocol["revision_receipt_sha256"]
        and protocol["candidate_policy_id"] == proposed["policy_id"]
        and read(root / "candidate_policy.json") == proposed["policy"],
        "Revision receipt or policy changed",
    )
    allocations = design["schedule"]
    barrier = verify_barrier(root, allocations)
    require([x["id"] for x in report["allocations"]] == [x["id"] for x in allocations], "Missing or reordered worker")
    output.mkdir(parents=True, exist_ok=False)
    verified = []
    outcomes = []
    for row, worker in zip(allocations, report["allocations"], strict=True):
        folder = root / "arms" / row["id"]
        state = GoalStore(folder).read()
        arm, result = read(folder / "protocol.json"), read(folder / "comparison_result.json")
        require(
            file_hash(folder / "protocol.json") == protocol["arm_protocols"][row["id"]]
            and state["definition_sha256"] == protocol["goal_definitions"][row["id"]],
            "Arm binding changed",
        )
        expected_policy = proposed["policy"] if row["condition"] == CONDITIONS[1] else read(root / "parent_policy.json")
        require(
            read(folder / "improver_policy.json") == expected_policy
            and arm["improver"]["policy_id"] == policy(expected_policy)
            and read(folder / "shared_development_history.json") == read(root / "shared_development_history.json")
            and arm["improver"]["shared_history_sha256"] == receipt["history_sha256"],
            "Unequal policy or shared history",
        )
        definition = state["definition"]
        require(
            definition["condition"] == "science_goal"
            and definition["profile"] == design["profiles"][row["condition"]]
            and all(definition["goal"][k] == v for k, v in row["goal_budget"].items()),
            "Goal contract differs",
        )
        expected = design["partitions"][row["case"] + ":" + str(row["seed"])]
        require(
            all(arm[k] == expected[k] for k in ["scope", "source", "split", "data_hashes"]), "Arm partition differs"
        )
        require(
            result["allocation"] == row
            and arm["comparison_allocation"] == row
            and worker["exit_code"] == 0
            and worker["status"] == state["status"],
            "Worker outcome changed",
        )
        replay = audit_goal(folder, implementation_root)
        require(
            result["criteria"] == state["assessment"] and result["usage"] == state["usage"], "Goal assessment differs"
        )
        counts = turn_accounting(folder, row, state, result)
        selected = selection_from_replayed_assessments(folder, state, replay)
        require(result["selected_model"] == selected, "Selection differs from independent development replay")
        scored = read(folder / "selected_future_metrics.json")
        metrics = None
        require(scored["selected_model"] == selected, "Scored candidate differs")
        if selected:
            model = read(folder / selected["controller"] / "artifacts/versions" / selected["model_id"] / "model.json")
            try:
                _, metrics = independent_metrics(model, read(folder / "data/future.json"), arm["scope"])
            except Exception as exc:
                require(
                    scored["status"] == "scoring_failed" and scored["error_type"] == type(exc).__name__,
                    "Scoring failure differs",
                )
            else:
                require(scored["status"] == "completed", "Missing future score")
                compare_metrics(scored["metrics"], metrics)
        else:
            require(scored["status"] == "no_verified_candidate", "Missing-model outcome differs")
        measured = scored["metrics"] if metrics else None
        tokens = usage([p / "model_requests.jsonl" for p in sorted((folder / "episodes").glob("*/execution"))])
        tokens["complete"] &= tokens["requests"] == counts["observed_model_calls"] == state["usage"]["model_calls"]
        outcome = {
            **row,
            "status": state["status"],
            "complete_objective": state["status"] == "completed",
            "selected_model": selected,
            "future_scoring_status": scored["status"],
            "spearman": measured["spearman"] if measured else None,
            "mse_skill": measured["mse_skill_vs_training_mean"] if measured else None,
            "ranking_utility": measured["spearman"] if measured else -1.0,
            "utility_is_failure_floor": metrics is None,
            "episodes": len(state["cycles"]),
            "charged_usage": state["usage"],
            **counts,
            "token_usage": tokens,
        }
        outcomes.append(outcome)
        path = output / (row["id"] + ".json")
        path.write_text(json.dumps({"goal": replay, "outcome": outcome}, indent=2, allow_nan=False) + "\n")
        verified.append({"id": row["id"], "audit_sha256": file_hash(path)})
        print(json.dumps(verified[-1]), flush=True)
    analysis = read(root / "analysis.json")
    require(
        analysis["rows"] == outcomes and analysis["paired_contrasts"] == contrasts(outcomes),
        "Outcomes or contrasts differ",
    )
    require(
        analysis["revision_receipt"] == receipt
        and analysis["parent_id"] == proposed["parent_id"]
        and analysis["candidate_id"] == proposed["policy_id"]
        and analysis["promoted"] is False
        and analysis["recursive_effectiveness_verified"] is False,
        "Unsupported policy claim",
    )
    require(
        sum(r["complete_objective"] for r in outcomes) == report["reported_complete_objectives"],
        "Completion denominator differs",
    )
    check_costs(outcomes, receipt, analysis["costs"])
    return {
        "status": "passed",
        "allocations_checked": 18,
        "contrasts_checked": 18,
        "barrier_count": barrier["count"],
        "goals": verified,
        "revision_audit": revision,
        "plan_sha256": file_hash(root / "plan.json"),
        "preflight_audit_sha256": file_hash(preflight_path),
        "analysis_sha256": file_hash(root / "analysis.json"),
        "scope": "Independent policy provenance, shared history, budgets, numeric outcomes and total costs. One retrospective revision; not independent biological replicates or multigeneration RSI efficacy.",
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["plan", "campaign"])
    p.add_argument("--campaign", type=Path, required=True)
    p.add_argument("--implementation-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--preflight-audit", type=Path)
    a = p.parse_args()
    root, impl, out = a.campaign.resolve(), a.implementation_root.resolve(), a.output.resolve()
    require(
        not out.is_relative_to(root) and not out.is_relative_to(impl),
        "Audit output must be outside historical evidence",
    )
    if a.action == "plan":
        result = preflight(root, impl)
        out.mkdir(parents=True, exist_ok=False)
    else:
        if not a.preflight_audit:
            p.error("Campaign audit needs the pre-execution plan audit")
        result = audit(root, impl, out, a.preflight_audit)
    helpers = [
        Path(__file__),
        Path(__file__).with_name("audit_improver_revision.py"),
        Path(__file__).with_name("audit_reap_goal_comparison.py"),
        Path(__file__).with_name("audit_reap_science_goal.py"),
        Path(__file__).with_name("audit_reap_generated.py"),
    ]
    result["auditors"] = {h.name: file_hash(h) for h in helpers}
    (out / "audit.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    for h in helpers:
        shutil.copy2(h, out / h.name)
    print(json.dumps({k: v for k, v in result.items() if k not in ["goals", "revision_audit"]}), flush=True)
