"""Independent audit of the full REAP goal comparison, including rejected models.

This does not rerun LLMs or alter experiment results. Numerical replay is audit
work, not an additional experiment or an independent biological replicate.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import shutil
from pathlib import Path

from audit_goal_reservations import retained_timeout_reservation
from audit_reap_generated import compare_metrics, independent_metrics, read, require
from audit_reap_science_goal import audit as audit_goal

from agent.harness.contracts import digest, file_hash
from agent.science_goal import GoalStore, goal_cycle_prompt

CONDITIONS = ("science_goal", "persistent_agent_ablation", "scripted_agent_turns")
FAMILIES = ("P450_BM3_FL62", "SaSrtA")


def verify_barrier(root, allocations):
    report, barrier = read(root / "report.json"), read(root / "development_barrier.json")
    require(
        file_hash(root / "development_barrier.json") == report["development_barrier_sha256"],
        "Global goal barrier changed",
    )
    require(
        len(allocations) == len({x["id"] for x in allocations}) == barrier["count"] == 18,
        "All 18 unique allocations required",
    )
    require(barrier["future_metrics_present"] is False, "Premature future scoring")
    require(
        [x["id"] for x in barrier["allocations"]] == [x["id"] for x in allocations],
        "Barrier order or denominator differs",
    )
    for row, binding in zip(allocations, barrier["allocations"], strict=True):
        folder = root / "arms" / row["id"]
        state, result = GoalStore(folder).read(), read(folder / "comparison_result.json")
        require(
            file_hash(folder / "goal.json") == binding["goal_sha256"]
            and file_hash(folder / "comparison_result.json") == binding["comparison_result_sha256"],
            "Terminal outcome changed after barrier",
        )
        require(
            state["status"] == result["status"] == binding["status"]
            and state["status"] in {"completed", "budget_exhausted"},
            "Goal was not terminal at scoring",
        )
        require(
            state["inflight"] is None and result["implementation_unchanged"], "Goal source or inflight state invalid"
        )
    return barrier


def selection_from_replayed_assessments(
    folder, state, replay, *, metric="mse_skill_vs_training_mean", compare=compare_metrics
):
    candidates = []
    for item in replay["episodes"]:
        if not item.get("training_replayed"):
            continue
        index = item["cycle"]
        episode = folder / "episodes" / f"cycle_{index:03d}"
        assessment = read(episode / "acceptance.json")
        compare(assessment["metrics"], item["acceptance_metrics"])
        skill = assessment["metrics"][metric]
        require(math.isfinite(skill), "Nonfinite development selection metric")
        candidates.append(
            {
                "cycle": index,
                "controller": str((episode / "controller").relative_to(folder)),
                "model_id": assessment["model_id"],
                "accepted": item["accepted"],
                "development_skill": skill,
                "assessment_sha256": digest(assessment),
            }
        )
    incumbent = state["progress"].get("accepted_model")
    if incumbent:
        options = [
            x
            for x in candidates
            if x["model_id"] == incumbent["model_id"] and x["controller"] == incumbent["controller"] and x["accepted"]
        ]
        require(len(options) == 1, "Goal incumbent lacks a unique independently replayed acceptance")
        return {**options[0], "selection_reason": "goal_incumbent"}
    accepted = sorted((x for x in candidates if x["accepted"]), key=lambda x: x["cycle"])
    if accepted:
        return {**accepted[0], "selection_reason": "first_independently_accepted"}
    if candidates:
        candidates.sort(key=lambda x: (-x["development_skill"], x["cycle"]))
        return {**candidates[0], "selection_reason": "best_development_submission"}
    return None


def turn_accounting(folder, allocation, state, result):
    condition = allocation["condition"]
    turns = sorted((folder / "turns").glob("turn_*.json"))
    require(len(turns) == len(state["cycles"]) == result["episodes"], "An episode/turn receipt is missing")
    counts = {
        "external_scripted_turns": 0,
        "internal_scripted_gates": 0,
        "human_interventions": 0,
        "observed_model_calls": 0,
        "observed_tool_calls": 0,
    }
    charged = {"model_calls": 0, "tool_calls": 0}
    progress = {}
    failures = []
    for index, path in enumerate(turns):
        turn, cycle = read(path), state["cycles"][index]
        checkpoint = turn["checkpoint"]
        reservation = retained_timeout_reservation(folder, state, cycle)
        require(turn["index"] == index and turn["condition"] == condition, "Turn identity changed")
        require(
            checkpoint["definition"] == state["definition"]
            and checkpoint["cycles"] == state["cycles"][:index]
            and checkpoint["progress"] == progress,
            "Continuation history changed",
        )
        require(
            turn["context"] == goal_cycle_prompt(checkpoint) and digest(turn["context"]) == turn["context_sha256"],
            "Continuation prompt changed",
        )
        external = index == 0 or condition == "scripted_agent_turns"
        require(
            turn["external_user_turn"] is external and turn["human_intervention"] is False, "User-turn policy changed"
        )
        counts["external_scripted_turns"] += external
        expected = copy.deepcopy(state["definition"]["profile"])
        for key in charged:
            expected["budget"][key] = min(expected["budget"][key], state["definition"]["goal"][key] - charged[key])
            require(
                checkpoint["usage"][key] == charged[key] + expected["budget"][key], "Checkpoint reservation changed"
            )
        wall = turn["episode_profile"]["budget"]["wall_seconds"]
        require(0 < wall <= expected["budget"]["wall_seconds"], "Invalid episode time ceiling")
        expected["budget"]["wall_seconds"] = wall
        require(turn["episode_profile"] == expected, "Remaining episode budget or model changed")
        execution = folder / "episodes" / f"cycle_{index:03d}" / "execution"
        run, lock = read(execution / "run.json"), read(execution / "profile.lock.json")
        require(lock["profile"] == expected, "Execution differs from allocated profile")
        counts["internal_scripted_gates"] += len(run.get("scripted_feedback", []))
        counts["human_interventions"] += run.get("human_interventions", 0)
        if run["status"] != "completed":
            failures.append({"episode": index, "status": run["status"]})
        for filename, field, event, target, usage_key in [
            ("model_requests.jsonl", "event", "request", "observed_model_calls", "model_calls"),
            ("events.jsonl", "kind", "tool_call_start", "observed_tool_calls", "tool_calls"),
        ]:
            path = execution / filename
            require(path.exists() or reservation is not None, "Finalized episode trace missing")
            value = sum(
                json.loads(line).get(field) == event
                for line in (path.read_text().splitlines() if path.exists() else [])
                if line.strip()
            )
            debit = reservation[usage_key] if reservation is not None else cycle["usage"][usage_key]
            require(
                value <= debit if reservation is not None else value == debit,
                "Observed call count differs from charged episode",
            )
            counts[target] += value
            charged[usage_key] += debit
        if "progress" in cycle:
            progress = cycle["progress"]
    for key, value in charged.items():
        require(value == state["usage"][key] == result["usage"][key], "Cumulative accounting changed")
    for key in ["external_scripted_turns", "internal_scripted_gates", "human_interventions"]:
        require(counts[key] == result[key], "Intervention accounting changed")
    require(
        counts["human_interventions"] == 0 and failures == result["runtime_failures"],
        "Unexpected intervention or dropped runtime failure",
    )
    return counts


def paired_contrasts(rows):
    table = {(x["case"], x["seed"], x["condition"]): x for x in rows}
    require(len(table) == len(rows) == 18, "All 18 unique outcome rows required")
    contrasts = []
    for case, seed in sorted({key[:2] for key in table}):
        for left, right in [
            (CONDITIONS[0], CONDITIONS[2]),
            (CONDITIONS[0], CONDITIONS[1]),
            (CONDITIONS[1], CONDITIONS[2]),
        ]:
            a, b = table[(case, seed, left)], table[(case, seed, right)]
            contrasts.append(
                {
                    "case": case,
                    "seed": seed,
                    "left": left,
                    "right": right,
                    "complete_objective_difference": int(a["complete_objective"]) - int(b["complete_objective"]),
                    "ranking_utility_difference": a["ranking_utility"] - b["ranking_utility"],
                    "observed_model_calls_difference": a["observed_model_calls"] - b["observed_model_calls"],
                    "external_scripted_turns_difference": a["external_scripted_turns"] - b["external_scripted_turns"],
                }
            )
    return contrasts


def audit(root, implementation_root, output):
    root, implementation_root, output = root.resolve(), implementation_root.resolve(), output.resolve()
    report, protocol = read(root / "report.json"), read(root / "protocol.json")
    require(
        report["status"] == "campaign_completed" and report["all_allocations_attempted"],
        "Complete 18-goal campaign required",
    )
    require(
        not output.is_relative_to(root) and not output.is_relative_to(implementation_root),
        "Audit output must be outside historical evidence",
    )
    design = protocol["design"]
    require(digest(design) == protocol["design_sha256"] == report["design_sha256"], "Goal design changed")
    require(file_hash(root / "analysis.json") == report["analysis_sha256"], "Aggregate analysis changed")
    require(
        design["cases"] == list(FAMILIES)
        and len(set(design["seeds"])) == 3
        and all(type(s) is int and s in range(4, 10) for s in design["seeds"]),
        "Declared protein/seed design differs",
    )
    rows = design["schedule"]
    require(
        {(x["case"], x["seed"], x["condition"]) for x in rows}
        == {(f, s, c) for f in FAMILIES for s in design["seeds"] for c in CONDITIONS},
        "Unbalanced goal allocation",
    )
    require(
        len(report["allocations"]) == 18 and [x["id"] for x in report["allocations"]] == [x["id"] for x in rows],
        "Missing or reordered controller allocation",
    )
    profiles = [{k: v for k, v in design["profiles"][c].items() if k != "mode"} for c in CONDITIONS]
    require(profiles == [profiles[0]] * 3, "Model, harness or common budget differs")
    for name, sha in protocol["implementation"].items():
        require(
            file_hash(root / "implementation" / name) == sha == file_hash(implementation_root / name),
            "Frozen experiment code changed",
        )
    barrier = verify_barrier(root, rows)
    if design["start_after"]:
        dep = Path(design["start_after"])
        receipt = read(root / "dependency_receipt.json")
        require(
            file_hash(dep / "protocol.json") == design["dependency_protocol_sha256"]
            and receipt["report_sha256"] == file_hash(dep / "report.json")
            and read(dep / "report.json")["status"] == "campaign_completed"
            and receipt["finished_before_this_campaign"],
            "Dependency completion changed",
        )
    output.mkdir(parents=True, exist_ok=False)
    verified, outcomes = [], []
    for allocation in rows:
        folder = root / "arms" / allocation["id"]
        state, result, arm_protocol = (
            GoalStore(folder).read(),
            read(folder / "comparison_result.json"),
            read(folder / "protocol.json"),
        )
        require(
            file_hash(folder / "protocol.json") == protocol["arm_protocols"][allocation["id"]]
            and state["definition_sha256"] == protocol["goal_definitions"][allocation["id"]],
            "Allocation binding changed",
        )
        controller_record = next(x for x in report["allocations"] if x["id"] == allocation["id"])
        require(
            controller_record["status"] == state["status"] and controller_record["exit_code"] == 0,
            "Worker termination differs from reported result",
        )
        require(
            result["allocation"] == allocation and state["definition"]["condition"] == allocation["condition"],
            "Condition identity differs",
        )
        mode = "science_expert" if allocation["condition"] == "science_goal" else "science_agent"
        require(
            state["definition"]["profile"] == design["profiles"][allocation["condition"]]
            and state["definition"]["profile"]["mode"] == mode,
            "Native executor mode differs",
        )
        require(
            all(state["definition"]["goal"][k] == v for k, v in design["goal_budget"].items()),
            "Common goal budget differs",
        )
        require(
            arm_protocol["data_hashes"]
            == protocol["paired_data_hashes"][allocation["case"] + ":" + str(allocation["seed"])],
            "Paired data differs",
        )
        require(
            arm_protocol["split"]["family"] == allocation["case"]
            and arm_protocol["split"]["seed"] == allocation["seed"],
            "Task partition changed",
        )
        replay = audit_goal(folder, implementation_root)
        require(
            result["criteria"] == state["assessment"] and result["usage"] == state["usage"],
            "Reported goal outcome differs",
        )
        counts = turn_accounting(folder, allocation, state, result)
        selected = selection_from_replayed_assessments(folder, state, replay)
        require(
            selected == result["selected_model"],
            "Selection used something other than independently replayed development results",
        )
        scored = read(folder / "selected_future_metrics.json")
        require(scored["selected_model"] == selected, "Scored source identity differs")
        metrics = None
        if selected:
            model = read(folder / selected["controller"] / "artifacts/versions" / selected["model_id"] / "model.json")
            try:
                _, metrics = independent_metrics(model, read(folder / "data/future.json"), arm_protocol["scope"])
            except Exception as exc:
                require(
                    scored["status"] == "scoring_failed" and scored["error_type"] == type(exc).__name__,
                    "Future scoring failure differs",
                )
            else:
                require(scored["status"] == "completed", "Future score missing")
                compare_metrics(scored["metrics"], metrics)
        else:
            require(scored["status"] == "no_verified_candidate", "Invalid absent-model score")
        measured = scored["metrics"] if metrics else None
        outcomes.append(
            {
                **allocation,
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
            }
        )
        path = output / (allocation["id"] + ".json")
        path.write_text(
            json.dumps(
                {"goal_audit": replay, "turn_accounting": counts, "selection": selected, "future_metrics": metrics},
                indent=2,
                allow_nan=False,
            )
            + "\n"
        )
        verified.append({"id": allocation["id"], "audit_sha256": file_hash(path)})
        print(json.dumps(verified[-1]), flush=True)
    observed = read(root / "analysis.json")
    require(outcomes == observed["rows"], "All-allocation outcomes or costs differ")
    contrasts = paired_contrasts(outcomes)
    require(contrasts == observed["paired_contrasts"] and len(contrasts) == 18, "Paired contrasts differ")
    require(
        sum(x["complete_objective"] for x in outcomes) == report["reported_complete_objectives"],
        "Completion count differs",
    )
    helpers = [
        Path(__file__),
        Path(__file__).with_name("audit_reap_science_goal.py"),
        Path(__file__).with_name("audit_reap_generated.py"),
        Path(__file__).with_name("audit_goal_reservations.py"),
    ]
    result = {
        "status": "passed",
        "allocations_checked": 18,
        "contrasts_checked": 18,
        "development_barrier_allocations": barrier["count"],
        "goals": verified,
        "protocol_sha256": file_hash(root / "protocol.json"),
        "report_sha256": file_hash(root / "report.json"),
        "analysis_sha256": file_hash(root / "analysis.json"),
        "auditors": {p.name: file_hash(p) for p in helpers},
        "scope": "Independent goal/turn/budget/partition audit with numerical training and selected future-metric replay. Failed goals retained; shared retrospective cohorts are not independent biological replicates or proof of recursive improvement.",
    }
    (output / "audit.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    for path in helpers:
        shutil.copy2(path, output / path.name)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--implementation-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.campaign, args.implementation_root, args.output)), flush=True)
