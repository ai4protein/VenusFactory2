"""Independent full-denominator audit of the NucB feedback/mode factorial.

All 18 goals must terminate before this auditor reads any final metrics. The
original experiment is immutable; no API, training experiment or GPU job is launched.
"""

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

from scripts.rsi.evaluation.audit_goal_reservations import retained_timeout_reservation
from scripts.rsi.cases.audit_nucb_science_goal import METRIC, compare_metrics, independent_metrics
from scripts.rsi.cases.audit_nucb_science_goal import audit as audit_goal
from scripts.rsi.cases.audit_reap_generated import read, require
from scripts.rsi.cases.audit_reap_goal_comparison import selection_from_replayed_assessments, turn_accounting, verify_barrier

from agent.harness.contracts import digest, file_hash
from agent.science_goal import GoalStore

CONDITIONS = ("science_goal", "persistent_agent_ablation", "scripted_agent_turns")
FEEDBACK = ("none", "source_and_metrics")


def verify_design(protocol, report):
    design = protocol["design"]
    rows = design["schedule"]
    require(digest(design) == protocol["design_sha256"] == report["design_sha256"], "Design binding changed")
    require(len(rows) == len({r["id"] for r in rows}) == 18, "All 18 unique allocations required")
    require(
        {(r["repeat"], r["condition"], r["feedback"], r["case"]) for r in rows}
        == {(i, c, f, "nucb") for i in range(3) for c in CONDITIONS for f in FEEDBACK},
        "Unbalanced feedback/mode design",
    )
    expected = []
    combinations = [(c, f) for c in CONDITIONS for f in FEEDBACK]
    for repeat in range(3):
        order = combinations[repeat * 2 :] + combinations[: repeat * 2]
        expected.extend(
            {"id": f"r{repeat:02d}_{c}_{f}", "case": "nucb", "repeat": repeat, "condition": c, "feedback": f}
            for c, f in order
        )
    require(rows == expected, "Predeclared rotated allocation order changed")
    profiles = [{k: v for k, v in design["profiles"][c].items() if k != "mode"} for c in CONDITIONS]
    require(profiles == [profiles[0]] * 3, "Common model, harness or episode budget differs")
    require(
        profiles[0]["model"]["model"] == "glm-5.2" and profiles[0]["harness"] == "langgraph", "Declared backend differs"
    )
    require(
        design["goal_budget"] == {"max_cycles": 6, "model_calls": 64, "tool_calls": 32, "wall_seconds": 1800},
        "Declared total budget differs",
    )
    require(
        profiles[0]["budget"]
        == {
            "wall_seconds": 900,
            "request_seconds": 300,
            "model_calls": 24,
            "tool_calls": 12,
            "output_tokens": 12288,
            "temperature": 0.1,
            "thinking": "enabled",
        },
        "Declared episode budget differs",
    )
    require(design["runtime_concurrency"] == 1 and design["start_after"] is None, "Execution order policy differs")
    return rows


def independent_contrasts(rows):
    table = {(x["repeat"], x["condition"], x["feedback"]): x for x in rows}
    result = []
    for repeat in range(3):
        for condition in CONDITIONS:
            a, b = table[(repeat, condition, "none")], table[(repeat, condition, "source_and_metrics")]
            result.append(
                {
                    "repeat": repeat,
                    "condition": condition,
                    "comparison": "source_and_metrics_minus_none",
                    "complete_objective_difference": int(b["complete_objective"]) - int(a["complete_objective"]),
                    "brier_skill_difference": b["all_allocation_brier_skill"] - a["all_allocation_brier_skill"],
                }
            )
        for feedback in FEEDBACK:
            a = table[(repeat, "science_goal", feedback)]
            for condition in CONDITIONS[1:]:
                b = table[(repeat, condition, feedback)]
                result.append(
                    {
                        "repeat": repeat,
                        "feedback": feedback,
                        "comparison": "science_goal_minus_" + condition,
                        "complete_objective_difference": int(a["complete_objective"]) - int(b["complete_objective"]),
                        "brier_skill_difference": a["all_allocation_brier_skill"] - b["all_allocation_brier_skill"],
                    }
                )
    return result


def verify_request_parameters(folder, profile):
    count = 0
    for path in (folder / "episodes").glob("*/execution/model_requests.jsonl"):
        for line in path.read_text().splitlines():
            event = json.loads(line)
            if event["event"] != "request":
                continue
            require(
                event["model"] == profile["model"]["model"]
                and event["temperature"] == profile["budget"]["temperature"]
                and event["max_tokens"] == profile["budget"]["output_tokens"]
                and event["thinking_sent"] == {"type": profile["budget"]["thinking"], "clear_thinking": True},
                "Observed model request parameters differ",
            )
            count += 1
    return count


def verify_revision_receipts(folder, state):
    changed = 0
    for cycle in state["cycles"]:
        if retained_timeout_reservation(folder, state, cycle) is not None:
            # The shared audit binds the actual prompt/capability even when the
            # cancelled subclass never writes its final revision receipt.
            continue
        episode = folder / "episodes" / f"cycle_{cycle['index']:03d}"
        feedback = read(episode / "development_feedback.json")
        require(
            cycle["feedback_source_sha256"] == [row["source_sha256"] for row in feedback],
            "Revision parent sources changed",
        )
        candidate = episode / "workspace/candidate.json"
        if candidate.exists():
            sha = read(candidate)["source_sha256"]
            expected = bool(feedback) and sha not in cycle["feedback_source_sha256"]
            require(
                cycle["submitted_source_sha256"] == sha and cycle["source_bytes_changed_from_feedback"] == expected,
                "Revision source identity changed",
            )
            changed += int(expected)
        else:
            require(
                "submitted_source_sha256" not in cycle and "source_bytes_changed_from_feedback" not in cycle,
                "Missing candidate has a revision claim",
            )
    return changed


def audit(root, implementation_root, output, *, assets, rebuilt):
    root, implementation_root, output = root.resolve(), implementation_root.resolve(), output.resolve()
    report = read(root / "report.json")
    require(
        report["status"] == "campaign_completed" and report["all_allocations_attempted"],
        "Complete 18-goal campaign required",
    )
    require(
        not output.is_relative_to(root) and not output.is_relative_to(implementation_root),
        "Audit output must be outside historical evidence",
    )
    protocol = read(root / "protocol.json")
    rows = verify_design(protocol, report)
    design = protocol["design"]
    require(file_hash(root / "analysis.json") == report["analysis_sha256"], "Analysis binding changed")
    require(
        [x["id"] for x in report["allocations"]] == [x["id"] for x in rows],
        "Missing or reordered controller allocation",
    )
    for name, sha in protocol["implementation"].items():
        require(
            file_hash(root / "implementation" / name) == sha == file_hash(implementation_root / name),
            "Frozen implementation changed",
        )
    verify_barrier(root, rows)
    before = {str(p.relative_to(root)): file_hash(p) for p in root.rglob("*") if p.is_file()}
    output.mkdir(parents=True, exist_ok=False)
    outcomes = []
    verified = []
    for allocation in rows:
        folder = root / "arms" / allocation["id"]
        state = GoalStore(folder).read()
        result = read(folder / "comparison_result.json")
        arm = read(folder / "protocol.json")
        require(
            file_hash(folder / "protocol.json") == protocol["arm_protocols"][allocation["id"]]
            and state["definition_sha256"] == protocol["goal_definitions"][allocation["id"]],
            "Allocation binding changed",
        )
        controller = next(x for x in report["allocations"] if x["id"] == allocation["id"])
        require(controller["status"] == state["status"] and controller["exit_code"] == 0, "Worker termination differs")
        require(
            result["allocation"] == allocation and state["definition"]["condition"] == allocation["condition"],
            "Condition identity differs",
        )
        mode = "science_expert" if allocation["condition"] == "science_goal" else "science_agent"
        require(
            state["definition"]["profile"] == design["profiles"][allocation["condition"]]
            and state["definition"]["profile"]["mode"] == mode,
            "Executor differs",
        )
        require(
            all(state["definition"]["goal"][k] == v for k, v in design["goal_budget"].items()), "Total budget differs"
        )
        require(
            arm["data_hashes"] == protocol["paired_data_hashes"]
            and arm["development_feedback"] == allocation["feedback"],
            "Paired input or feedback policy differs",
        )
        replay = audit_goal(folder, implementation_root, assets=assets, rebuilt=rebuilt)
        require(
            result["criteria"] == state["assessment"] and result["usage"] == state["usage"], "Reported outcome differs"
        )
        counts = turn_accounting(folder, allocation, state, result)
        require(
            verify_request_parameters(folder, state["definition"]["profile"]) == counts["observed_model_calls"],
            "Observed request denominator differs",
        )
        revisions = verify_revision_receipts(folder, state)
        selected = selection_from_replayed_assessments(folder, state, replay, metric=METRIC, compare=compare_metrics)
        require(selected == result["selected_model"], "Development-only selection differs")
        scored = read(folder / "selected_future_metrics.json")
        require(scored["selected_model"] == selected, "Scored source identity differs")
        measured = None
        if selected:
            model = read(folder / selected["controller"] / "artifacts/versions" / selected["model_id"] / "model.json")
            try:
                _, measured = independent_metrics(model, read(folder / "data/future.json"), arm["scope"])
            except Exception as exc:
                require(
                    scored["status"] == "scoring_failed" and scored["error_type"] == type(exc).__name__,
                    "Scoring failure differs",
                )
            else:
                require(
                    scored["status"] == "completed" and scored["training_replayed"],
                    "Future scoring or independent refit missing",
                )
                compare_metrics(scored["metrics"], measured)
        else:
            require(scored["status"] == "no_verified_candidate", "Absent model score differs")
        metrics = scored["metrics"] if measured is not None else None
        outcomes.append(
            {
                **allocation,
                "status": state["status"],
                "complete_objective": state["status"] == "completed",
                "selected_model": selected,
                "future_scoring_status": scored["status"],
                "raw_metrics": metrics,
                "all_allocation_brier_skill": metrics[METRIC] if metrics else 0.0,
                "uses_prior_fallback": metrics is None,
                "episodes": len(state["cycles"]),
                "source_revision_episodes": revisions,
                "charged_usage": state["usage"],
                **counts,
            }
        )
        path = output / (allocation["id"] + ".json")
        path.write_text(
            json.dumps(
                {
                    "goal_audit": replay,
                    "selection": selected,
                    "future_metrics": measured,
                    "turn_accounting": counts,
                    "source_revision_episodes": revisions,
                },
                indent=2,
                allow_nan=False,
            )
            + "\n"
        )
        verified.append({"id": allocation["id"], "audit_sha256": file_hash(path)})
        print(json.dumps(verified[-1]), flush=True)
    observed = read(root / "analysis.json")
    require(outcomes == observed["rows"], "All-allocation outcomes differ")
    contrast = independent_contrasts(outcomes)
    require(contrast == observed["paired_contrasts"] and len(contrast) == 21, "Factorial contrasts differ")
    require(
        sum(x["complete_objective"] for x in outcomes) == report["reported_complete_objectives"],
        "Completion count differs",
    )
    require(all(file_hash(root / n) == sha for n, sha in before.items()), "Historical experiment changed during audit")
    helpers = [
        Path(__file__),
        Path(__file__).with_name("audit_nucb_science_goal.py"),
        Path(__file__).with_name("audit_reap_science_goal.py"),
        Path(__file__).with_name("audit_reap_goal_comparison.py"),
        Path(__file__).with_name("audit_reap_generated.py"),
        Path(__file__).with_name("audit_goal_reservations.py"),
    ]
    result = {
        "status": "passed",
        "allocations_checked": 18,
        "contrasts_checked": 21,
        "external_model_calls": 0,
        "source_run_unchanged": True,
        "goals": verified,
        "protocol_sha256": file_hash(root / "protocol.json"),
        "report_sha256": file_hash(root / "report.json"),
        "analysis_sha256": file_hash(root / "analysis.json"),
        "auditors": {p.name: file_hash(p) for p in helpers},
        "scope": "Independent source, feedback, runtime, budget, training and probability-metric replay on all 18 retrospective goals. Byte changes do not establish algorithmic novelty or useful recursive improvement.",
    }
    (output / "audit.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    for p in helpers:
        shutil.copy2(p, output / p.name)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--implementation-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--rebuilt", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(audit(args.run, args.implementation_root, args.output, assets=args.assets, rebuilt=args.rebuilt)),
        flush=True,
    )
