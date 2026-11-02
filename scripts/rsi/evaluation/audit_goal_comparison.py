"""Read-only audit of matched objective allocations, including failed cells.

Partial reports keep pending allocations in the planned denominator. Replays
are audit evidence, not additional model/biological replicates.
"""

from __future__ import annotations

# Direct-script entry point; implementation imports remain package-qualified.
import sys as _entry_sys
from pathlib import Path as _EntryPath
_entry_root = _EntryPath(__file__).resolve().parents[3]
_entry_sys.path[:0] = [str(_entry_root), str(_entry_root / 'src'), str(_entry_root / 'scripts/rsi')]


import argparse
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

from scripts.rsi.evaluation.audit_generated_campaign import require_equal
from scripts.rsi.evaluation.audit_science_goal import audit as replay_complete_goal
from run_generated_lifecycle import spec
from run_goal_comparison import CONDITIONS, MODES, ROOT, read, verify_protocol
from run_protein_lifecycle import audit_events

from agent.harness.contracts import digest, file_hash
from agent.harness.events import audit_requests, audit_trace
from agent.harness.host import write_json
from agent.science_goal import GoalStore, goal_cycle_prompt


def audit_allocation(root, row, output, source_cache):
    folder = root / "arms" / row["id"]
    store = GoalStore(folder)
    state = store.read()
    condition = row["condition"]
    require_equal(state["definition"]["condition"], condition, "Condition changed")
    require_equal(state["definition"]["profile"]["mode"], MODES[condition], "Executor mode changed")
    result_path = folder / "comparison_result.json"
    if not result_path.exists():
        return {**row, "status": "pending_or_missing_result", "verified_success": False}
    result = read(result_path)
    require_equal(result["allocation"], row, "Allocation identity changed")
    if result["status"] == "completed" and state["status"] != "completed":
        raise ValueError("Reported completion differs from goal state")
    original_protocol = read(folder / "protocol.json")
    for name, sha in original_protocol["data_hashes"].items():
        require_equal(digest(read(folder / "data" / (name + ".json"))), sha, "Case data changed")
    turns = sorted((folder / "turns").glob("turn_*.json"))
    external, gates, humans, observed_models, observed_tools = 0, 0, 0, 0, 0
    charged = {"model_calls": 0, "tool_calls": 0}
    unknown = []
    for index, path in enumerate(turns):
        turn = read(path)
        require_equal(turn["index"], index, "Nonconsecutive episode receipt")
        require_equal(turn["condition"], condition, "Turn condition changed")
        require_equal(turn["checkpoint"]["definition"], state["definition"], "Turn objective changed")
        require_equal(turn["context"], goal_cycle_prompt(turn["checkpoint"]), "Continuation context changed")
        require_equal(turn["context_sha256"], digest(turn["context"]), "Context digest changed")
        expected_external = index == 0 or condition == "scripted_agent_turns"
        require_equal(turn["external_user_turn"], expected_external, "User-turn policy changed")
        require_equal(turn["human_intervention"], False, "Unexpected human intervention")
        external += expected_external
        episode = folder / "episodes" / f"cycle_{index:03d}"
        cycle = next((c for c in state["cycles"] if c["index"] == index), None)
        usage = cycle.get("usage") if cycle else None
        for key in charged:
            charged[key] += usage[key] if usage else turn["episode_profile"]["budget"][key]
        if not usage:
            unknown.append(index)
        execution = episode / "execution"
        if not (execution / "run.json").exists():
            if cycle and cycle["status"] == "completed":
                raise ValueError("Completed episode has no runtime record")
            continue
        run = read(execution / "run.json")
        lock = read(execution / "profile.lock.json")
        require_equal(lock["sha256"], digest({k: v for k, v in lock.items() if k != "sha256"}), "Profile lock changed")
        require_equal(run["profile_sha256"], lock["sha256"], "Runtime binding changed")
        require_equal(lock["profile"], turn["episode_profile"], "Episode profile differs from reserved allocation")
        require_equal(run["mode"], MODES[condition], "Runtime mode changed")
        checks = audit_requests(execution / "model_requests.jsonl")
        events = audit_trace(execution / "events.jsonl", lock["task"]["tools"], lock["profile"]["budget"]["tool_calls"])
        observed_models += checks["model_calls"]
        observed_tools += events["tool_calls"]
        if usage:
            require_equal(usage["model_calls"], checks["model_calls"], "Model-call count changed")
            require_equal(usage["tool_calls"], events["tool_calls"], "Tool-call count changed")
        for key, value in [("requests", checks), ("events", events)]:
            require_equal(run["audit"][key]["passed"], value["passed"], "Execution audit changed")
        gates += len(run.get("scripted_feedback", []))
        humans += run.get("human_interventions", 0)
        if cycle and cycle["status"] == "completed":
            declaration = spec(cycle["phase"])
            domain = audit_events(
                execution, cycle["phase"], run, declaration["required_tools"], declaration["recoverable_tools"]
            )
            require_equal(domain["passed"], True, "Completed episode has no successful domain-tool chain")
            if MODES[condition] == "science_expert":
                require_equal(run.get("four_role_path_completed"), True, "Expert roles missing")
                require_equal(run.get("finalizer_succeeded"), True, "Expert final report missing")
            elif run.get("four_role_path_completed", False):
                raise ValueError("Single-agent condition unexpectedly executed expert-team architecture")
    for key, value in charged.items():
        require_equal(value, state["usage"][key], "Cumulative charged accounting changed")
        require_equal(value, result["usage"][key], "Reported accounting changed")
    require_equal(external, result["external_scripted_turns"], "External turn count changed")
    require_equal(gates, result["internal_scripted_gates"], "Internal confirmation count changed")
    require_equal(humans, result["human_interventions"], "Human intervention count changed")
    require_equal(humans, 0, "Unexpected human help")
    verified = False
    replay = None
    if result["status"] == "completed":
        if unknown:
            raise ValueError(
                "Completed allocation contains conservatively charged unverified episodes; inspect before efficacy analysis"
            )
        replay_complete_goal(SimpleNamespace(goal=folder, output=output, source_cache=source_cache))
        replay = read(output / "report.json")
        require_equal(replay["future_metrics"], result["future_metrics"], "Reported future score changed")
        require_equal(result["implementation_unchanged"], True, "Runtime source drift")
        verified = True
    return {
        **row,
        "status": result["status"],
        "verified_success": verified,
        "budget_charged": state["usage"],
        "observed_model_calls": observed_models,
        "observed_tool_calls": observed_tools,
        "unknown_cost_episodes": unknown,
        "external_scripted_turns": external,
        "internal_scripted_gates": gates,
        "human_interventions": humans,
        "future_metrics": replay["future_metrics"] if replay else None,
        "allocation_result_sha256": file_hash(result_path),
        "replay_report_sha256": file_hash(output / "report.json") if replay else None,
    }


def archived_protocol(root):
    """Terminal evidence remains auditable after a versioned runtime repair."""
    protocol = read(root / "protocol.json")
    require_equal(digest(protocol["design"]), protocol["design_sha256"], "Comparison design changed")
    for name, expected in protocol["implementation"].items():
        require_equal(file_hash(root / "implementation" / name), expected, "Archived comparison source changed")
    for row in protocol["design"]["schedule"]:
        folder = root / "arms" / row["id"]
        require_equal(
            file_hash(folder / "protocol.json"), protocol["arm_protocols"][row["id"]], "Allocation protocol changed"
        )
        require_equal(
            GoalStore(folder).read()["definition_sha256"],
            protocol["goal_definitions"][row["id"]],
            "Goal definition changed",
        )
    return protocol


def audit(args):
    root_report = read(args.run / "report.json")
    # Running partial audits retain the strict live-source freeze. Completed
    # experiments are bound to their archived implementation, not today's tree.
    protocol = archived_protocol(args.run) if root_report["status"] == "completed" else verify_protocol(args.run)
    if root_report["status"] != "completed" and not args.allow_partial:
        raise ValueError("Campaign is not terminal; use --allow-partial only for an explicitly incomplete report")
    args.output.mkdir(parents=True, exist_ok=False)
    design = protocol["design"]
    profiles = [{k: v for k, v in p.items() if k != "mode"} for p in design["profiles"].values()]
    require_equal(profiles, [profiles[0]] * len(CONDITIONS), "Non-mode profile settings differ")
    rows = []
    for row in design["schedule"]:
        result = audit_allocation(args.run, row, args.output / row["id"], args.source_cache)
        rows.append(result)
    summaries = []
    for case in design["cases"]:
        for condition in CONDITIONS:
            cells = [r for r in rows if r["case"] == case and r["condition"] == condition]
            summaries.append(
                {
                    "case": case,
                    "condition": condition,
                    "planned": len(cells),
                    "verified_successes": sum(r["verified_success"] for r in cells),
                    "statuses": dict(Counter(r["status"] for r in cells)),
                    "future_metrics_conditional_on_success": [
                        r["future_metrics"] for r in cells if r["verified_success"]
                    ],
                    "call_costs_including_failed_cells": [
                        {
                            "repeat": r["repeat"],
                            "status": r["status"],
                            "charged": r.get("budget_charged"),
                            "observed_models": r.get("observed_model_calls"),
                            "observed_tools": r.get("observed_tool_calls"),
                            "unknown_cost_episodes": r.get("unknown_cost_episodes"),
                        }
                        for r in cells
                    ],
                }
            )
    report = {
        "status": "partial_audit" if root_report["status"] != "completed" else "audit_completed",
        "design_sha256": protocol["design_sha256"],
        "protocol_sha256": file_hash(args.run / "protocol.json"),
        "auditor_sha256": file_hash(Path(__file__)),
        "current_runtime_changes_since_snapshot": [
            name
            for name, expected in protocol["implementation"].items()
            if not (ROOT / name).exists() or file_hash(ROOT / name) != expected
        ],
        "implementation_binding": "Archived experiment sources; replay uses current host evaluator, whose hashes are recorded separately",
        "replay_sources": {
            name: file_hash(ROOT / name) for name in protocol["implementation"] if (ROOT / name).exists()
        },
        "all_planned_allocations_retained": len(rows) == len(design["schedule"]),
        "matched_profiles_verified": True,
        "rows": rows,
        "summary": summaries,
        "scope": design["scope"],
        "statistical_note": "Five cold-start repeats per reused protein partition estimate execution variability. No ranking inferred from a partial campaign; no biological replication or cross-family generalization claim.",
    }
    write_json(args.output / "report.json", report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "planned": len(rows),
                "verified_successes": sum(r["verified_success"] for r in rows),
            }
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--source-cache", type=Path, default=ROOT / "research_runs/rsi/20261011_paper_case_audit")
    p.add_argument("--allow-partial", action="store_true")
    audit(p.parse_args())
