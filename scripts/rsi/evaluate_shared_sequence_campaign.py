#!/usr/bin/env python
"""Score a sealed shared sequence campaign against its frozen comparison plan."""

from __future__ import annotations

import argparse
import collections
import datetime
import json
from pathlib import Path

from agent.rsi import position_baselines, shared_sequence_evaluation, traditional_baselines
from agent.rsi.protein_model import digest
from agent.rsi.scientist_model_campaign import process_live
from agent.rsi.shared_model_campaign import SharedCampaignStore, task_scope
from agent.rsi.shared_sequence_evaluation import evaluate_task, require_terminal
from agent.rsi.storage import atomic_json, sha256_file


def read(path):
    return json.loads(path.read_text())


def summarize(rows):
    acquired = [row for row in rows if row["scientist"]["status"] == "scored"]
    return {
        "allocated": len(rows),
        "scored_scientist_models": len(acquired),
        "not_acquired": len(rows) - len(acquired),
        "positive_final_skill": sum(
            (row["scientist"]["metrics"]["mse_skill_vs_training_mean"] or 0) > 0 for row in acquired
        ),
        "beats_best_control": sum(row["comparison"]["scientist_beats_best_control"] for row in acquired),
        "ties_best_control": sum(row["comparison"]["scientist_ties_best_control"] for row in acquired),
        "control_completion": dict(
            collections.Counter(c["method"] for row in rows for c in row["controls"] if c["status"] == "completed")
        ),
        "control_unsupported": dict(
            collections.Counter(c["method"] for row in rows for c in row["controls"] if c["status"] == "unsupported")
        ),
    }


def main(args):
    experiment, output = args.experiment.absolute(), args.output.absolute()
    if not output.is_relative_to(experiment) or output == experiment or output.exists():
        raise ValueError("Evaluation needs a new directory inside the source experiment")
    if any(p.is_symlink() for p in [experiment, output, *output.parents]):
        raise ValueError("Symlinked evaluation paths are not supported")
    summary_path = experiment / "terminal_summary_v1.json"
    plan_path = experiment / "classical_comparison_plan.json"
    if (
        sha256_file(summary_path) != args.terminal_summary_sha256
        or sha256_file(plan_path) != args.comparison_plan_sha256
    ):
        raise ValueError("Approved terminal evidence or frozen comparison plan changed")
    terminal, plan = read(summary_path), read(plan_path)
    preparation = read(experiment / "preparation.json")
    store = SharedCampaignStore(Path(preparation["campaign"]))
    state = store.read()
    require_terminal(state, runner_live=process_live(read(experiment / "runner_identity.json")))
    if terminal["status"] != "terminal_audit_and_agent_reuse_passed" or not terminal["source_goals_unchanged"]:
        raise ValueError("Independent Agent reuse has not been verified")
    campaign_path = store.root / "campaign.json"
    if sha256_file(campaign_path) != terminal["campaign_sha256"] or state["last_audit"] != terminal["audit_id"]:
        raise ValueError("Acquisition changed after its independent audit")
    audit_path = store.root / "audits" / (state["last_audit"] + ".json")
    audit = read(audit_path)
    if digest(audit) != state["last_audit"] or sha256_file(audit_path) != terminal["audit_sha256"]:
        raise ValueError("Immutable publication audit changed")
    if not audit["all_batches_terminal"]:
        raise ValueError("Incomplete acquisition barrier")
    if (
        plan["fit_partition"] != "training"
        or plan["score_partition"] != "future_only_after_all_goals_terminal"
        or not plan["no_hyperparameter_search"]
    ):
        raise ValueError("Unsupported frozen comparison protocol")
    if plan["common_methods"] != list(traditional_baselines.estimators()) or plan["positional_methods"] != list(
        shared_sequence_evaluation.POSITIONAL_METHODS.values()
    ):
        raise ValueError("Predeclared control methods changed")
    sources = {
        "src/agent/rsi/traditional_baselines.py": Path(traditional_baselines.__file__),
        "src/agent/rsi/position_baselines.py": Path(position_baselines.__file__),
    }
    if {name: sha256_file(path) for name, path in sources.items()} != plan["existing_sources"]:
        raise ValueError("Predeclared control implementations changed")
    specs = store.specifications(state)
    tasks, published, bindings, goals = {}, {}, {}, {}
    terminal_batches = {row["batch"]: row for row in terminal["batches"]}
    for name, batch in specs.items():
        goal = store.goal(name)
        child = read(goal / "goal.json")
        if sha256_file(goal / "goal.json") != terminal_batches[name]["goal_sha256"]:
            raise ValueError("An acquisition goal changed after reuse audit")
        require_terminal(
            {"status": "needs_tasks", "owner": None, "inflight": None, "batches": {name: {"status": child["status"]}}},
            runner_live=False,
        )
        goals[name] = {"path": str(goal / "goal.json"), "sha256": sha256_file(goal / "goal.json")}
        for task in batch["tasks"]:
            if task["id"] in tasks or task["plugin_id"] != "generated_sequence.v1":
                raise ValueError("Repeated or nonsequence task")
            tasks[task["id"]] = task
        outcomes = state["batches"][name]["attempts"][0]["publication"]["outcomes"]
        for outcome in outcomes:
            task_id = outcome["task_id"]
            if task_id in published:
                raise ValueError("Repeated publication")
            record = child["progress"]["targets"][task_id]
            folder = goal / record["reuse_episode"] / "workspace/targets" / task_id
            published[task_id] = read(folder / "predictions.json")
            bindings[task_id] = {
                "entry_id": outcome["entry_id"],
                "predictions_path": str(folder / "predictions.json"),
                "predictions_sha256": sha256_file(folder / "predictions.json"),
                "query_sha256": digest(read(folder / "inputs.json")),
            }
    if (
        len({task_scope(task) for task in tasks.values()}) != len(tasks)
        or len(published) != terminal["registered_scopes"]
    ):
        raise ValueError("Task or publication coverage changed")
    if {row["entry_id"] for row in terminal["agent_inferences"]} != {row["entry_id"] for row in bindings.values()}:
        raise ValueError("Independent Agent checks do not cover all scored models")
    output.mkdir()
    implementation = {
        str(path): sha256_file(path)
        for path in [Path(__file__), Path(shared_sequence_evaluation.__file__), *sources.values()]
    }
    manifest = {
        "schema": shared_sequence_evaluation.SCHEMA,
        "utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "experiment": str(experiment),
        "campaign_sha256": sha256_file(campaign_path),
        "terminal_summary_sha256": args.terminal_summary_sha256,
        "comparison_plan_sha256": args.comparison_plan_sha256,
        "publication_audit": state["last_audit"],
        "goals": goals,
        "scientist_prediction_bindings": bindings,
        "task_bindings": {key: digest(task) for key, task in tasks.items()},
        "implementation": implementation,
        "final_label_access_begins_after_this_manifest": True,
        "fit_partition": "training",
        "score_partition": "future",
        "hyperparameter_search": False,
        "external_model_calls": 0,
    }
    atomic_json(output / "manifest.json", manifest)
    rows = []
    for index, (task_id, task) in enumerate(tasks.items(), 1):
        result = evaluate_task(task, published.get(task_id))
        if task_id in bindings and result["query_sha256"] != bindings[task_id]["query_sha256"]:
            raise ValueError("Final inputs differ from independently reused inputs")
        atomic_json(output / (task_id + ".json"), result)
        rows.append(result)
        print(
            json.dumps(
                {
                    "completed": index,
                    "allocated": len(tasks),
                    "task_id": task_id,
                    "scientist": result["scientist"]["status"],
                }
            ),
            flush=True,
        )
    if sha256_file(campaign_path) != manifest["campaign_sha256"] or any(
        sha256_file(Path(row["path"])) != row["sha256"] for row in goals.values()
    ):
        raise ValueError("Acquisition changed during final scoring")
    report = {
        "status": "scored_pending_independent_replay",
        "schema": shared_sequence_evaluation.SCHEMA,
        "utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "manifest_sha256": sha256_file(output / "manifest.json"),
        "totals": summarize(rows),
        "by_stage": {
            stage: summarize([row for row in rows if row["stage"] == stage])
            for stage in sorted({row["stage"] for row in rows})
        },
        "task_reports": {row["task_id"]: sha256_file(output / (row["task_id"] + ".json")) for row in rows},
        "distinct_source_programs": terminal["distinct_published_source_programs"],
        "final_labels_scored": True,
        "source_goals_unchanged": True,
        "external_model_calls": 0,
        "no_cross_property_raw_rmse_aggregation": True,
        "statistical_independence_assumed": False,
        "domain_baseline_comparison_complete": False,
    }
    atomic_json(output / "report.json", report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--terminal-summary-sha256", required=True)
    parser.add_argument("--comparison-plan-sha256", required=True)
    main(parser.parse_args())
