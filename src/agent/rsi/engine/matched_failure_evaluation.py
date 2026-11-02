"""Evaluate closed matched studies while retaining unpublished failures."""

import datetime
import hashlib
import json
from pathlib import Path

from agent.rsi import (
    context_baselines,
    context_controls,
    context_v2_controls,
    generated_context,
    generated_context_v2,
    model_tasks,
    prediction_metrics,
    reference_thermal_baselines,
    shared_context_evaluation,
    shared_sequence_evaluation,
)
from .matched_failure_closeout import verify
from .matched_history_analysis import require, summarize
from .matched_history_evidence import Evidence, unit_evidence, verify_task
from agent.rsi.campaigns.matched_history_operations import collect_unit, summarize_operations
from .protein_model import digest


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def failed_unit_evidence(evidence, unit, plan, paths, closeout):
    """There is no successful reuse receipt when no model was published."""
    require(set(paths) == {"evaluation"}, "Unpublished failure cannot claim model reuse")
    folder = Path(unit["plan"]).parent
    path = evidence.path(paths["evaluation"]["path"], inside=folder)
    require(path.name == "report.json", "Explicit report.json path required")
    report = evidence.read(path, paths["evaluation"]["sha256"])
    require(
        report["schema"] == shared_context_evaluation.SCHEMA
        and report["status"] == "unpublished_failure_controls_scored_pending_replay"
        and report["final_labels_scored"] is True,
        "Failed-unit control evaluation is incomplete",
    )
    protocol = evidence.read(path.parent / "protocol.json", report["protocol_sha256"])
    require(protocol["terminal_closeout"] == closeout, "Failed evaluation closeout differs")
    require(
        protocol["fit_partition"] == "training"
        and protocol["score_partition"] == "future"
        and protocol["hyperparameter_search"] is False,
        "Failed evaluation partitions differ",
    )
    evidence.check_bindings(protocol["source_bindings"])
    evidence.check_bindings(protocol["implementation"])
    tasks = {t["id"]: t for t in plan["batches"][0]["tasks"]}
    require(
        protocol["task_bindings"] == {k: digest(t) for k, t in tasks.items()}
        and len(protocol["tasks"]) == len(tasks)
        and {t["id"]: t for t in protocol["tasks"]} == tasks
        and set(report["task_reports"]) == set(tasks),
        "Failed task coverage differs",
    )
    rows, refits = [], []
    for tid, task in tasks.items():
        row = evidence.read(path.parent / (tid + ".json"), report["task_reports"][tid])
        require(row["task_id"] == tid and row["comparison"] is None, "Failed task comparison differs")
        refits.append(verify_task(evidence, task, row, None))
        rows.append(row)
    require(
        report["totals"] == failure_totals(rows),
        "Failed-unit totals differ",
    )
    return rows, refits


def failure_totals(rows):
    require(
        all(r["scientist"] == {"status": "not_registered", "metrics": None, "predictions": None} for r in rows),
        "Unpublished checkpoint cannot receive Scientist predictions",
    )
    return {
        "allocated": len(rows),
        "scored_scientist_models": 0,
        "not_registered": len(rows),
        "positive_final_skill": 0,
        "beats_best_control": 0,
        "ties_best_control": 0,
        "completed_control_fits": sum(c["status"] == "completed" for r in rows for c in r["controls"]),
        "failed_control_fits": sum(c["status"] != "completed" for r in rows for c in r["controls"]),
    }


def score_unpublished(study, closeout, unit_id, output, *, repository):
    """Score fixed controls only, after the entire study has closed acquisition."""
    facts, design, plans, _ = verify(study, closeout, repository=repository)
    require(facts["units"][unit_id]["classification"] == "unpublished_failure", "Not an unpublished failure")
    unit = next(u for u in design["units"] if u["id"] == unit_id)
    evidence = Evidence(repository)
    folder = Path(unit["plan"]).parent
    output = evidence.path(output, inside=folder)
    require(output != folder and not output.exists(), "Evaluation output must be a new child directory")
    evidence.check_bindings(facts["source_bindings"])
    evidence.read_bytes(closeout["path"], closeout["sha256"])
    snapshot = evidence.read(Path(study) / "snapshot.json")["files"]
    implementations = {}
    for module in (
        context_baselines,
        context_controls,
        context_v2_controls,
        generated_context,
        generated_context_v2,
        model_tasks,
        prediction_metrics,
        reference_thermal_baselines,
        shared_context_evaluation,
        shared_sequence_evaluation,
    ):
        path = evidence.path(module.__file__)
        relative = "src/" + module.__name__.replace(".", "/") + ".py"
        evidence.read_bytes(path, snapshot[relative])
        implementations[str(path)] = sha(path)
    implementations[str(Path(__file__))] = sha(Path(__file__))
    tasks = plans[unit_id]["batches"][0]["tasks"]
    for task in tasks:
        for reference in [*task["data"].values(), *task["references"]]:
            evidence.read_bytes(reference["path"], reference["sha256"])
    output.mkdir()
    protocol = {
        "schema": shared_context_evaluation.SCHEMA,
        "utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "terminal_closeout": closeout,
        "classification": "unpublished_failure",
        "fit_partition": "training",
        "score_partition": "future",
        "hyperparameter_search": False,
        "task_bindings": {t["id"]: digest(t) for t in tasks},
        "tasks": tasks,
        "source_bindings": dict(evidence.bindings),
        "implementation": implementations,
        "policy": "All allocated tasks remain in the denominator. Only frozen controls are fitted; unpublished checkpoints have no Scientist score or reuse receipt.",
        "external_model_calls": 0,
        "new_autonomous_models": 0,
    }
    write(output / "protocol.json", protocol)
    rows = []
    for task in tasks:
        row = shared_context_evaluation.evaluate_task(task, None)
        write(output / (task["id"] + ".json"), row)
        rows.append(row)
    evidence.check_bindings(implementations)
    evidence.unchanged()
    verify(study, closeout, repository=repository)
    report = {
        "schema": shared_context_evaluation.SCHEMA,
        "status": "unpublished_failure_controls_scored_pending_replay",
        "protocol_sha256": sha(output / "protocol.json"),
        "totals": failure_totals(rows),
        "task_reports": {r["task_id"]: sha(output / (r["task_id"] + ".json")) for r in rows},
        "final_labels_scored": True,
        "source_goals_and_prior_evidence_unchanged": True,
        "external_model_calls": 0,
        "new_autonomous_models": 0,
    }
    write(output / "report.json", report)
    return report


def analyze(study, manifest, *, repository):
    """Run the same paired analysis with explicit zero-publication failures."""
    require(
        manifest["schema"] == "matched_context_failure_analysis_input.v1"
        and manifest["study"] == str(Path(study).absolute()),
        "Failure-aware manifest differs",
    )
    closeout = manifest["terminal_closeout"]
    facts, design, plans, contexts = verify(study, closeout, repository=repository)
    require(set(manifest["units"]) == set(plans), "All assigned units are required")
    evidence = Evidence(repository)
    evidence.check_bindings(facts["source_bindings"])
    evidence.read_bytes(closeout["path"], closeout["sha256"])
    rows, refits, operations = {}, {}, {}
    for unit in design["units"]:
        uid = unit["id"]
        if facts["units"][uid]["classification"] == "unpublished_failure":
            rows[uid], refits[uid] = failed_unit_evidence(evidence, unit, plans[uid], manifest["units"][uid], closeout)
        else:
            rows[uid], refits[uid] = unit_evidence(evidence, unit, plans[uid], contexts[uid], manifest["units"][uid])
        operations[uid] = collect_unit(evidence, plans[uid], contexts[uid])
        require(
            operations[uid]["registered"] == sum(r["scientist"]["status"] == "scored" for r in rows[uid]),
            "Operational and scored publications differ",
        )
    report = summarize(design, plans, rows)
    report["operational_analysis"] = summarize_operations(design, operations)
    evidence.unchanged()
    verify(study, closeout, repository=repository)
    report.update(
        {
            "status": "matched_failure_analysis_and_independent_metric_control_replay_passed",
            "terminal_closeout": closeout,
            "unpublished_failures": {
                k: v for k, v in facts["units"].items() if v["classification"] == "unpublished_failure"
            },
            "source_bindings": evidence.bindings,
            "control_refits": refits,
            "final_labels_scored": True,
            "domain_baseline_comparison_complete": False,
        }
    )
    return report
