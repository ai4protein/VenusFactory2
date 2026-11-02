"""Evaluate a terminal shared campaign after independent registered-model reuse."""

import argparse
import datetime
import json
from pathlib import Path

from agent.rsi import (
    context_baselines,
    context_controls,
    context_v2_controls,
    context_v3_controls,
    generated_context,
    generated_context_v2,
    generated_context_v3,
    model_tasks,
    prediction_metrics,
    reference_thermal_baselines,
    shared_context_evaluation,
    shared_sequence_evaluation,
)
from agent.rsi.protein_model import digest
from agent.rsi.scientist_model_campaign import process_live
from agent.rsi.shared_model_campaign import SharedCampaignStore
from agent.rsi.shared_sequence_evaluation import require_terminal
from agent.rsi.storage import sha256_file


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def prepare(args):
    experiment, reuse = args.experiment.absolute(), args.reuse.absolute()
    terminal_path, report_path = experiment / "terminal_summary_v1.json", reuse / "report.json"
    if (
        sha256_file(terminal_path) != args.terminal_summary_sha256
        or sha256_file(report_path) != args.reuse_report_sha256
    ):
        raise ValueError("Frozen acquisition or reuse evidence changed")
    terminal, report = read(terminal_path), read(report_path)
    identity = read(experiment / "runner_identity.json")
    store = SharedCampaignStore(Path(identity["campaign"]))
    state = store.read()
    require_terminal(state, runner_live=process_live(identity))
    if (
        sha256_file(store.root / "campaign.json") != terminal["campaign_sha256"]
        or state["last_audit"] != terminal["audit_id"]
    ):
        raise ValueError("Acquisition changed after its terminal audit")
    audit_path = store.root / "audits" / (state["last_audit"] + ".json")
    if sha256_file(audit_path) != terminal["audit_sha256"] or digest(read(audit_path)) != state["last_audit"]:
        raise ValueError("Publication audit changed")
    if report["status"] != "independent_api_reuse_passed" or read(reuse / "exit.json")["returncode"] != 0:
        raise ValueError("Independent model reuse has not passed")
    if (
        sha256_file(reuse / "protocol.json") != report["protocol_sha256"]
        or sha256_file(reuse / "inference_results.json") != report["results_sha256"]
    ):
        raise ValueError("Independent reuse inputs or predictions changed")
    receipt_audit = read(reuse / "independent_receipt_audit.json")
    if receipt_audit["status"] != "passed" or receipt_audit["report_sha256"] != args.reuse_report_sha256:
        raise ValueError("Independent receipt audit missing or stale")
    protocol = read(reuse / "protocol.json")
    bindings = {**protocol["source_bindings"], **receipt_audit["checked_files"]}
    for path in [
        terminal_path,
        report_path,
        audit_path,
        reuse / "protocol.json",
        reuse / "inference_results.json",
        reuse / "independent_receipt_audit.json",
    ]:
        bindings[str(path)] = sha256_file(path)
    if any(sha256_file(Path(p)) != h for p, h in bindings.items()):
        raise ValueError("A frozen source or registered-model receipt changed")
    return experiment, store, state, protocol, bindings


def summarize(rows):
    acquired = [r for r in rows if r["scientist"]["status"] == "scored"]
    return {
        "allocated": len(rows),
        "scored_scientist_models": len(acquired),
        "not_registered": len(rows) - len(acquired),
        "positive_final_skill": sum(
            (r["scientist"]["metrics"]["mse_skill_vs_training_mean"] or 0) > 0 for r in acquired
        ),
        "beats_best_control": sum(r["comparison"]["scientist_beats_best_control"] for r in acquired),
        "ties_best_control": sum(r["comparison"]["scientist_ties_best_control"] for r in acquired),
        "completed_control_fits": sum(c["status"] == "completed" for r in rows for c in r["controls"]),
        "failed_control_fits": sum(c["status"] != "completed" for r in rows for c in r["controls"]),
    }


def main(args):
    output = args.output.absolute()
    if output.exists() or output == args.experiment.absolute() or not output.is_relative_to(args.experiment.absolute()):
        raise ValueError("Evaluation requires a new directory inside the experiment")
    if any(p.is_symlink() for p in [output, *output.parents]):
        raise ValueError("Symlinked output paths are unsupported")
    experiment, store, state, reuse, bindings = prepare(args)
    tasks, published = {}, {}
    for name, batch in store.specifications(state).items():
        for task in batch["tasks"]:
            if task["id"] in tasks or task["plugin_id"] not in {
                "generated_context.v1",
                "generated_context.v2",
                "generated_context.v3",
            }:
                raise ValueError("Distinct contextual tasks required")
            tasks[task["id"]] = task
            for ref in [*task["data"].values(), *task["references"]]:
                if sha256_file(Path(ref["path"])) != ref["sha256"]:
                    raise ValueError("Bound task data or provenance changed")
                bindings[ref["path"]] = ref["sha256"]
        for attempt in state["batches"][name]["attempts"]:
            for row in (attempt.get("publication") or {}).get("outcomes", []):
                if row["task_id"] in published:
                    raise ValueError("Repeated task publication")
                published[row["task_id"]] = row["entry_id"]
    expected = {r["task_id"]: r for r in reuse["task_bindings"]}
    predictions = {
        r["receipt"]["entry_id"]: r["predictions"] for r in read(args.reuse / "inference_results.json")["results"]
    }
    if published != {k: r["entry_id"] for k, r in expected.items()} or set(published.values()) != set(predictions):
        raise ValueError("Independent inference must cover every published model exactly")
    implementation = {
        str(Path(m.__file__)): sha256_file(Path(m.__file__))
        for m in [
            context_baselines,
            context_controls,
            context_v2_controls,
            context_v3_controls,
            generated_context,
            generated_context_v2,
            generated_context_v3,
            model_tasks,
            prediction_metrics,
            reference_thermal_baselines,
            shared_context_evaluation,
            shared_sequence_evaluation,
        ]
    }
    implementation[str(Path(__file__))] = sha256_file(Path(__file__))
    protocol = {
        "schema": shared_context_evaluation.SCHEMA,
        "utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "design": "Descriptive comparison fixed after acquisition and before this final evaluation; not a prospective or randomized memory comparison.",
        "fit_partition": "training",
        "score_partition": "future",
        "hyperparameter_search": False,
        "score_partition_meaning": "Held-out experimental groups; not necessarily a future calendar period.",
        "task_bindings": {k: digest(t) for k, t in tasks.items()},
        "tasks": list(tasks.values()),
        "source_bindings": bindings,
        "implementation": implementation,
        "external_model_calls": 0,
        "new_autonomous_models": 0,
        "statistical_independence_assumed": False,
    }
    output.mkdir()
    write(output / "protocol.json", protocol)
    rows = []
    for number, (task_id, task) in enumerate(tasks.items(), 1):
        result = shared_context_evaluation.evaluate_task(task, predictions.get(published.get(task_id)))
        if task_id in expected and (
            result["query_sha256"] != expected[task_id]["input_sha256"]
            or digest(result["scientist"]["predictions"]) != expected[task_id]["prediction_sha256"]
        ):
            raise ValueError("Scored predictions differ from independently reused outputs")
        write(output / (task_id + ".json"), result)
        rows.append(result)
        if number % 16 == 0:
            print(json.dumps({"completed": number, "allocated": len(tasks)}), flush=True)
    if any(sha256_file(Path(p)) != h for p, h in {**bindings, **implementation}.items()):
        raise ValueError("Source evidence or evaluator changed during final scoring")
    conditions = sorted({r["scope"]["context"].get("growth_condition", "not_declared") for r in rows})
    report = {
        "schema": shared_context_evaluation.SCHEMA,
        "status": "scored_pending_independent_replay",
        "utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "protocol_sha256": sha256_file(output / "protocol.json"),
        "totals": summarize(rows),
        "by_condition": {
            c: summarize([r for r in rows if r["scope"]["context"].get("growth_condition", "not_declared") == c])
            for c in conditions
        },
        "task_reports": {r["task_id"]: sha256_file(output / (r["task_id"] + ".json")) for r in rows},
        "source_goals_and_prior_evidence_unchanged": True,
        "final_labels_scored": True,
        "statistical_independence_assumed": False,
        "domain_baseline_comparison_complete": False,
        "external_model_calls": 0,
        "new_autonomous_models": 0,
    }
    write(output / "report.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "task_reports"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--reuse", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--terminal-summary-sha256", required=True)
    parser.add_argument("--reuse-report-sha256", required=True)
    main(parser.parse_args())
