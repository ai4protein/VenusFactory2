"""Evaluate a frozen ProtPredict Scientist pilot and its predeclared XGBoost control.

This controller runs outside the frozen acquisition snapshot. It imports only that
snapshot's scientific evaluator, and uses the prospectively bound worker, runtime
and reference-protein parameters. It never contacts an external model service.
"""

# Direct-script entry point; implementation imports remain package-qualified.
import sys as _entry_sys
from pathlib import Path as _EntryPath
_entry_root = _EntryPath(__file__).resolve().parents[3]
_entry_sys.path[:0] = [str(_entry_root), str(_entry_root / 'src'), str(_entry_root / 'scripts/rsi')]


import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

METHODS = (
    "training_mean",
    "condition_ridge",
    "condition_random_forest",
    "condition_svr",
    "protpredict_shared_xgboost",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def path_without_links(path):
    path = Path(path).absolute()
    require(not any(p.is_symlink() for p in (path, *path.parents)), "Symlinked evaluation path")
    return path


def read(path):
    path = path_without_links(path)
    require(path.is_file(), "Missing evaluation evidence")
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path_without_links(path).read_bytes()).hexdigest()


def bound(reference):
    require(isinstance(reference, dict) and set(reference) == {"path", "sha256"}, "Invalid evidence binding")
    path = path_without_links(reference["path"])
    require(sha(path) == reference["sha256"], "Bound evidence changed")
    return path


def write(path, value):
    path = path_without_links(path)
    temporary = path.with_name(path.name + ".tmp")
    require(not temporary.exists(), "Unresolved evaluation write")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def verify_selected(definition, protocol):
    selected = protocol["selection"]["selected_tasks"]
    require(len(selected) == len(set(selected)) == 4, "Four distinct selected tasks required")
    allocations = definition["allocations"]
    require(not definition["pairs"] and len(allocations) == 4, "Four single-arm allocations required")
    observed = [a["task"]["id"] for a in allocations.values()]
    require(sorted(observed) == sorted(selected), "Selected tasks changed")
    for allocation in allocations.values():
        task = allocation["task"]
        require(task["plugin_id"] == "generated_context.v2", "Explicit version-2 task required")
        require(len(task["scope"]["covariates"]) == 19, "All 19 covariates required")
        require(task["scope"]["grouping"]["level"] == "deleted_gene", "Deleted-genotype grouping required")
        require(
            task["scope"]["context"]["organism"] == "Saccharomyces cerevisiae, laboratory S288c-derived strains",
            "Only the declared laboratory-yeast study is supported",
        )


def worker_request(training, scoring, scope, parameters):
    require(len(scope["covariates"]) == 19, "All 19 measured features required")
    require(len(training) == 600 and len(scoring) == 200, "Frozen 600/200 comparison required")
    require(
        len({r["id"] for r in training}) == 600 and len({r["id"] for r in scoring}) == 200,
        "Repeated observation identity",
    )
    require(
        len({r["group"] for r in training}) == 600 and len({r["group"] for r in scoring}) == 200,
        "One row per genotype required",
    )
    require(not {r["id"] for r in training}.intersection(r["id"] for r in scoring), "Training/scoring identity overlap")
    require(
        not {r["group"] for r in training}.intersection(r["group"] for r in scoring),
        "Training/scoring genotype overlap",
    )
    require(
        all(r["protein"] == scope["protein"] and len(r["values"]) == 19 for r in training + scoring),
        "Protein or feature membership changed",
    )
    return {
        "schema": "protpredict_xgb_request.v1",
        "action": "fit_predict",
        "feature_names": [x["name"] for x in scope["covariates"]],
        "training": {
            "x": [r["values"] for r in training],
            "y": [r["target"] for r in training],
            "groups": [r["group"] for r in training],
        },
        "parameters": parameters,
        "scoring": {"x": [r["values"] for r in scoring], "ids": [r["id"] for r in scoring]},
    }


def prediction_witness(scoring, predictions, digest):
    require([r["id"] for r in predictions] == [r["id"] for r in scoring], "Prediction identities or order changed")
    require(
        all(
            set(r) == {"id", "prediction"} and type(r["prediction"]) in (int, float) and math.isfinite(r["prediction"])
            for r in predictions
        ),
        "Invalid prediction",
    )
    return [
        {
            "id": row["id"],
            "protein": row["protein"],
            "group": row["group"],
            "values_sha256": digest(row["values"]),
            "prediction": pred["prediction"],
        }
        for row, pred in zip(scoring, predictions, strict=True)
    ]


def require_terminal(definition, state, execution_policy, allocation_execution, all_allocations):
    require(
        state["definition_sha256"] == definition["campaign_definition_sha256"]
        and state["plans"] == definition["plans"],
        "Campaign design changed",
    )
    require(state["owner"] is None and state["inflight"] is None, "Acquisition is unresolved")
    require(
        execution_policy(state["definition"]) == definition["execution_policy"] == all_allocations,
        "All-allocation execution required",
    )
    require(allocation_execution(state)["execution_complete"], "Every allocation must terminate before scoring")


def load_backend(pilot):
    repo = pilot / "repository"
    # Avoid mixing current packages with the historical numerical implementation.
    require("agent" not in sys.modules, "Run the evaluator in a fresh Python process")
    sys.path.insert(0, str(repo / "src"))
    from agent.rsi import campaign_evaluation as core
    from agent.rsi import model_tasks
    from agent.rsi.engine.prediction_metrics import measure_context
    from agent.rsi.engine.protein_model import digest
    from agent.rsi.campaigns.scientist_model_campaign import (
        ALL_ALLOCATIONS,
        ModelCampaignStore,
        allocation_execution,
        execution_policy,
    )

    require(Path(core.__file__).absolute().is_relative_to(repo), "Wrong scientific evaluator")
    return (
        core,
        model_tasks,
        measure_context,
        digest,
        ModelCampaignStore,
        allocation_execution,
        execution_policy,
        ALL_ALLOCATIONS,
    )


def load_binding(pilot, core):
    prep = read(pilot / "preparation.json")
    require(
        sha(pilot / "plan.json") == prep["plan_sha256"] and sha(pilot / "protocol.json") == prep["protocol_sha256"],
        "Prepared protocol changed",
    )
    protocol = read(pilot / "protocol.json")
    require(
        protocol["schema"] == "protpredict_scientist_pilot.v1" and tuple(protocol["final_controls"]) == METHODS,
        "Declared comparator membership changed",
    )
    for key in (
        "baseline_protocol",
        "batch_provenance",
        "xgboost_parameters",
        "xgboost_tuning_report",
        "xgboost_tuning_audit",
        "training_preflight",
    ):
        bound(protocol[key])
    require(read(bound(protocol["training_preflight"]))["status"] == "passed", "Training preflight did not pass")
    require(read(bound(protocol["xgboost_tuning_audit"]))["status"] == "passed", "Tuning replay did not pass")
    baseline = read(bound(protocol["baseline_protocol"]))
    for key in ("runtime", "wheel_hashes", "batch_provenance"):
        bound(baseline[key])
    for reference in baseline["source_files"].values():
        bound(reference)
    require(baseline["selection"] == protocol["selection"], "Baseline task selection changed")
    tuning = read(bound(protocol["xgboost_tuning_report"]))
    require(
        tuning["status"] == "completed" and tuning["search"] == {"iterations": 50, "folds": 5},
        "Incomplete source-declared search",
    )
    require(tuning["parameters"]["sha256"] == protocol["xgboost_parameters"]["sha256"], "Parameter transfer changed")
    snapshot = read(pilot / "snapshot.json")["files"]
    require(snapshot == core.source_identity(), "Use the frozen scientific implementation")
    require(all(sha(pilot / "repository" / name) == value for name, value in snapshot.items()), "Frozen source changed")
    worker = pilot / "repository/scripts/rsi/protpredict_xgb_worker.py"
    require(sha(worker) == baseline["worker"]["sha256"] == protocol["xgboost_worker_sha256"], "Baseline worker changed")
    runtime = path_without_links(protocol["xgboost_runtime"])
    require(
        runtime == bound(protocol["baseline_protocol"]).parent / "runtime/bin/python" and runtime.is_file(),
        "Use the declared isolated runtime",
    )
    definition = core.load_study(pilot / "evaluation_study")
    require(
        definition["campaign"] == prep["campaign"] and prep["study"] == str(pilot / "evaluation_study"),
        "Study location changed",
    )
    verify_selected(definition, protocol)
    return definition, protocol, worker, runtime, tuning


def evaluate(pilot, output, backend):
    core, model_tasks, measure_context, digest, Store, allocation_execution, execution_policy, all_allocations = backend
    definition, protocol, worker, runtime, tuning = load_binding(pilot, core)
    store = Store(definition["campaign"])
    state = store.read()
    require_terminal(definition, state, execution_policy, allocation_execution, all_allocations)
    output = path_without_links(output)
    require(
        pilot in output.parents and pilot / "repository" not in output.parents,
        "Output must be in a new evidence directory outside the frozen repository",
    )
    require(not output.exists(), "Evaluation output already exists")
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "schema": "protpredict_pilot_evaluation.v1",
        "status": "running",
        "controller_sha256": sha(Path(__file__)),
        "protocol_sha256": sha(pilot / "protocol.json"),
        "parameters_sha256": protocol["xgboost_parameters"]["sha256"],
        "rows": [],
        "external_model_calls": 0,
        "new_autonomous_models": 0,
    }
    write(output / "report.json", report)
    try:
        original = core.evaluate(pilot / "evaluation_study", output / "core_evaluation")
        require(original["status"] == "completed", "Core evaluation did not complete")
        audited = core.audit(pilot / "evaluation_study", output / "core_evaluation", output / "core_audit")
        require(audited["status"] == "passed", "Core replay did not pass")
        params = read(bound(protocol["xgboost_parameters"]))
        env = {k: os.environ[k] for k in ("PATH", "HOME", "LANG", "LC_ALL", "TMPDIR") if k in os.environ}
        env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        for original_row in original["rows"]:
            identity = original_row["allocation_id"]
            task = definition["allocations"][identity]["task"]
            # The frozen core terminal barrier and audit have completed before any new fit.
            require(store.read() == state, "Acquisition changed after terminal scoring")
            data = model_tasks.load_task(task)
            folder = output / identity
            folder.mkdir()
            request = worker_request(data["training"], data["future"], task["scope"], params)
            write(folder / "request.json", request)
            command = [
                str(runtime),
                str(worker),
                "--request",
                str(folder / "request.json"),
                "--output",
                str(folder / "xgb"),
            ]
            with (folder / "worker.log").open("x") as log:
                cp = subprocess.run(command, cwd=pilot, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=300)
            require(cp.returncode == 0, "XGBoost comparator failed")
            computed = read(folder / "xgb/report.json")
            require(
                computed["status"] == "completed"
                and computed["request_sha256"] == sha(folder / "request.json")
                and computed["worker_sha256"] == sha(worker)
                and computed["runtime"] == tuning["runtime"],
                "XGBoost execution binding changed",
            )
            for name in ("model", "predictor", "predictions"):
                reference = computed[name]
                require(
                    reference["path"] == name + ".json"
                    and sha(folder / "xgb" / reference["path"]) == reference["sha256"],
                    "XGBoost output changed",
                )
            witness = prediction_witness(data["future"], read(folder / "xgb/predictions.json"), digest)
            metrics = measure_context(data["training"], data["future"], witness, scope=task["scope"])
            write(folder / "prediction_witness.json", witness)
            controls = original_row["context_controls"]["outcomes"]
            require(
                tuple(x["method"] for x in controls) == METHODS[:4]
                and all(x["status"] == "completed" for x in controls),
                "Generic controls incomplete",
            )
            comparative = [{"method": x["method"], "metrics": x["metrics"]} for x in controls]
            comparative.append({"method": METHODS[-1], "metrics": metrics})
            best = min(comparative, key=lambda x: (x["metrics"]["rmse"], x["method"]))
            final = original_row["final_metrics"]
            report["rows"].append(
                {
                    "allocation_id": identity,
                    "task_id": task["id"],
                    "protein": task["scope"]["protein"],
                    "original_status": original_row["original_status"],
                    "publication_and_reuse_success": original_row["publication_and_reuse_success"],
                    "published_generations": len(original_row["generations"]),
                    "final_metrics": final,
                    "controls": comparative,
                    "best_control": best["method"],
                    "final_minus_best_rmse": final["rmse"] - best["metrics"]["rmse"] if final else None,
                    "worker_report_sha256": sha(folder / "xgb/report.json"),
                    "witness_sha256": sha(folder / "prediction_witness.json"),
                }
            )
            write(output / "report.json", report)
        require(store.read() == state, "Campaign changed during comparator evaluation")
        load_binding(pilot, core)
        require(
            all(
                core.tree(store.goal(i)) == files
                for i, files in read(output / "core_evaluation/development_barrier.json")["goal_files"].items()
            ),
            "Goal evidence changed during comparator evaluation",
        )
        report.update(
            status="completed",
            allocated=4,
            unique_scopes_with_model=len({x["task_id"] for x in report["rows"] if x["final_metrics"] is not None}),
            completed_two_generation_goals=sum(x["publication_and_reuse_success"] for x in report["rows"]),
            core_report_sha256=sha(output / "core_evaluation/report.json"),
            core_audit_sha256=sha(output / "core_audit/audit.json"),
        )
    except Exception as exc:
        report.update(status="failed", error_type=type(exc).__name__)
        write(output / "report.json", report)
        raise
    write(output / "report.json", report)
    return report


def audit(pilot, evaluation, output, backend):
    output = path_without_links(output)
    require(
        pilot in output.parents and pilot / "repository" not in output.parents and not output.exists(),
        "Use a new audit directory outside the frozen repository",
    )
    output.mkdir(parents=True, exist_ok=False)
    result = {"status": "running", "external_model_calls": 0, "new_autonomous_models": 0}
    write(output / "audit.json", result)
    try:
        replay = evaluate(pilot, output / "replay", backend)
        original = read(evaluation / "report.json")
        require(original == replay and replay["status"] == "completed", "Final comparison differs from replay")
        for row in original["rows"]:
            identity = row["allocation_id"]
            for name in (
                "request.json",
                "prediction_witness.json",
                "xgb/report.json",
                "xgb/model.json",
                "xgb/predictor.json",
                "xgb/predictions.json",
            ):
                require(
                    sha(evaluation / identity / name) == sha(output / "replay" / identity / name),
                    "Comparator artifact differs from replay",
                )
        result.update(status="passed", allocations=4, report_sha256=sha(evaluation / "report.json"))
    except Exception as exc:
        result.update(status="failed", error_type=type(exc).__name__)
        write(output / "audit.json", result)
        raise
    write(output / "audit.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("evaluate", "audit"))
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path)
    args = parser.parse_args()
    pilot = path_without_links(args.pilot)
    require(args.action != "audit" or args.evaluation is not None, "Audit requires an existing evaluation")
    backend = load_backend(pilot)
    result = (
        evaluate(pilot, args.output, backend)
        if args.action == "evaluate"
        else audit(pilot, args.evaluation, args.output, backend)
    )
    print(json.dumps({k: result[k] for k in ("status", "external_model_calls", "new_autonomous_models")}))


if __name__ == "__main__":
    main()
