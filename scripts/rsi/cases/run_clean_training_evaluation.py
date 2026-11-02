"""Evaluate both fixed CLEAN split10 trained heads and independently replay results."""

# Direct-script entry point; implementation imports remain package-qualified.
import sys as _entry_sys
from pathlib import Path as _EntryPath
_entry_root = _EntryPath(__file__).resolve().parents[3]
_entry_sys.path[:0] = [str(_entry_root), str(_entry_root / 'src'), str(_entry_root / 'scripts/rsi')]


import argparse
import importlib.metadata
import json
import os
import shutil
import sys
import time
from pathlib import Path

os.environ.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1")
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from agent.rsi import clean_annotation as annotation  # noqa: E402
from agent.rsi import clean_training_evaluation as evaluation  # noqa: E402
from agent.rsi import clean_training_inputs as inputs  # noqa: E402


def run(args):
    import numpy as np
    import pandas as pd
    import torch

    workspace, output = args.workspace_root.resolve(), args.output.resolve()
    source, training = args.inputs.resolve(), args.training.resolve()
    for path in (output, source, training):
        path.relative_to(workspace / "research_runs/rsi")
        if path == workspace / "research_runs/rsi":
            raise ValueError("Dedicated research directories required")
    if os.environ.get("PYTHONHASHSEED") != "1234":
        raise ValueError("Launch with PYTHONHASHSEED=1234")
    if {k: importlib.metadata.version(k) for k in inputs.ENVIRONMENT} != inputs.ENVIRONMENT:
        raise ValueError("Pinned CLEAN environment required")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {
        "status": "running",
        "stage": "admission",
        "allocation": evaluation.ALLOCATION,
        "external_model_calls": 0,
        "training_completed": False,
        "evaluation_completed": False,
        "execution_origin": "controller_executed_author_training_and_numerically_audited_evaluation",
        "results": {},
        "failures": {},
        **evaluation.LIMITS,
    }
    inputs.save(output / "report.json", report)
    try:
        assets = workspace / "research_runs/rsi/case_assets/CLEAN-training-v1"
        model_module, distance_module, evaluate_module, utils_module = evaluation.author_modules(assets)
        utils_module.seed_everything(1234)
        implementation = {}
        for path in (Path(__file__), Path(evaluation.__file__), Path(annotation.__file__), Path(inputs.__file__)):
            shutil.copy2(path, output / path.name)
            implementation[path.name] = inputs.sha(path)
        inputs.save(output / "protocol.json", evaluation.evaluation_protocol(implementation))
        inputs.save(output / "input_admission.json", evaluation.admit_inputs(source))
        admitted = {name: evaluation.admit_training(training / name, assets, name) for name in evaluation.SCHEDULES}
        inputs.save(output / "training_admission.json", admitted)
        report.update(stage="native_evaluation", training_completed=True)
        inputs.save(output / "report.json", report)
        plan = json.loads((source / "plan.json").read_text())
        train_records, train_means = evaluation.load_means(source, plan, "train")
        id_ec, ec_ids = utils_module.get_ec_id_dict(str(assets / "source/app/data/split10.csv"))
        if list(id_ec) != [r["id"] for r in train_records] or dict(ec_ids) != annotation.class_index(
            inputs.read_rows(assets / "source/app/data/split10.csv")
        ):
            raise ValueError("Author training membership differs")
        train_ids, train_sequences = set(id_ec), {r["sequence"] for r in train_records}
        positions = {r["id"]: i for i, r in enumerate(train_records)}
        packed_positions = [positions[entry] for members in ec_ids.values() for entry in list(members)]
        for algorithm, schedule in evaluation.SCHEDULES.items():
            state = torch.load(training / algorithm / "final.pth", map_location="cpu", weights_only=True)
            evaluation.state_arrays(state, schedule["out_dim"])
            model = model_module.LayerNormNet(512, schedule["out_dim"], "cpu", torch.float32)
            model.load_state_dict(state, strict=True)
            model.eval()
            with torch.no_grad():
                projected_train = model(torch.from_numpy(train_means))
                packed_train = projected_train[packed_positions]
                native_centers = distance_module.get_cluster_center(packed_train, ec_ids)
            np.save(output / (algorithm + ".train_projection.npy"), projected_train.numpy(), allow_pickle=False)
            np.save(
                output / (algorithm + ".centers.npy"),
                np.stack([native_centers[ec].numpy() for ec in ec_ids]),
                allow_pickle=False,
            )
            inputs.save(output / (algorithm + ".classes.json"), list(ec_ids))
            for cohort in evaluation.COHORTS:
                key = algorithm + "." + cohort
                try:
                    records, means = evaluation.load_means(source, plan, cohort)
                    ids = [r["id"] for r in records]
                    # Identity-only keys go to native inference; evaluation labels are used only afterwards.
                    with torch.no_grad():
                        projected = model(torch.from_numpy(means))
                        distances = distance_module.get_dist_map_test(
                            packed_train, projected, ec_ids, dict.fromkeys(ids), "cpu", torch.float32, dot=False
                        )
                    frame = pd.DataFrame.from_dict(distances).reindex(index=list(ec_ids), columns=ids)
                    np.save(output / (key + ".projection.npy"), projected.numpy(), allow_pickle=False)
                    np.save(output / (key + ".distances.npy"), frame.to_numpy().T, allow_pickle=False)
                    inputs.save(output / (key + ".ids.json"), evaluation.identity_records(records))
                    evaluate_module.write_max_sep_choices(
                        frame, str(output / key), first_grad=True, use_max_grad=False, gmm=None
                    )
                    labels, _ = annotation.read_predictions(output / (key + "_maxsep.csv"), ids)
                    truth = [r["labels"] for r in records]
                    report["results"][key] = {
                        "rows": len(records),
                        "full_union_metrics": evaluation.sklearn_metrics(truth, labels),
                        "truth_only_label_universe_metrics": evaluation.sklearn_metrics(truth, labels, truth_only=True),
                        "train_id_overlap": sum(r["id"] in train_ids for r in records),
                        "train_exact_sequence_overlap": sum(r["sequence"] in train_sequences for r in records),
                    }
                except Exception as exc:
                    report["failures"][key] = {"error_type": type(exc).__name__, "error": str(exc)}
                inputs.save(output / "report.json", report)
        if report["failures"]:
            raise ValueError("One or more native allocations failed; all attempted allocations retained")
        report["artifact_sha256"] = {
            name: inputs.sha(inputs.ordinary(output, name)) for name in sorted(evaluation.artifact_names())
        }
        report.update(status="computed_pending_audit", stage="independent_audit")
        inputs.save(output / "report.json", report)
        audit = evaluation.verify_output(output, source, training, assets)
        inputs.save(output / "independent_audit.json", audit)
        report["independent_audit_sha256"] = inputs.sha(output / "independent_audit.json")
        if audit["status"] != "passed":
            raise ValueError("Independent numerical audit failed; mismatches retained")
        report.update(status="completed", stage="audited_evaluation", evaluation_completed=True)
    except Exception as exc:
        report.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        report["wall_seconds"] = time.monotonic() - started
        inputs.save(output / "report.json", report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, default=ROOT)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
