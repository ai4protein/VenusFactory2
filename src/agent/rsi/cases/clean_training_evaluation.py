"""Fixed CLEAN split10 training admission and independent numerical evaluation.

This software uses or is derived from CLEAN, developed at the University of
Illinois Urbana-Champaign, Department of Chemical and Biomolecular Engineering.
The separately licensed author model and prediction functions remain unchanged.
Training provenance is audited; stochastic training and ESM are not rerun here.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

from agent.rsi import clean_annotation as annotation
from agent.rsi import clean_training_inputs as inputs

INPUT_PROTOCOL = "d04d94e3e7b07aee4b8d412b72cef5e044885515ee70cb8828e5a03336c05318"
PREPARATION = {
    "clean_training_inputs.py": "128c2161ec37730874a6a36996281b4c011f4138b47015d155c9beac88ef9c7a",
    "run_clean_training_inputs.py": "345abecb607c9bd96a3746b9ef2f0aa9367aac9df0bde524c63115767a326589",
}
TRAINING_EXECUTOR = "4e35f262e06121aeea01b6bae128dc84326cf37033f78a21962f30020d2fbfad"
SCHEDULES = {
    "triplet": {"script": "train-triplet.py", "epochs": 2000, "out_dim": 128, "adaptive_rate": 100},
    "supconH": {"script": "train-supconH.py", "epochs": 1500, "out_dim": 256, "adaptive_rate": 60},
}
AUTHOR_SCRIPTS = {
    "triplet": "fe8b3e2e02220878a1034ff08a847953cc1e2007161be934767f306f6e61d3bc",
    "supconH": "427975b16faa2b7d0b56e7ed7c75c6437a7bb7e4c0c59a57712b9c7750e51c30",
}
COHORTS = {name: inputs.COUNTS[name] for name in ("new", "price", "halogenase")}
ALLOCATION = {"models": list(SCHEDULES), "cohorts": COHORTS, "sequence_records": 578, "prediction_records": 1156}
# Fixed before either training schedule produces evaluation results.
NUMERICS = {"atol": 5e-5, "rtol": 5e-6, "metric_atol": 1e-12, "exact_labels": True}
LIMITS = {
    "full_paper_reproduced": False,
    "autonomous_model_developed": False,
    "new_model_developed": False,
    "recursive_improvement_established": False,
    "publication_cross_validation_reproduced": False,
    "homology_independence_established": False,
    "transformer_rerun_by_auditor": False,
    "training_rerun_by_auditor": False,
}


def validate_protocol(protocol):
    if inputs.digest(protocol) != INPUT_PROTOCOL or protocol.get("implementation") != PREPARATION:
        raise ValueError("Training input protocol or admitted preparation source differs")


def admit_inputs(root):
    protocol = json.loads(inputs.ordinary(root, "protocol.json").read_text())
    validate_protocol(protocol)
    for name, expected in PREPARATION.items():
        if inputs.sha(inputs.ordinary(root, name)) != expected:
            raise ValueError("Archived preparation source differs")
    report = json.loads(inputs.ordinary(root, "report.json").read_text())
    if type(report.get("external_model_calls")) is not int or report["external_model_calls"] != 0:
        raise ValueError("Unexpected model calls in fixed input preparation")
    audit = inputs.verify_completed(root)
    return {**audit, "protocol_sha256": INPUT_PROTOCOL, "report_sha256": inputs.sha(root / "report.json")}


def state_arrays(state, out_dim):
    """Strict state layout; loading uses torch weights_only at the call site."""
    if out_dim not in (128, 256):
        raise ValueError("Unadmitted CLEAN output dimension")
    shapes = {
        "fc1.weight": (512, 1280),
        "fc1.bias": (512,),
        "fc2.weight": (512, 512),
        "fc2.bias": (512,),
        "fc3.weight": (out_dim, 512),
        "fc3.bias": (out_dim,),
        **{f"ln{i}.{part}": (512,) for i in (1, 2) for part in ("weight", "bias")},
    }
    if not isinstance(state, dict) or set(state) != set(shapes):
        raise ValueError("Unexpected CLEAN checkpoint state keys")
    result = {}
    for name, shape in shapes.items():
        value = state[name]
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        value = np.asarray(value)
        if value.shape != shape or value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError("Invalid CLEAN checkpoint tensor: " + name)
        result[name] = value
    return result


def audit_epochs(records, count):
    if len(records) != count:
        raise ValueError("Incomplete epoch trace")
    previous = -math.inf
    for index, row in enumerate(records, 1):
        if type(row.get("epoch")) is not int or row["epoch"] != index:
            raise ValueError("Repeated or unordered epoch trace")
        loss, at = row.get("loss"), row.get("at")
        if type(loss) not in (int, float) or not math.isfinite(loss) or loss < 0:
            raise ValueError("Invalid epoch loss")
        if type(at) not in (int, float) or not math.isfinite(at) or at <= 0 or at < previous:
            raise ValueError("Invalid epoch timestamp order")
        previous = at


def admit_training(root, assets, algorithm):
    import torch

    schedule = SCHEDULES[algorithm]
    report = json.loads(inputs.ordinary(root, "report.json").read_text())
    expected = {
        "status": "completed",
        "stage": "trained_checkpoint",
        "algorithm": algorithm,
        "schedule": schedule,
        "source_commit": inputs.COMMIT,
        "author_script_sha256": AUTHOR_SCRIPTS[algorithm],
        "input_protocol_sha256": INPUT_PROTOCOL,
        "training_completed": True,
        "evaluation_completed": False,
        "new_model_developed": False,
        "autonomous_model_developed": False,
        "full_paper_reproduced": False,
    }
    if any(type(report.get(k)) is not type(v) or report[k] != v for k, v in expected.items()):
        raise ValueError("Training report scope or provenance differs")
    if type(report.get("completed_epochs")) is not int or report["completed_epochs"] != schedule["epochs"]:
        raise ValueError("Training epoch count differs")
    if type(report.get("external_model_calls")) is not int or report["external_model_calls"] != 0:
        raise ValueError("Unexpected model calls in author training")
    if type(report.get("cache_hits")) is not int or report["cache_hits"] <= 0:
        raise ValueError("Missing training cache execution evidence")
    hashes = report.get("artifact_sha256", {})
    names = {"initial.pth", "final.pth", "epochs.jsonl", schedule["script"], "clean_author_training.py"}
    if set(hashes) != names:
        raise ValueError("Training artifact coverage differs")
    for name, expected_hash in hashes.items():
        if inputs.sha(inputs.ordinary(root, name)) != expected_hash:
            raise ValueError("Training artifact changed: " + name)
    if (
        hashes["clean_author_training.py"] != TRAINING_EXECUTOR
        or hashes[schedule["script"]] != AUTHOR_SCRIPTS[algorithm]
    ):
        raise ValueError("Training executor or author script not admitted")
    author_script = inputs.ordinary(assets, "source/app/" + schedule["script"])
    if inputs.sha(author_script) != AUTHOR_SCRIPTS[algorithm]:
        raise ValueError("Pinned author training script changed")
    if inputs.sha(inputs.ordinary(root, f"work/data/model/split10_{algorithm}.pth")) != hashes["final.pth"]:
        raise ValueError("Final checkpoint differs from author output")
    if inputs.sha(inputs.ordinary(root, "work/data/split10.csv")) != inputs.sha(
        inputs.ordinary(assets, "source/app/data/split10.csv")
    ):
        raise ValueError("Training cohort changed")
    records = [json.loads(line) for line in (root / "epochs.jsonl").read_text().splitlines()]
    audit_epochs(records, schedule["epochs"])
    before, after = [
        state_arrays(torch.load(root / name, map_location="cpu", weights_only=True), schedule["out_dim"])
        for name in ("initial.pth", "final.pth")
    ]
    changed = sum(int(np.count_nonzero(before[name] != after[name])) for name in before)
    if (
        not changed
        or type(report.get("parameter_values_changed")) is not int
        or report["parameter_values_changed"] != changed
    ):
        raise ValueError("Checkpoint parameter change evidence differs")
    return {
        "status": "passed",
        "algorithm": algorithm,
        "epochs": len(records),
        "parameter_values_changed": changed,
        "final_sha256": hashes["final.pth"],
        "report_sha256": inputs.sha(root / "report.json"),
        "artifact_sha256": hashes,
        "training_rerun_by_auditor": False,
        "boundary": "Bound author source, complete loss trace and checkpoints; no independent stochastic-training replay or distance-cache deserialization",
    }


def forward(state, x, out_dim):
    weights = state_arrays(state, out_dim)
    h = np.asarray(x, dtype=np.float64)
    if h.ndim != 2 or h.shape[1] != 1280 or not np.isfinite(h).all():
        raise ValueError("Invalid ESM mean matrix")
    for i in (1, 2, 3):
        h = h @ weights[f"fc{i}.weight"].astype(np.float64).T + weights[f"fc{i}.bias"]
        if i != 3:
            h = (h - h.mean(1, keepdims=True)) / np.sqrt(h.var(1, keepdims=True) + 1e-5)
            h = np.maximum(h * weights[f"ln{i}.weight"] + weights[f"ln{i}.bias"], 0)
    if not np.isfinite(h).all():
        raise ValueError("Nonfinite independent projection")
    return h


def class_centers(projected, rows):
    """Independent means from CSV-order proteins, deduplicated per EC membership."""
    matrix = np.asarray(projected, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != len(rows) or not np.isfinite(matrix).all():
        raise ValueError("Invalid projected training matrix")
    ids = [row["Entry"] for row in rows]
    if len(set(ids)) != len(ids) or not ids:
        raise ValueError("Missing or duplicated training identities")
    members = {}
    for index, row in enumerate(rows):
        for ec in dict.fromkeys(row["EC number"].split(";")):
            if not ec:
                raise ValueError("Empty training class")
            members.setdefault(ec, []).append(index)
    return list(members), np.stack([matrix[index].mean(0) for index in members.values()])


def compare(actual, expected):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if (
        actual.shape != expected.shape
        or actual.size == 0
        or not np.isfinite(actual).all()
        or not np.isfinite(expected).all()
    ):
        raise ValueError("Numerical replay shape or finiteness differs")
    difference = np.abs(actual - expected)
    return {
        "passed": bool(np.all(difference <= NUMERICS["atol"] + NUMERICS["rtol"] * np.abs(expected))),
        "max_absolute_error": float(difference.max()),
        "outside_tolerance": int(np.count_nonzero(difference > NUMERICS["atol"] + NUMERICS["rtol"] * np.abs(expected))),
    }


def sklearn_metrics(truth, predictions, truth_only=False):
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    from sklearn.preprocessing import MultiLabelBinarizer

    true_labels, predicted = set().union(*map(set, truth)), set().union(*map(set, predictions))
    labels = true_labels if truth_only else true_labels | predicted
    mlb = MultiLabelBinarizer(classes=sorted(labels))
    a = mlb.fit_transform(truth)
    b = mlb.transform([[v for v in row if v in labels] for row in predictions])
    result = {
        "rows": len(truth),
        "truth_labels": len(true_labels),
        "union_labels": len(labels),
        "exact_match": float(accuracy_score(a, b)),
        "predicted_labels_absent_from_truth": sorted(predicted - true_labels),
    }
    for mode in ("weighted", "micro"):
        precision, recall, f1, _ = precision_recall_fscore_support(a, b, average=mode, zero_division=0)
        result.update({mode + "_precision": float(precision), mode + "_recall": float(recall), mode + "_f1": float(f1)})
    return result


def same_metrics(actual, expected):
    if not isinstance(actual, dict) or set(actual) != set(expected):
        return False
    for key, value in expected.items():
        other = actual[key]
        if type(value) is float:
            if (
                type(other) not in (int, float)
                or not math.isfinite(other)
                or abs(other - value) > NUMERICS["metric_atol"]
            ):
                return False
        elif type(other) is not type(value) or other != value:
            return False
    return True


def load_means(input_root, plan, cohort):
    records = [row for row in plan["records"] if row["cohort"] == cohort]
    tensors = [
        inputs.verify_receipt(
            input_root / "work", input_root / "receipts", row, plan["encoder"]["model"]["sha256"], INPUT_PROTOCOL
        ).numpy()
        for row in records
    ]
    return records, np.stack(tensors)


def identity_records(records):
    return [
        {
            "id": r["id"],
            "key": r["key"],
            "sequence_sha256": hashlib.sha256(r["sequence"].encode()).hexdigest(),
            "length": r["tokens"],
        }
        for r in records
    ]


def author_modules(assets):
    inputs.verify_assets(assets)
    source = assets / "source/app/src"
    for name, module in list(sys.modules.items()):
        if name == "CLEAN" or name.startswith("CLEAN."):
            if not Path(module.__file__).resolve().is_relative_to(source.resolve()):
                raise ValueError("A different CLEAN source is already loaded")
    sys.path.insert(0, str(source))
    from CLEAN import distance_map, evaluate, model, utils

    return model, distance_map, evaluate, utils


def artifact_names():
    return (
        {
            "protocol.json",
            "training_admission.json",
            "input_admission.json",
            "clean_training_evaluation.py",
            "run_clean_training_evaluation.py",
            "clean_annotation.py",
            "clean_training_inputs.py",
        }
        | {
            algorithm + suffix
            for algorithm in SCHEDULES
            for suffix in (".train_projection.npy", ".centers.npy", ".classes.json")
        }
        | {
            algorithm + "." + cohort + suffix
            for algorithm in SCHEDULES
            for cohort in COHORTS
            for suffix in (".projection.npy", ".distances.npy", "_maxsep.csv", ".ids.json")
        }
    )


def evaluation_protocol(implementation):
    return {
        "schema": 1,
        "source_commit": inputs.COMMIT,
        "asset_manifest_sha256": inputs.MANIFEST_SHA256,
        "input_protocol_sha256": INPUT_PROTOCOL,
        "training_executor_sha256": TRAINING_EXECUTOR,
        "author_scripts": AUTHOR_SCRIPTS,
        "schedules": SCHEDULES,
        "allocation": ALLOCATION,
        "numerical_tolerances": NUMERICS,
        "environment": inputs.ENVIRONMENT,
        "implementation": implementation,
        "seed": 1234,
        "threads": 4,
        "device": "cpu",
        "selection": "Both fixed schedules; final author checkpoint; no evaluation-based model selection or tuning",
        "metric_policy": "Primary full union of truth and predicted ECs; separately report truth-only label-universe metrics; no AUROC parity claim",
        "limits": LIMITS,
    }


def verify_output(output, input_root, training_root, assets):
    """Recompute from bound ESM means and final state tensors, without native model calls."""
    import torch

    inputs.verify_assets(assets)
    report = json.loads(inputs.ordinary(output, "report.json").read_text())
    if report.get("status") not in ("computed_pending_audit", "completed") or report.get("allocation") != ALLOCATION:
        raise ValueError("Incomplete evaluation report")
    if type(report.get("external_model_calls")) is not int or report["external_model_calls"] != 0:
        raise ValueError("Unexpected external model calls")
    if any(report.get(key) is not value for key, value in LIMITS.items()):
        raise ValueError("Unsupported scientific claim")
    complete = report["status"] == "completed"
    if report.get("training_completed") is not True or report.get("evaluation_completed") is not complete:
        raise ValueError("Training/evaluation completion flags differ")
    if report.get("execution_origin") != "controller_executed_author_training_and_numerically_audited_evaluation":
        raise ValueError("Evaluation origin differs")
    expected_stage = "audited_evaluation" if complete else "independent_audit"
    if report.get("stage") != expected_stage or report.get("failures") != {}:
        raise ValueError("Evaluation stage or failed allocations differ")
    hashes = report.get("artifact_sha256", {})
    if set(hashes) != artifact_names():
        raise ValueError("Incomplete evaluation artifact coverage")
    for name, expected in hashes.items():
        if inputs.sha(inputs.ordinary(output, name)) != expected:
            raise ValueError("Evaluation artifact changed: " + name)
    source_names = [
        "clean_training_evaluation.py",
        "run_clean_training_evaluation.py",
        "clean_annotation.py",
        "clean_training_inputs.py",
    ]
    protocol = json.loads((output / "protocol.json").read_text())
    if protocol != evaluation_protocol({name: hashes[name] for name in source_names}):
        raise ValueError("Evaluation protocol differs")
    # Pin the evaluator's archived implementation to the actual verifier, not merely a self-declared report hash.
    for module in (sys.modules[__name__], annotation, inputs):
        name = Path(module.__file__).name
        if hashes[name] != inputs.sha(Path(module.__file__)):
            raise ValueError("Evaluator implementation differs from audit source")
    runner = Path(__file__).resolve().parents[4] / "scripts/rsi/cases/run_clean_training_evaluation.py"
    if hashes[runner.name] != inputs.sha(runner):
        raise ValueError("Evaluation runner differs from audit source")
    current_input = admit_inputs(input_root)
    if json.loads((output / "input_admission.json").read_text()) != current_input:
        raise ValueError("Input admission record differs")
    admitted = {name: admit_training(training_root / name, assets, name) for name in SCHEDULES}
    if json.loads((output / "training_admission.json").read_text()) != admitted:
        raise ValueError("Training admission record differs")
    expected_keys = {a + "." + c for a in SCHEDULES for c in COHORTS}
    if set(report.get("results", {})) != expected_keys:
        raise ValueError("Evaluation allocation coverage differs")
    plan = json.loads((input_root / "plan.json").read_text())
    train_records, train_means = load_means(input_root, plan, "train")
    train_rows = inputs.read_rows(assets / "source/app/data/split10.csv")
    if [r["id"] for r in train_records] != [r["Entry"] for r in train_rows]:
        raise ValueError("Training embedding order differs")
    train_ids, train_sequences = {r["id"] for r in train_records}, {r["sequence"] for r in train_records}
    audit = {"status": "passed", "training": admitted, "allocations": {}, "limits": LIMITS}
    for algorithm, schedule in SCHEDULES.items():
        state = torch.load(training_root / algorithm / "final.pth", map_location="cpu", weights_only=True)
        projected = forward(state, train_means, schedule["out_dim"])
        labels, centroid = class_centers(projected, train_rows)
        projection_check = compare(
            np.load(output / (algorithm + ".train_projection.npy"), allow_pickle=False), projected
        )
        center_check = compare(np.load(output / (algorithm + ".centers.npy"), allow_pickle=False), centroid)
        if json.loads((output / (algorithm + ".classes.json")).read_text()) != labels:
            raise ValueError("EC center class order differs")
        for cohort in COHORTS:
            key = algorithm + "." + cohort
            records, means = load_means(input_root, plan, cohort)
            truth = [r["labels"] for r in records]
            expected_ids = identity_records(records)
            if json.loads((output / (key + ".ids.json")).read_text()) != expected_ids:
                raise ValueError("Evaluation sequence identity differs")
            prediction = forward(state, means, schedule["out_dim"])
            distances = annotation.distances(prediction, centroid)
            saved = np.load(output / (key + ".distances.npy"), allow_pickle=False)
            checks = {
                "training_projection": projection_check,
                "class_centers": center_check,
                "evaluation_projection": compare(
                    np.load(output / (key + ".projection.npy"), allow_pickle=False), prediction
                ),
                "distances": compare(saved, distances),
            }
            native_labels, rounded = annotation.read_predictions(
                output / (key + "_maxsep.csv"), [r["id"] for r in records]
            )
            replay_labels = annotation.select_labels(distances, labels)
            saved_labels = annotation.select_labels(saved, labels)
            mismatches = [
                {"id": r["id"], "native": native, "independent": independent, "from_saved_distances": from_saved}
                for r, native, independent, from_saved in zip(records, native_labels, replay_labels, saved_labels)
                if native != independent or native != from_saved
            ]
            positions = {label: i for i, label in enumerate(labels)}
            rounded_ok = all(
                label in positions and abs(float(saved[i, positions[label]]) - distance) <= 5.1e-5
                for i, (values, scores) in enumerate(zip(native_labels, rounded))
                for label, distance in zip(values, scores)
            )
            # Metrics compare the same native labels; label agreement is a separate exact gate.
            measured = annotation.metrics(truth, native_labels)
            truth_set = set().union(*map(set, truth))
            truth_only = annotation.metrics(
                truth, [[label for label in row if label in truth_set] for row in native_labels]
            )
            truth_only["predicted_labels_absent_from_truth"] = sorted(set().union(*map(set, native_labels)) - truth_set)
            record = report["results"][key]
            metrics_ok = same_metrics(record.get("full_union_metrics"), measured) and same_metrics(
                record.get("truth_only_label_universe_metrics"), truth_only
            )
            overlap = {
                "train_id_overlap": sum(r["id"] in train_ids for r in records),
                "train_exact_sequence_overlap": sum(r["sequence"] in train_sequences for r in records),
            }
            metadata_ok = record.get("rows") == COHORTS[cohort] and all(record.get(k) == v for k, v in overlap.items())
            passed = (
                all(c["passed"] for c in checks.values())
                and not mismatches
                and rounded_ok
                and metrics_ok
                and metadata_ok
            )
            audit["allocations"][key] = {
                "status": "passed" if passed else "failed",
                "rows": len(records),
                "checks": checks,
                "label_mismatches": mismatches,
                "rounded_distances_match": rounded_ok,
                "metrics_match": metrics_ok,
                "metadata_match": metadata_ok,
                "full_union_metrics": measured,
                "independent_label_metrics": annotation.metrics(truth, replay_labels),
                **overlap,
            }
            if not passed:
                audit["status"] = "failed"
    if complete:
        path = inputs.ordinary(output, "independent_audit.json")
        if inputs.sha(path) != report.get("independent_audit_sha256") or json.loads(path.read_text()) != audit:
            raise ValueError("Completed independent audit binding differs")
        if audit["status"] != "passed":
            raise ValueError("Completed evaluation no longer passes audit")
    return audit
