"""Source-bound CataPro training inputs and independent full-row output audits."""

from __future__ import annotations

import csv
import fcntl
import json
import os
from contextlib import contextmanager

import numpy as np

from .catapro_case import file_sha, numpy_forward

PARAMETERS = {
    "kcat": {"lr": 1e-5, "batch_size": 8, "dropout": 0.0},
    "km": {"lr": 1e-5, "batch_size": 16, "dropout": 0.1},
    "activity": {"lr": 0.01, "batch_size": 16, "dropout": 0.1},
}
STAGES = {
    "kcat": ("kcat", "kcat", "kcat"),
    "km": ("km", "km", "Km"),
    "activity_kcat": ("activity", "kcat", "kcat"),
    "activity_km": ("activity", "km", "Km"),
    "activity": ("activity", "activity", "kcat_over_Km"),
}


def make_table(rows, features, target, raw_kinetics=None):
    """The last columns reproduce the positional ABI of author CVDatasets."""
    import pandas as pd

    if target not in PARAMETERS:
        raise ValueError("Unknown training target")
    ids = [r["id"] for r in rows]
    folds = np.asarray([r["fold"] for r in rows])
    if len(set(ids)) != len(ids) or not ids:
        raise ValueError("Training row identifiers must be nonempty and unique")
    if folds.dtype.kind not in "iu" or set(folds.tolist()) != set(range(10)):
        raise ValueError("All ten original folds are required")
    if features.shape != (len(rows), 1959) or not np.isfinite(features).all():
        raise ValueError("Feature rows, dimensions or finiteness differ")
    if target == "activity":
        labels = np.asarray(raw_kinetics, dtype=np.float64)
        if labels.shape != (len(rows), 2) or not np.isfinite(labels).all() or np.any(labels <= 0):
            raise ValueError("Activity requires positive raw kcat and Km in mM")
        intended = np.asarray([r["targets"]["activity"] for r in rows])
        if not np.allclose(np.log10(labels[:, 0] / labels[:, 1]), intended, rtol=0, atol=1e-10):
            raise ValueError("Activity label units or row alignment differ")
        columns = ["kcat_s_inverse", "km_mM"]
    else:
        labels = np.asarray([[r["targets"][target]] for r in rows], dtype=np.float64)
        columns = ["log10_" + target]
    values = np.concatenate([features, labels, folds[:, None]], axis=1).astype(np.float32)
    if not np.isfinite(values).all():
        raise ValueError("Float32 training table is not finite")
    return pd.DataFrame(values, index=ids, columns=[f"feature_{i:04d}" for i in range(1959)] + columns + ["fold"])


def training_arguments(root, stage, epochs, device="cuda:0"):
    if stage not in STAGES or type(epochs) is not int or not 1 <= epochs <= 150:
        raise ValueError("Unknown stage or invalid epoch cap")
    target = STAGES[stage][1]
    params = PARAMETERS[target]
    args = [
        "-feat_fpath" if target == "activity" else "-data_fpath",
        str(root / "tables" / (stage + ".pkl")),
        "-lr",
        str(params["lr"]),
        "-batch_size",
        str(params["batch_size"]),
        "-rate" if target == "activity" else "-drop_rate",
        str(params["dropout"]),
        "-epochs",
        str(epochs),
        "-device",
        device,
    ]
    if target == "activity":
        args += [
            "-alpha",
            "0.4",
            "-kcat_model_dpath",
            str(root / "stages/activity_kcat"),
            "-Km_model_dpath",
            str(root / "stages/activity_km"),
        ]
    return args


def validate_fold_batches(table, batch_size):
    result = []
    for fold in range(10):
        valid = int(np.sum(table["fold"].to_numpy() == fold))
        train = len(table) - valid
        if not valid or train < 2 or train % batch_size == 1:
            raise ValueError("Author BatchNorm training cannot consume a singleton batch")
        result.append(
            {
                "fold": fold,
                "train_rows": train,
                "validation_rows": valid,
                "last_train_batch": train % batch_size or batch_size,
            }
        )
    return result


def metrics(target, prediction):
    from scipy.stats import spearmanr

    target, prediction = np.asarray(target, dtype=np.float64), np.asarray(prediction, dtype=np.float64)
    if target.shape != prediction.shape or not np.isfinite(prediction).all():
        raise ValueError("Invalid prediction array")
    return {
        "PCC": float(np.corrcoef(target, prediction)[0, 1]),
        "SCC": float(spearmanr(target, prediction).statistic),
        "R2": float(1 - np.sum((target - prediction) ** 2) / np.sum((target - target.mean()) ** 2)),
        "RMSE": float(np.sqrt(np.mean((target - prediction) ** 2))),
    }


def verify_files(root, bindings):
    for name, expected in bindings.items():
        p = root / name
        if p.is_symlink() or not p.is_file() or file_sha(p) != expected:
            raise ValueError("Frozen training file changed: " + name)


def audit_stage(root, stage, *, output=None):
    """Recompute all OOF predictions from newly trained weights using NumPy."""
    import pandas as pd
    import torch

    destination = root / "stages" / stage if output is None else output
    if output is not None:
        if output.resolve().is_relative_to(root.resolve()):
            raise ValueError("External audit output must not modify the source campaign")
        output.mkdir(parents=True, exist_ok=False)
    protocol = json.loads((root / "protocol.json").read_text())
    binding = protocol["tables"][stage]
    table_path = root / binding["path"]
    if file_sha(table_path) != binding["sha256"]:
        raise ValueError("Training table changed")
    table = pd.read_pickle(table_path)
    features = table.iloc[:, :1959].to_numpy()
    folds = table["fold"].to_numpy()
    target_kind = STAGES[stage][1]
    target = (
        np.log10(table.iloc[:, -3].to_numpy() / table.iloc[:, -2].to_numpy())
        if target_kind == "activity"
        else table.iloc[:, -2].to_numpy()
    )
    prediction = np.full(len(table), np.nan, dtype=np.float64)
    folder = root / "stages" / stage
    checkpoints = []
    base_checks = []
    for fold in range(10):
        path = folder / f"{fold}_bestmodel.pth"
        state = torch.load(path, map_location="cpu", weights_only=True)
        indices = np.flatnonzero(folds == fold)
        for offset in range(0, len(indices), 256):
            selected = indices[offset : offset + 256]
            prediction[selected] = numpy_forward(features[selected], state, activity=target_kind == "activity")
        checkpoints.append({"fold": fold, "sha256": file_sha(path), "validation_rows": len(indices)})
        if target_kind == "activity":
            for base_stage, prefix in [("activity_kcat", "kcat_model."), ("activity_km", "Km_model.")]:
                base = torch.load(
                    root / "stages" / base_stage / f"{fold}_bestmodel.pth", map_location="cpu", weights_only=True
                )
                parameter_keys = [
                    k for k in base if not k.endswith(("running_mean", "running_var", "num_batches_tracked"))
                ]
                unchanged = all(torch.equal(v, state[prefix + k]) for k, v in base.items() if k in parameter_keys)
                counters = [int(state[prefix + k] - v) for k, v in base.items() if k.endswith("num_batches_tracked")]
                if not unchanged:
                    raise ValueError("Detached pretrained parameters unexpectedly changed")
                base_checks.append(
                    {
                        "fold": fold,
                        "stage": base_stage,
                        "parameters_unchanged": unchanged,
                        "batchnorm_counter_increases": counters,
                    }
                )
    with (folder / "final_pred_label.csv").open(newline="") as stream:
        saved = list(csv.DictReader(stream))
    if (
        len(saved) != len(table)
        or len({r[""] for r in saved}) != len(table)
        or {r[""] for r in saved} != set(table.index)
    ):
        raise ValueError("Author prediction rows differ from full cohort")
    positions = {key: i for i, key in enumerate(table.index)}
    p_error = max(abs(float(r["pred"]) - prediction[positions[r[""]]]) for r in saved)
    y_error = max(abs(float(r["label"]) - target[positions[r[""]]]) for r in saved)
    recomputed = metrics(target, prediction)
    with (folder / "final_results.csv").open(newline="") as stream:
        original = next(csv.DictReader(stream))
    metric_error = {k: abs(float(original[k]) - v) for k, v in recomputed.items()}
    if p_error > 2e-4 or y_error > 6e-5 or max(metric_error.values()) > 2e-5:
        raise ValueError(f"Independent trained-checkpoint parity failed: {p_error}, {y_error}, {metric_error}")
    np.save(destination / "independent_predictions.npy", prediction, allow_pickle=False)
    result = {
        "status": "passed",
        "stage": stage,
        "rows": len(table),
        "folds": 10,
        "epochs_cap": protocol["epochs"],
        "training_scope": protocol["scope"],
        "metrics": recomputed,
        "author_metrics": {k: float(original[k]) for k in recomputed},
        "max_prediction_difference_from_rounded_author_csv": float(p_error),
        "max_label_difference_from_rounded_author_csv": float(y_error),
        "metric_differences": metric_error,
        "checkpoints": checkpoints,
        "detached_base_checks": base_checks,
        "independent_predictions_sha256": file_sha(destination / "independent_predictions.npy"),
        "author_prediction_sha256": file_sha(folder / "final_pred_label.csv"),
        "unbiased_new_holdout": False,
        "new_model_or_rsi_advantage": False,
    }
    (destination / "audit.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    return result


@contextmanager
def gpu_lease(path):
    """Reuse an inherited broker lease without releasing the broker's lock."""
    inherited = os.environ.get("VENUS_RSI_CASE_LOCK_FD")
    if inherited is not None:
        fd = int(inherited)
        held, expected = os.fstat(fd), path.stat()
        if (held.st_dev, held.st_ino) != (expected.st_dev, expected.st_ino):
            raise ValueError("Inherited GPU lease refers to a different lock")
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with os.fdopen(os.dup(fd), "a") as stream:
            yield stream
    else:
        with path.open("a") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            yield stream


def stop_training_process(process, grace=1.0):
    """Reap the separate training group before the broker's three-second kill."""
    import signal
    import subprocess

    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        pass
    # A DataLoader descendant can survive even when its parent exited.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()
