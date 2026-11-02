"""Read-only independent ridge reconstruction of terminal McbA notebook tasks.

Reuses author residue descriptors, but fits each selected ridge model with an
augmented-design SVD instead of the author's GridSearchCV/Ridge prediction path.
This does not independently reselect alpha or create new scientific replicates.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text())


def ridge_svd(x, y, alpha):
    if not np.isfinite(x).all() or not np.isfinite(y).all() or not alpha > 0:
        raise ValueError("Invalid ridge inputs")
    x_mean, y_mean = x.mean(axis=0), y.mean()
    design = np.vstack([x - x_mean, np.sqrt(alpha) * np.eye(x.shape[1])])
    response = np.concatenate([y - y_mean, np.zeros(x.shape[1])])
    coef = np.linalg.lstsq(design, response, rcond=None)[0]
    return coef, float(y_mean - x_mean @ coef)


def signed_ndcg(y, predictions):
    weights = 1 / np.log2(np.arange(len(y)) + 2)
    # Match the published pandas quicksort tie convention explicitly.
    order = pd.Series(predictions).sort_values(ascending=False).index.to_numpy()
    ideal = pd.Series(y).sort_values(ascending=False).index.to_numpy()
    denominator = y[ideal] @ weights
    if denominator == 0:
        raise ValueError("Undefined author DCG ratio")
    return float((y[order] @ weights) / denominator)


def prediction_comparison(reconstructed, recorded, identities):
    reconstructed, recorded = np.asarray(reconstructed), np.asarray(recorded)
    if reconstructed.ndim != 1 or reconstructed.shape != recorded.shape or len(identities) != len(recorded):
        raise ValueError("Prediction shape or identity count changed")
    if not np.isfinite(reconstructed).all() or not np.isfinite(recorded).all():
        raise ValueError("Non-finite prediction evidence")
    difference = np.abs(reconstructed - recorded)
    tolerance = 1e-5 + 1e-7 * np.abs(recorded)
    failures = np.flatnonzero(difference > tolerance)
    return {
        "passed": not len(failures),
        "compared_rows": len(recorded),
        "outside_tolerance": len(failures),
        "max_absolute_error": float(difference.max(initial=0)),
        "max_tolerance_ratio": float((difference / tolerance).max(initial=0)),
        "examples": [
            {
                "id": str(identities[i]),
                "reconstructed": float(reconstructed[i]),
                "recorded": float(recorded[i]),
                "absolute_error": float(difference[i]),
            }
            for i in failures[:5]
        ],
    }


def summarize(checked):
    incomplete = [x["id"] for x in checked if x.get("scientific_result") is False]
    discrepant = [x["id"] for x in checked if x["status"] == "numerical_or_reference_discrepancy"]
    return {
        "status": "incomplete_reproduction"
        if incomplete
        else "verified_with_discrepancies"
        if discrepant
        else "passed",
        "all_selected_tasks_completed": not incomplete,
        "unsuccessful_tasks": incomplete,
        "discrepant_tasks": discrepant,
        "independent_prediction_replay_passed": not incomplete
        and all(x.get("numerical_prediction_replay_passed") is True for x in checked),
        "historical_reference_counts": {
            "passed": sum(x.get("historical_reference_passed") is True for x in checked),
            "failed": sum(x.get("historical_reference_passed") is False for x in checked),
            "unavailable": sum(x.get("historical_reference_passed") is None for x in checked),
        },
    }


def main(args):
    campaign, source = args.campaign.resolve(), args.source.resolve()
    protocol, report = load(campaign / "protocol.json"), load(campaign / "report.json")
    if report["protocol_sha256"] != sha(campaign / "protocol.json"):
        raise ValueError("Protocol changed")
    if sha(campaign / "runner.py") != protocol["runner_sha256"]:
        raise ValueError("Archived runner changed")
    manifest = load(campaign / "source_manifest.json")
    if sha(source / "source.json") != protocol["source_manifest_sha256"] or load(source / "source.json") != manifest:
        raise ValueError("Source manifest changed")
    for entry in manifest["files"]:
        if sha(source / entry["path"]) != entry["sha256"]:
            raise ValueError("Author source changed")
    if args.task_id:
        selected = set(args.task_id)
        planned = [t for t in protocol["tasks"] if t["id"] in selected]
        if len(planned) != len(selected):
            raise ValueError("Unknown selected task")
        scope = "selected terminal tasks only: " + ", ".join(t["id"] for t in planned)
    elif args.ranks_only:
        planned = [t for t in protocol["tasks"] if t["kind"] == "rank"]
        scope = "nine completed ranking tasks only; campaign may still be running"
    else:
        if report["status"] != "campaign_completed":
            raise ValueError("Complete campaign required")
        planned = protocol["tasks"]
        scope = "all allocated notebook tasks; unsuccessful tasks retained"
    rows = {t["id"]: t for t in report["tasks"]}
    if len(rows) != len(report["tasks"]) or list(rows) != [t["id"] for t in protocol["tasks"]]:
        raise ValueError("Allocation changed")
    if any(rows[t["id"]]["status"] in {"running", "allocated"} for t in planned):
        raise ValueError("Requested tasks have not terminated")
    args.output.mkdir(parents=True, exist_ok=False)
    audit_protocol = {
        "scope": scope,
        "absolute_prediction_tolerance": 1e-5,
        "relative_prediction_tolerance": 1e-7,
        "test_metric_tolerance": 1e-5,
        "source": str(source),
        "campaign": str(campaign),
        "campaign_protocol_sha256": sha(campaign / "protocol.json"),
        "auditor_sha256": sha(Path(__file__)),
        "method": "independent augmented-design least-squares SVD with selected alpha; author descriptor tables reused; does not repeat hyperparameter selection",
        "discrepancy_policy": "Collect all finite numerical discrepancies at unchanged tolerances. They remain failures of numerical/reference agreement, not integrity failures or successful reproductions. Structural, hash and non-finite evidence errors still abort.",
    }
    (args.output / "protocol.json").write_text(json.dumps(audit_protocol, indent=2))
    (args.output / "auditor.py").write_bytes(Path(__file__).read_bytes())
    sys.path.insert(0, str(source))
    helper = importlib.import_module("src.AugmentedMLDE_HelperFuncs")
    mutations = np.asarray(["".join(x) for x in itertools.product(helper.ALL_AAS, repeat=4)], dtype="U4")
    positions = {x: i for i, x in enumerate(mutations)}
    encodings = helper.encode(pd.DataFrame(), 4)
    zero_cache = {}
    checked = []
    for task in planned:
        record = rows[task["id"]]
        if any(record[k] != task[k] for k in ["kind", "product", "subset", "reference"]):
            raise ValueError("Allocation identity changed")
        folder = campaign / task["id"]
        if record["status"] != "completed":
            checked.append({"id": task["id"], "status": record["status"], "scientific_result": False})
            continue
        local = load(folder / "report.json")
        if any(record[k] != value for k, value in local.items()):
            raise ValueError("Task report differs from campaign")
        for name, h in record["input_hashes"].items():
            if sha(source / name) != h:
                raise ValueError("Task input changed")
        for name, h in record["output_hashes"].items():
            if sha(folder / name) != h:
                raise ValueError("Task output changed")
        training_files = [name for name in record["input_hashes"] if name.endswith(".xlsx") and "_train" in name]
        if len(training_files) != 1:
            raise ValueError("Ambiguous training source")
        train = pd.read_excel(source / training_files[0], keep_default_na=False)
        ids = [positions[x] for x in train["AminoAcid"]]
        raw_y = train["Activity"].to_numpy(dtype=float)
        y = (raw_y - raw_y.mean()) / raw_y.std(ddof=1)
        product = task["product"]
        for name in [n for n in record["input_hashes"] if n.endswith(".csv")]:
            if name not in zero_cache:
                frame = pd.read_csv(source / name).sort_values("Mutations").set_index("Mutations")
                if frame.index.tolist() != mutations.tolist():
                    raise ValueError("Zero-shot grid is not the declared combinatorial ordering")
                raw = frame["Predictions"].to_numpy(dtype=float)
                zero_cache[name] = -10000 * (raw - raw.mean()) / raw.std(ddof=0)
        zero = {
            "EC": zero_cache.get("data/zero-shot_predictions/" + product + "_zeroshot_EC.csv"),
            "Energy": zero_cache.get("data/zero-shot_predictions/" + product + "_zeroshot_maestro.csv"),
            "ESM": zero_cache.get("data/zero-shot_predictions/" + product + "_zeroshot_ESM1b_nc.csv"),
        }
        saved = np.load(folder / "predictions.npz", allow_pickle=False)
        if not np.array_equal(saved["mutations"], mutations):
            raise ValueError("Prediction grid changed")
        columns = saved["columns"].tolist()
        saved_matrix = saved["predictions"]
        if saved_matrix.shape != (len(mutations), len(columns)) or len(set(columns)) != len(columns):
            raise ValueError("Prediction matrix or architecture identities changed")
        architecture_names = ["AugmentedEC: Georgiev"] if task["kind"] == "rank" else columns
        searches = load(folder / "grid_search.json")
        if len(searches) != len(architecture_names):
            raise ValueError("Search and architecture counts differ")
        test_files = [n for n in record["input_hashes"] if n.endswith("_test.xlsx")]
        test = pd.read_excel(source / test_files[0], keep_default_na=False) if test_files else None
        test_prediction = pd.read_csv(folder / "test_predictions.csv") if test is not None else None
        metrics = pd.read_csv(folder / "metrics.csv").set_index("architecture") if test is not None else None
        maximum = 0.0
        metric_discrepancies = []
        prediction_discrepancies = []
        heldout_discrepancies = []
        architecture_checks = []
        for j, (architecture, search) in enumerate(zip(architecture_names, searches, strict=True)):
            model, encoding = architecture.split(": ")
            keys = [] if model == "Simple" else model.removeprefix("Augmented").split("_")
            parts = [zero[key][:, None] for key in keys] + [encodings[encoding]]
            features = np.concatenate(parts, axis=1)
            if search["folds"] != 100 or not np.array_equal(search["alpha_grid"], np.linspace(0.01, 100, 100)):
                raise ValueError("Hyperparameter grid changed")
            if search["best_alpha"] not in search["alpha_grid"]:
                raise ValueError("Selected alpha is outside the declared grid")
            coef, intercept = ridge_svd(features[ids], y, search["best_alpha"])
            predictions = features @ coef + intercept
            comparison = prediction_comparison(predictions, saved_matrix[:, j], mutations)
            maximum = max(maximum, comparison["max_absolute_error"])
            architecture_checks.append({"architecture": architecture, "grid": comparison})
            if not comparison["passed"]:
                prediction_discrepancies.append({"architecture": architecture, **comparison})
            if test is not None:
                t_ids = [positions[x] for x in test["AminoAcid"]]
                if test_prediction["Mutation"].tolist() != test["AminoAcid"].tolist() or not np.allclose(
                    test_prediction["Actual"], test["Activity"], rtol=0, atol=1e-12
                ):
                    raise ValueError("Held-out test rows differ")
                comparison = prediction_comparison(
                    predictions[t_ids], test_prediction[architecture].to_numpy(), test["AminoAcid"].to_numpy()
                )
                architecture_checks[-1]["heldout"] = comparison
                if not comparison["passed"]:
                    heldout_discrepancies.append({"architecture": architecture, **comparison})
                actual = test["Activity"].to_numpy(dtype=float)
                actual = (actual - actual.mean()) / actual.std(ddof=1)
                rho = float(spearmanr(actual, predictions[t_ids]).statistic)
                ndcg = signed_ndcg(actual, predictions[t_ids])
                if not np.isclose(rho, metrics.loc[architecture, "spearman_r"], rtol=0, atol=1e-5) or not np.isclose(
                    ndcg, metrics.loc[architecture, "NDCG"], rtol=0, atol=1e-5
                ):
                    metric_discrepancies.append(
                        {
                            "architecture": architecture,
                            "reconstructed_spearman": rho,
                            "recorded_spearman": float(metrics.loc[architecture, "spearman_r"]),
                            "reconstructed_author_NDCG": ndcg,
                            "recorded_author_NDCG": float(metrics.loc[architecture, "NDCG"]),
                        }
                    )
        checked.append(
            {
                "id": task["id"],
                "status": "numerical_or_reference_discrepancy"
                if (
                    prediction_discrepancies
                    or heldout_discrepancies
                    or metric_discrepancies
                    or record["reference_passed"] is False
                )
                else "passed",
                "numerical_prediction_replay_passed": not prediction_discrepancies and not heldout_discrepancies,
                "grid_prediction_discrepancies": prediction_discrepancies,
                "heldout_prediction_discrepancies": heldout_discrepancies,
                "architecture_checks": architecture_checks,
                "test_metric_discrepancies": metric_discrepancies,
                "architectures": len(architecture_names),
                "prediction_rows_per_architecture": len(mutations),
                "max_svd_prediction_absolute_error": maximum,
                "historical_reference_passed": record["reference_passed"],
            }
        )
        (args.output / "progress.json").write_text(json.dumps(checked, indent=2))
        print(json.dumps(checked[-1]), flush=True)
    result = {
        **summarize(checked),
        "scope": scope,
        "tasks": checked,
        "source_integrity_passed": True,
        "campaign_completed": report["status"] == "campaign_completed",
        "new_efficacy_replicate": False,
    }
    (args.output / "audit.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--ranks-only", action="store_true")
    selection.add_argument("--task-id", action="append")
    main(parser.parse_args())
