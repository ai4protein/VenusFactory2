"""Independent full-cohort CataPro evidence; model invention is never inferred."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .catapro_case import DATASETS, checked_manifest, dataset_rows, file_sha, numpy_forward

COMMIT = "cc89b2c81768665cf6fd76dfda607ce88691f601"
COUNTS = {"kcat": 27658, "km": 42018, "activity": 25831}
EXISTING = {"catapro_full_20261017": "research_runs/rsi/20261017_catapro_full_cohorts_v1"}
REGISTERED_PROTOCOL_SHA = "d97f1fa07b65243316ff049c037895fefc4da9cd61a96b026554f59aa1248191"
FOLDERS = {"kcat": "kcat_models", "km": "Km_models", "activity": "act_models"}


def read(path):
    return json.loads(Path(path).read_text())


def regression_metrics(prediction, target):
    from scipy.stats import pearsonr, spearmanr

    return {
        "rows": len(target),
        "rmse": float(np.sqrt(np.mean((prediction - target) ** 2))),
        "pcc": float(pearsonr(prediction, target).statistic),
        "scc": float(spearmanr(prediction, target).statistic),
    }


def audit_cohort(folder, source, kind):
    import torch

    protocol, report = read(folder / "protocol.json"), read(folder / "report.json")
    raw = source / "datasets" / DATASETS[kind]
    if (
        report.get("status") != "completed"
        or report.get("dataset") != kind
        or report.get("rows_encoded") != COUNTS[kind]
        or protocol.get("rows") != COUNTS[kind]
        or protocol.get("dataset") != kind
        or protocol.get("source_commit") != COMMIT
        or protocol.get("input_sha256") != file_sha(raw)
        or protocol.get("source_admission_sha256") != file_sha(source / "admission.json")
        or protocol.get("checkpoint_admission_sha256") != file_sha(source / "checkpoints_admission_api.json")
    ):
        raise ValueError("Incomplete or changed CataPro cohort identity")
    rows = dataset_rows(source, kind)
    if len(rows) != COUNTS[kind]:
        raise ValueError("Cohort row coverage differs")
    for name in ["features", "predictions"]:
        if file_sha(folder / (name + ".npy")) != report.get(name + "_sha256"):
            raise ValueError("Cohort array hash differs")
    feature = np.load(folder / "features.npy", mmap_mode="r", allow_pickle=False)
    prediction = np.load(folder / "predictions.npy", allow_pickle=False)
    if feature.shape != (len(rows), 1959) or prediction.shape != (len(rows), 1):
        raise ValueError("Missing or reordered cohort dimensions")
    if feature.dtype != np.float32 or not np.isfinite(feature).all() or not np.isfinite(prediction).all():
        raise ValueError("Invalid feature or prediction values")
    replay = np.full(len(rows), np.nan, dtype=np.float32)
    folds = []
    for fold in range(10):
        indices = np.array([i for i, r in enumerate(rows) if r["fold"] == fold])
        if not len(indices):
            raise ValueError("Missing original fold")
        name = "models/" + FOLDERS[kind] + "/" + str(fold) + "_bestmodel.pth"
        if file_sha(source / name) != protocol["checkpoints"].get(name):
            raise ValueError("Author checkpoint changed")
        state = torch.load(source / name, map_location="cpu", weights_only=True)
        for start in range(0, len(indices), 512):
            batch = indices[start : start + 512]
            replay[batch] = numpy_forward(feature[batch], state, activity=kind == "activity")
        folds.append({"fold": fold, "rows": len(indices)})
    if not np.isfinite(replay).all() or not np.allclose(replay, prediction[:, 0], atol=1e-4, rtol=1e-5):
        raise ValueError("Independent full-row checkpoint replay differs")
    target = np.array([r["targets"][kind] for r in rows])
    measured = regression_metrics(prediction[:, 0], target)
    for key in ["rmse", "pcc", "scc"]:
        if not math.isfinite(measured[key]) or not math.isclose(
            measured[key], report["result"]["metrics"][key], abs_tol=1e-10, rel_tol=1e-10
        ):
            raise ValueError("Recomputed cohort metrics differ")
    return {
        "kind": kind,
        "rows": len(rows),
        "folds": folds,
        "metrics": measured,
        "independent_predictions_passed": True,
        "maximum_absolute_prediction_difference": float(np.max(np.abs(replay - prediction[:, 0]))),
        "files": {
            n: file_sha(folder / n)
            for n in [
                "protocol.json",
                "report.json",
                "features.npy",
                "predictions.npy",
                "results.json",
                "checkpoint_parity.json",
            ]
        },
    }


def audit_campaign(campaign, source):
    admitted = checked_manifest(source, "admission.json")
    checked_manifest(source, "checkpoints_admission_api.json")
    if admitted.get("commit") != COMMIT:
        raise ValueError("Unregistered CataPro source commit")
    execution, protocol = read(campaign / "execution.json"), read(campaign / "protocol.json")
    if execution.get("status") != "completed":
        raise ValueError("All three cohorts must be terminal before evidence reuse")
    if protocol["cohorts"] != [{"name": n, "expected_rows": c} for n, c in COUNTS.items()]:
        raise ValueError("Full original allocation required")
    records = execution.get("cohorts", [])
    if (
        len(records) != 3
        or [r["name"] for r in records] != list(COUNTS)
        or any(r["status"] != "completed" for r in records)
    ):
        raise ValueError("Missing completed cohort")
    for name, sha in protocol["source_sha256"].items():
        path = (campaign / "repository" / name).resolve()
        path.relative_to((campaign / "repository").resolve())
        if file_sha(path) != sha:
            raise ValueError("Frozen reproduction implementation changed")
    for kind in COUNTS:
        child = read(campaign / kind / "protocol.json")
        module = "src/agent/rsi/cases/catapro_case.py"
        if child["feature_contract"]["implementation_sha256"] != protocol["source_sha256"][module]:
            raise ValueError("Feature implementation differs from frozen campaign")
        if file_sha(campaign / kind / "implementation" / module) != protocol["source_sha256"][module]:
            raise ValueError("Archived feature implementation changed")
        entry = "scripts/rsi/cases/run_catapro_case.py"
        if file_sha(campaign / kind / "implementation" / entry) != protocol["source_sha256"][entry]:
            raise ValueError("Archived runner changed")
    for row in records:
        if row.get("report_sha256") != file_sha(campaign / row["name"] / "report.json"):
            raise ValueError("Controller cohort receipt changed")
    datasets = {kind: audit_cohort(campaign / kind, source, kind) for kind in COUNTS}
    return {
        "schema": 1,
        "source_commit": COMMIT,
        "datasets": datasets,
        "total_rows": sum(COUNTS.values()),
        "fold_checkpoints_replayed": 30,
        "all_rows_independently_replayed": True,
        "source_campaign": str(campaign.resolve()),
        "source_campaign_hashes": {n: file_sha(campaign / n) for n in ["protocol.json", "execution.json"]},
        "source_assets": {
            str(p.relative_to(source)): file_sha(p)
            for p in [
                source / "admission.json",
                source / "checkpoints_admission_api.json",
                *sorted((source / "datasets").glob("*.csv")),
                *sorted((source / "models").glob("*/*.pth")),
            ]
        },
        "scope": "Author checkpoint CV replay. Outer validation also selected checkpoints; no new training, autonomous new model or unbiased holdout claim.",
    }


def canonical_catapro(report):
    if report.get("execution_origin") not in {"new_computation", "verified_existing"}:
        raise ValueError("CataPro computation origin missing")
    audit = report.get("audit_evidence", {})
    datasets = audit.get("datasets", {})
    if (
        audit.get("source_commit") != COMMIT
        or audit.get("total_rows") != sum(COUNTS.values())
        or audit.get("fold_checkpoints_replayed") != 30
        or audit.get("all_rows_independently_replayed") is not True
        or set(datasets) != set(COUNTS)
    ):
        raise ValueError("Full CataPro evidence required")
    values = {}
    for name, count in COUNTS.items():
        row = datasets[name]
        folds = row.get("folds", [])
        if (
            row.get("rows") != count
            or row.get("independent_predictions_passed") is not True
            or len(folds) != 10
            or [f["fold"] for f in folds] != list(range(10))
            or sum(f["rows"] for f in folds) != count
            or row.get("metrics", {}).get("rows") != count
        ):
            raise ValueError("Incomplete CataPro row/fold coverage")
        metric = row["metrics"]
        if not all(type(metric[k]) in {int, float} and math.isfinite(metric[k]) for k in ["pcc", "scc", "rmse"]):
            raise ValueError("Invalid measured CataPro result")
        if metric["rmse"] < 0 or any(not -1 <= metric[k] <= 1 for k in ["pcc", "scc"]):
            raise ValueError("CataPro metric outside range")
        values[name] = metric
    return {
        "computational_scope": "catapro_raw_sequences_and_smiles_to_all_released_cv_predictions",
        "execution_origin": report["execution_origin"],
        "source_commit": COMMIT,
        "source_campaign": report["source_campaign"],
        "cohort_rows": dict(COUNTS),
        "fold_checkpoints_replayed": 30,
        "independent_full_row_replay_passed": True,
        "metrics": values,
        "km_output_unit": "mM",
        "author_validation_used_for_checkpoint_selection": True,
        "training_reproduced": False,
        "new_model_or_self_evolution_established": False,
        "independent_biological_replication": False,
    }


def verify_artifacts(report, output, source):
    canonical_catapro(report)
    root = source.parents[4]
    operation = report.get("operation")
    if operation == "verify_existing":
        registered = report.get("requested_campaign")
        if registered not in EXISTING or report.get("execution_origin") != "verified_existing":
            raise ValueError("Unregistered existing computation")
        expected = (root / EXISTING[registered]).resolve()
        if file_sha(expected / "protocol.json") != REGISTERED_PROTOCOL_SHA:
            raise ValueError("Registered original protocol changed")
    elif operation == "reproduce":
        if report.get("requested_campaign") is not None or report.get("execution_origin") != "new_computation":
            raise ValueError("Invalid new computation origin")
        expected = (output / "campaign").resolve()
    else:
        raise ValueError("Unknown CataPro operation")
    if report.get("source_campaign") != str(expected.relative_to(root)):
        raise ValueError("Reported campaign origin differs")
    if report["audit_evidence"].get("source_campaign") != str(expected):
        raise ValueError("Audited campaign origin differs")
    audit_path = output / "audit.json"
    if file_sha(audit_path) != report["audit_sha256"] or read(audit_path) != report["audit_evidence"]:
        raise ValueError("CataPro audit receipt changed")
    audit = report["audit_evidence"]
    campaign = Path(audit["source_campaign"]).resolve()
    for name, sha in audit["source_campaign_hashes"].items():
        if file_sha(campaign / name) != sha:
            raise ValueError("Source campaign changed after audit")
    for kind, row in audit["datasets"].items():
        for name, sha in row["files"].items():
            if file_sha(campaign / kind / name) != sha:
                raise ValueError("Source cohort changed after audit")
    for name, sha in audit["source_assets"].items():
        path = (source / name).resolve()
        path.relative_to(source.resolve())
        if file_sha(path) != sha:
            raise ValueError("CataPro source/checkpoint changed")
