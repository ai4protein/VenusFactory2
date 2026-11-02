"""Source-bound, read-only acceptance of all five CataPro training stages."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .catapro_case import DATASETS, checked_manifest, dataset_rows, file_sha
from .catapro_evidence import COMMIT, COUNTS
from .catapro_training import PARAMETERS, STAGES, audit_stage, training_arguments

EXISTING = {"catapro_training_20261017": "research_runs/rsi/20261017_catapro_training_full_v1"}
REGISTERED_PROTOCOL_SHA = "c7e13b9cc199b307cb9e3a8cdd08798d9967ab5a7a628b01d0485bc0561dfb5c"
FEATURE_ROOT = "research_runs/rsi/20261017_catapro_full_cohorts_v1"
AUTHOR_FILES = [
    f"training/{family}/{name}.py"
    for family in ["kcat", "Km", "kcat_over_Km"]
    for name in ["training", "model", "utils"]
]


def read(path):
    return json.loads(path.read_text())


def verify_bound_files(root, bindings):
    for name, expected in bindings.items():
        path = root / name
        if Path(name).is_absolute() or path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("Evidence path escapes its admitted root")
        if not path.is_file() or file_sha(path) != expected:
            raise ValueError("Bound evidence changed: " + name)


def validate_terminal_protocol(campaign):
    protocol, report = read(campaign / "protocol.json"), read(campaign / "report.json")
    if report.get("status") != "completed":
        raise ValueError("Only a terminal completed training campaign can be audited")
    if report.get("protocol_sha256") != file_sha(campaign / "protocol.json"):
        raise ValueError("Training protocol changed")
    if (
        protocol.get("source_commit") != COMMIT
        or protocol.get("epochs") != 150
        or protocol.get("training_fits") != 50
        or protocol.get("stages") != list(STAGES)
        or protocol.get("parameters") != PARAMETERS
        or protocol.get("alpha", {}).get("value") != 0.4
        or set(protocol.get("tables", {})) != set(STAGES)
    ):
        raise ValueError("Full original-fold 150-cap training protocol required")
    stages = report.get("stages", [])
    if [s.get("stage") for s in stages] != list(STAGES) or any(
        s.get("status") != "completed" or s.get("returncode") != 0 for s in stages
    ):
        raise ValueError("All five completed training stages required")
    expected_author = {"author_source/" + n for n in AUTHOR_FILES}
    if set(protocol.get("author_source", {})) != expected_author:
        raise ValueError("All nine unchanged author training files required")
    if not protocol.get("implementation") or not protocol.get("reconstruction_decisions"):
        raise ValueError("Frozen implementation and reconstruction decisions required")
    return protocol, report


def verify_table(campaign, stage, binding, rows, features):
    import pandas as pd

    verify_bound_files(campaign, {binding["path"]: binding["sha256"]})
    table = pd.read_pickle(campaign / binding["path"])
    target = STAGES[stage][1]
    if (
        binding["rows"] != len(rows)
        or binding["cohort"] != STAGES[stage][0]
        or binding["target"] != target
        or table.shape != (len(rows), 1962 if target == "activity" else 1961)
        or table.index.tolist() != [r["id"] for r in rows]
        or not np.array_equal(table.iloc[:, :1959].to_numpy(), features)
        or not np.array_equal(table.iloc[:, -1].to_numpy(), [r["fold"] for r in rows])
    ):
        raise ValueError("Training table lost original rows, features or folds")
    if target == "activity":
        observed = np.log10(table.iloc[:, -3].to_numpy() / table.iloc[:, -2].to_numpy())
    else:
        observed = table.iloc[:, -2].to_numpy()
    expected = np.asarray([r["targets"][target] for r in rows])
    if not np.allclose(observed, expected, atol=2e-6, rtol=0):
        raise ValueError("Training labels or units differ from raw source")


def audit_training_campaign(campaign, source, features, output):
    """Recompute predictions into a new receipt; never rewrite the old campaign."""
    protocol, report = validate_terminal_protocol(campaign)
    if checked_manifest(source, "admission.json")["commit"] != COMMIT:
        raise ValueError("Wrong author source revision")
    verify_bound_files(campaign, protocol["author_source"] | protocol["implementation"])
    for name in AUTHOR_FILES:
        if file_sha(source / name) != protocol["author_source"]["author_source/" + name]:
            raise ValueError("Training did not use unchanged admitted author code")
    if protocol["alpha"]["source_sha256"] != file_sha(source / "inference/act_model.py"):
        raise ValueError("Activity mixture reference changed")
    output.mkdir(parents=True, exist_ok=False)
    assets = {
        "protocol.json": file_sha(campaign / "protocol.json"),
        "report.json": file_sha(campaign / "report.json"),
        **protocol["author_source"],
        **protocol["implementation"],
    }
    feature_assets, stage_audits = {}, {}
    for cohort, count in COUNTS.items():
        rows = dataset_rows(source, cohort)
        meta = protocol["inputs"][cohort]
        if (
            len(rows) != count
            or meta["rows"] != count
            or meta["raw_csv_sha256"] != file_sha(source / "datasets" / DATASETS[cohort])
        ):
            raise ValueError("Training cohort differs from complete raw source")
        for name, key in [
            ("features.npy", "features_sha256"),
            ("protocol.json", "feature_protocol_sha256"),
            ("report.json", "feature_report_sha256"),
        ]:
            feature_assets[cohort + "/" + name] = meta[key]
        verify_bound_files(features, feature_assets)
        matrix = np.load(features / cohort / "features.npy", mmap_mode="r", allow_pickle=False)
        if matrix.shape != (count, 1959) or matrix.dtype != np.float32 or not np.isfinite(matrix).all():
            raise ValueError("Full reconstructed feature matrix required")
        for stage, (dataset, _, _) in STAGES.items():
            if dataset != cohort:
                continue
            binding = protocol["tables"][stage]
            verify_table(campaign, stage, binding, rows, matrix)
            assets[binding["path"]] = binding["sha256"]
            folder = campaign / "stages" / stage
            runtime = read(folder / "runtime.json")
            if (
                runtime["arguments"] != training_arguments(campaign, stage, 150)
                or runtime["seed"] != protocol["stage_seeds"][stage]
            ):
                raise ValueError("Actual training arguments or seed differ from frozen protocol")
            saved = next(s for s in report["stages"] if s["stage"] == stage)
            if file_sha(folder / "audit.json") != saved["audit_sha256"]:
                raise ValueError("Original stage audit changed")
            original = read(folder / "audit.json")
            replay = audit_stage(campaign, stage, output=output / stage)
            if (
                original.get("status") != "passed"
                or original.get("checkpoints") != replay["checkpoints"]
                or original.get("independent_predictions_sha256") != replay["independent_predictions_sha256"]
            ):
                raise ValueError("Fresh audit differs from recorded trained checkpoint evidence")
            stage_audits[stage] = replay
            for name in [
                "runtime.json",
                "audit.json",
                "final_results.csv",
                "final_pred_label.csv",
                "independent_predictions.npy",
                *[f"{fold}_bestmodel.pth" for fold in range(10)],
            ]:
                assets[f"stages/{stage}/{name}"] = file_sha(folder / name)
    # Bind both the original historical outputs and the new read-only receipt.
    verify_bound_files(campaign, assets)
    audit = {
        "source_commit": COMMIT,
        "source_campaign": str(campaign.resolve()),
        "epochs_cap": 150,
        "training_fits": 50,
        "all_rows_independently_replayed": True,
        "cohort_rows": dict(COUNTS),
        "stages": stage_audits,
        "campaign_files": assets,
        "feature_files": feature_assets,
        "source_files": {
            n: file_sha(source / n)
            for n in [
                "admission.json",
                "inference/act_model.py",
                *AUTHOR_FILES,
                *["datasets/" + n for n in DATASETS.values()],
            ]
        },
        "replay_files": {str(p.relative_to(output)): file_sha(p) for p in output.rglob("*") if p.is_file()},
        "reconstruction_decisions": protocol["reconstruction_decisions"],
        "alpha_decision": protocol["alpha"],
    }
    return audit


def canonical_training(report):
    audit = report.get("audit_evidence", {})
    stages = audit.get("stages", {})
    if (
        report.get("status") != "completed"
        or report.get("execution_origin") not in {"new_computation", "verified_existing"}
        or audit.get("source_commit") != COMMIT
        or audit.get("epochs_cap") != 150
        or audit.get("training_fits") != 50
        or audit.get("cohort_rows") != COUNTS
        or audit.get("all_rows_independently_replayed") is not True
        or set(stages) != set(STAGES)
        or not audit.get("reconstruction_decisions")
        or audit.get("alpha_decision", {}).get("value") != 0.4
    ):
        raise ValueError("Complete fifty-fit training evidence required")
    values = {}
    for stage, (cohort, _, _) in STAGES.items():
        row = stages[stage]
        checkpoints = row.get("checkpoints", [])
        if (
            row.get("status") != "passed"
            or row.get("epochs_cap") != 150
            or row.get("rows") != COUNTS[cohort]
            or row.get("folds") != 10
            or len(checkpoints) != 10
            or [c.get("fold") for c in checkpoints] != list(range(10))
            or sum(c.get("validation_rows", 0) for c in checkpoints) != COUNTS[cohort]
        ):
            raise ValueError("Missing full-row training stage or fold")
        metric = row.get("metrics", {})
        if set(metric) != {"PCC", "SCC", "R2", "RMSE"} or not all(
            type(v) in {int, float} and math.isfinite(v) for v in metric.values()
        ):
            raise ValueError("Invalid training metric")
        if metric["RMSE"] < 0 or metric["R2"] > 1 or any(not -1 <= metric[k] <= 1 for k in ["PCC", "SCC"]):
            raise ValueError("Training metric outside range")
        values[stage] = metric
    return {
        "computational_scope": "catapro_verified_features_to_fifty_author_training_fits",
        "execution_origin": report["execution_origin"],
        "source_campaign": report["source_campaign"],
        "source_commit": COMMIT,
        "cohort_rows": dict(COUNTS),
        "training_fits": 50,
        "training_reproduced": True,
        "independent_full_row_replay_passed": True,
        "features_reused_from_verified_reconstruction": True,
        "metrics": values,
        "reconstruction_decisions": audit["reconstruction_decisions"],
        "alpha_decision": audit["alpha_decision"],
        "author_validation_used_for_checkpoint_selection": True,
        "exact_author_stochastic_equivalence": False,
        "original_grid_search_reproduced": False,
        "new_model_or_self_evolution_established": False,
        "independent_biological_replication": False,
    }


def verify_artifacts(report, output, root):
    canonical_training(report)
    operation = report.get("operation")
    if operation == "verify_existing":
        if report.get("requested_campaign") not in EXISTING or report["execution_origin"] != "verified_existing":
            raise ValueError("Unregistered training campaign")
        campaign = root / EXISTING[report["requested_campaign"]]
        if file_sha(campaign / "protocol.json") != REGISTERED_PROTOCOL_SHA:
            raise ValueError("Registered training protocol changed")
    elif (
        operation == "reproduce"
        and report.get("requested_campaign") is None
        and report["execution_origin"] == "new_computation"
    ):
        campaign = output / "campaign"
    else:
        raise ValueError("Training operation origin differs")
    if report["source_campaign"] != str(campaign.relative_to(root)) or report["audit_evidence"][
        "source_campaign"
    ] != str(campaign.resolve()):
        raise ValueError("Training campaign identity changed")
    if (
        file_sha(output / "audit.json") != report["audit_sha256"]
        or read(output / "audit.json") != report["audit_evidence"]
    ):
        raise ValueError("Training audit receipt changed")
    protocol, _ = validate_terminal_protocol(campaign)
    audit = report["audit_evidence"]
    expected_campaign = {
        "protocol.json",
        "report.json",
        *protocol["author_source"],
        *protocol["implementation"],
        *[v["path"] for v in protocol["tables"].values()],
    }
    for stage in STAGES:
        expected_campaign.update(
            f"stages/{stage}/{name}"
            for name in [
                "runtime.json",
                "audit.json",
                "final_results.csv",
                "final_pred_label.csv",
                "independent_predictions.npy",
                *[f"{fold}_bestmodel.pth" for fold in range(10)],
            ]
        )
    expected_features = {
        cohort + "/" + name for cohort in COUNTS for name in ["features.npy", "protocol.json", "report.json"]
    }
    expected_source = {
        "admission.json",
        "inference/act_model.py",
        *AUTHOR_FILES,
        *["datasets/" + n for n in DATASETS.values()],
    }
    expected_replay = {stage + "/" + name for stage in STAGES for name in ["audit.json", "independent_predictions.npy"]}
    if any(
        set(audit.get(key, {})) != expected
        for key, expected in [
            ("campaign_files", expected_campaign),
            ("feature_files", expected_features),
            ("source_files", expected_source),
            ("replay_files", expected_replay),
        ]
    ):
        raise ValueError("Training receipt omits required source, model or replay bindings")
    verify_bound_files(campaign, audit["campaign_files"])
    verify_bound_files(root / FEATURE_ROOT, audit["feature_files"])
    verify_bound_files(root / "research_runs/rsi/case_assets/CataPro" / COMMIT, audit["source_files"])
    verify_bound_files(output / "replay", audit["replay_files"])
