"""Host-only, immutable per-row assessment evidence; labels are never exported."""

import json
import tempfile
from pathlib import Path

from . import prediction_batches
from .protein_model import digest
from .storage import atomic_json
from .versions import VersionStore

REFERENCE_KEYS = {"schema", "version_id", "evidence_sha256", "predictions_sha256"}


def prediction_replay_summary(predictions, repeated_predictions):
    """Compare the complete output, including class probabilities and row order."""
    reference = digest(predictions)
    repeated = digest(repeated_predictions)
    return {
        "schema": 1,
        "rows": len(predictions),
        "reference_sha256": reference,
        "replayed_sha256": repeated,
        "exact_match": reference == repeated,
    }


def archive_assessment(
    root, model_id, model, policy, metrics, predictions, validation, executions, runtime,
    *, repeated_predictions=None, batch_predictions=None,
):
    if model.get("plugin_id") in {
        "generated_context.v1",
        "generated_context.v2",
        "generated_context.v3",
        "generated_assay.v1",
        "generated_antibody.v1",
    }:
        from .context_interfaces import module_for

        matching_rows = module_for(model["plugin_id"]).prediction_rows_match(predictions, validation)
    else:
        matching_rows = [(p["id"], p["sequence"]) for p in predictions] == [
            (p["id"], p["sequence"]) for p in validation
        ]
    if (
        digest(validation) != policy["validation_sha256"]
        or len(predictions) != len(validation)
        or metrics["rows"] != len(validation)
        or metrics["predictions_sha256"] != digest(predictions)
        or not matching_rows
    ):
        raise ValueError("Per-row evidence does not match evaluated predictions")
    evidence = {
        "schema": 3 if batch_predictions is not None else 2 if repeated_predictions is not None else 1,
        "model_id": model_id,
        "model_sha256": digest(model),
        "policy_sha256": digest(policy),
        "training_sha256": policy["training_sha256"],
        "validation_sha256": policy["validation_sha256"],
        "metrics": metrics,
        "predictions_sha256": digest(predictions),
        "host_runtime": runtime,
        "isolated_executions": executions,
        "scope": "Immutable assessment predictions and runtime diagnostics; exact replay requirements unchanged",
    }
    if repeated_predictions is not None:
        evidence["prediction_replay"] = prediction_replay_summary(predictions, repeated_predictions)
        evidence["scope"] = "Frozen assessment predictions, same-input prediction replay and runtime diagnostics"
    if policy.get("schema") == 2:
        from agent.rsi.datasets.model_tasks import inference_inputs

        if (repeated_predictions is None or batch_predictions is None
                or digest(batch_predictions.get("inputs")) != digest(inference_inputs(validation, model["plugin_id"]))):
            raise ValueError("Required prediction batch evidence is missing or changed")
        evidence[prediction_batches.FIELD] = prediction_batches.summarize(
            predictions, batch_predictions, policy[prediction_batches.FIELD], model["plugin_id"]
        )
        evidence["scope"] = "Frozen predictions, exact replay and predeclared per-observation batch review"
    elif batch_predictions is not None:
        raise ValueError("Prediction batch review was not predeclared")
    metadata = {"kind": "assessment_numerical_evidence", "model_id": model_id, "policy_sha256": digest(policy)}
    with tempfile.TemporaryDirectory(prefix="numerical-evidence-", dir=root) as tmp:
        folder = Path(tmp)
        atomic_json(folder / "evidence.json", evidence)
        atomic_json(folder / "predictions.json", predictions)
        if repeated_predictions is not None:
            atomic_json(folder / "repeated_predictions.json", repeated_predictions)
        if batch_predictions is not None:
            atomic_json(folder / "batch_predictions.json", batch_predictions)
        manifest = VersionStore(root / "numerical_evidence").snapshot(folder, metadata=metadata)
    return {
        "schema": 1,
        "version_id": manifest["version_id"],
        "evidence_sha256": digest(evidence),
        "predictions_sha256": digest(predictions),
    }


def verify_assessment(root, model_id, model, policy, report):
    reference = report.get("numerical_evidence")
    if "numerical_evidence" not in report:
        if policy.get("schema") == 2 or prediction_batches.FIELD in report:
            raise ValueError("Required prediction batch evidence is missing")
        return {"status": "legacy_without_per_row_evidence"}
    if not isinstance(reference, dict) or set(reference) != REFERENCE_KEYS or reference["schema"] != 1:
        raise ValueError("Invalid numerical evidence reference")
    versions = VersionStore(root / "numerical_evidence")
    manifest = versions.verify(reference["version_id"])
    expected_metadata = {"kind": "assessment_numerical_evidence", "model_id": model_id, "policy_sha256": digest(policy)}
    files = set(manifest["files"])
    original_files = {"evidence.json", "predictions.json"}
    replay_files = original_files | {"repeated_predictions.json"}
    batch_files = replay_files | {"batch_predictions.json"}
    if files not in (original_files, replay_files, batch_files) or manifest["metadata"] != expected_metadata:
        raise ValueError("Numerical evidence scope changed")
    folder = root / "numerical_evidence/versions" / reference["version_id"]
    evidence = json.loads((folder / "evidence.json").read_text())
    predictions = json.loads((folder / "predictions.json").read_text())
    if (
        digest(evidence) != reference["evidence_sha256"]
        or evidence["schema"] not in (1, 2, 3)
        or evidence["model_id"] != model_id
        or evidence["model_sha256"] != digest(model)
        or evidence["policy_sha256"] != digest(policy)
        or evidence["training_sha256"] != policy["training_sha256"]
        or evidence["validation_sha256"] != policy["validation_sha256"]
        or evidence["metrics"] != report["metrics"]
        or evidence["predictions_sha256"] != reference["predictions_sha256"]
        or digest(predictions) != reference["predictions_sha256"]
        or digest(predictions) != report["metrics"]["predictions_sha256"]
        or len(predictions) != report["metrics"]["rows"]
    ):
        raise ValueError("Numerical evidence binding changed")
    if evidence["schema"] in (2, 3):
        if files != (batch_files if evidence["schema"] == 3 else replay_files):
            raise ValueError("Prediction replay evidence is missing")
        repeated = json.loads((folder / "repeated_predictions.json").read_text())
        replay = prediction_replay_summary(predictions, repeated)
        if (
            evidence.get("prediction_replay") != replay
            or report.get("prediction_replay") != replay
            or (report.get("passed") and not replay["exact_match"])
        ):
            raise ValueError("Prediction replay binding changed")
    elif files != original_files or "prediction_replay" in report:
        raise ValueError("Legacy assessment has no prediction replay evidence")
    if policy.get("schema") == 2:
        if evidence["schema"] != 3:
            raise ValueError("Required prediction batch evidence is missing")
        batches = json.loads((folder / "batch_predictions.json").read_text())
        reviewed = prediction_batches.summarize(
            predictions, batches, policy[prediction_batches.FIELD], model["plugin_id"]
        )
        if (evidence.get(prediction_batches.FIELD) != reviewed
                or report.get(prediction_batches.FIELD) != reviewed
                or (report.get("passed") and not reviewed["passed"])):
            raise ValueError("Prediction batch review binding changed")
    elif evidence["schema"] == 3 or prediction_batches.FIELD in report or prediction_batches.FIELD in evidence:
        raise ValueError("Legacy assessment has no prediction batch review evidence")
    return {"status": "verified", "evidence": evidence, "predictions": predictions}
