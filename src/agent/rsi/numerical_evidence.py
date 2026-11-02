"""Host-only, immutable per-row assessment evidence; labels are never exported."""

import json
import tempfile
from pathlib import Path

from .protein_model import digest
from .storage import atomic_json
from .versions import VersionStore

REFERENCE_KEYS = {"schema", "version_id", "evidence_sha256", "predictions_sha256"}


def archive_assessment(root, model_id, model, policy, metrics, predictions, validation, executions, runtime):
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
        "schema": 1,
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
    metadata = {"kind": "assessment_numerical_evidence", "model_id": model_id, "policy_sha256": digest(policy)}
    with tempfile.TemporaryDirectory(prefix="numerical-evidence-", dir=root) as tmp:
        folder = Path(tmp)
        atomic_json(folder / "evidence.json", evidence)
        atomic_json(folder / "predictions.json", predictions)
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
        return {"status": "legacy_without_per_row_evidence"}
    if not isinstance(reference, dict) or set(reference) != REFERENCE_KEYS or reference["schema"] != 1:
        raise ValueError("Invalid numerical evidence reference")
    versions = VersionStore(root / "numerical_evidence")
    manifest = versions.verify(reference["version_id"])
    expected_metadata = {"kind": "assessment_numerical_evidence", "model_id": model_id, "policy_sha256": digest(policy)}
    if set(manifest["files"]) != {"evidence.json", "predictions.json"} or manifest["metadata"] != expected_metadata:
        raise ValueError("Numerical evidence scope changed")
    folder = root / "numerical_evidence/versions" / reference["version_id"]
    evidence = json.loads((folder / "evidence.json").read_text())
    predictions = json.loads((folder / "predictions.json").read_text())
    if (
        digest(evidence) != reference["evidence_sha256"]
        or evidence["schema"] != 1
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
    return {"status": "verified", "evidence": evidence, "predictions": predictions}
