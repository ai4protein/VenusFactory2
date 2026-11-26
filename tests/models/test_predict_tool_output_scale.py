"""Prediction MCP carries complete result interpretation without inventing units."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from mcp_tools.models.factory.target_transform import fit_target_transform, target_output_scale
from mcp_tools.models.factory.training import train_operations as operations


@pytest.fixture
def prediction(tmp_path):
    config = {"problem_type": "regression", "normalize": "quantile", "num_labels": 1}
    state = fit_target_transform([1.0, 2.0, 3.0, 4.0], "quantile", target_unit="nM")
    scale = target_output_scale(config, state, inverse_applied=True)
    output = tmp_path / "predictions.csv"
    frame = pd.DataFrame(
        {
            "sequence": ["AA"],
            "prediction": [2.0],
            "prediction_value_space": [scale["value_space"]],
            "prediction_unit": [scale["unit"]],
            "inverse_transform_applied": [True],
        }
    )
    frame.to_csv(output, index=False)
    sidecar = Path(str(output) + ".output-scale.json")
    sidecar.write_text(json.dumps(scale))
    return output, sidecar, frame, scale


def test_complete_quantile_interpretation_and_exact_artifact_path_are_returned(prediction):
    output, sidecar, frame, scale = prediction
    result = operations._prediction_scale_attachment(str(output), frame, "regression")
    assert result["output_scale"] == result["biological_metadata"]["output_scale"] == scale
    assert result["output_scale"]["inverse"]["approximate"] is True
    assert result["output_scale"]["inverse"]["bounds"] == "clamp_to_training_range"
    assert "outside [0,1]" in result["output_scale"]["inverse"]["limitations"]
    assert result["output_scale_file"] == str(sidecar)


@pytest.mark.parametrize("damage", ["schema", "unit", "flag", "inverse", "hash", "path", "unknown_method"])
def test_invalid_or_mismatched_metadata_is_rejected(prediction, damage):
    output, sidecar, frame, original = prediction
    scale = copy.deepcopy(original)
    if damage == "schema":
        scale["schema"] = "unknown"
    elif damage == "unit":
        scale["unit"] = "Celsius"
    elif damage == "flag":
        scale["inverse_transform_applied"] = "true"
    elif damage == "inverse":
        scale["inverse"]["approximate"] = False
    elif damage == "hash":
        scale["transform_sha256"] = "not-a-hash"
    elif damage == "path":
        scale["path"] = "/private/arbitrary.json"
    else:
        scale["normalization"] = "unknown"
    sidecar.write_text(json.dumps(scale))
    with pytest.raises(ValueError):
        operations._prediction_scale_attachment(str(output), frame, "regression")


def test_missing_new_sidecar_is_not_silently_downgraded_to_legacy(prediction):
    output, sidecar, frame, _ = prediction
    sidecar.unlink()
    with pytest.raises(ValueError, match="missing"):
        operations._prediction_scale_attachment(str(output), frame, "regression")


@pytest.mark.parametrize("damage", ["symlink", "fifo", "oversize", "bad_json"])
def test_sidecar_file_boundary_is_checked(prediction, damage):
    output, sidecar, frame, _ = prediction
    if damage == "symlink":
        sidecar.unlink()
        sidecar.symlink_to(output)
    elif damage == "fifo":
        import os

        sidecar.unlink()
        os.mkfifo(sidecar)
    elif damage == "oversize":
        sidecar.write_text(" " * 65537)
    else:
        sidecar.write_text("not JSON")
    with pytest.raises(ValueError):
        operations._prediction_scale_attachment(str(output), frame, "regression")


def test_legacy_csv_and_classification_do_not_gain_invented_scale(tmp_path, prediction):
    output = tmp_path / "legacy.csv"
    frame = pd.DataFrame({"prediction": [0.2]})
    frame.to_csv(output, index=False)
    assert operations._prediction_scale_attachment(str(output), frame, "regression") == {}
    output, sidecar, frame, _ = prediction
    sidecar.write_text("invalid stale regression metadata")
    assert operations._prediction_scale_attachment(str(output), frame, "single_label_classification") == {}


def test_legacy_normalized_metadata_keeps_model_space(tmp_path):
    config = {"problem_type": "regression", "normalize": "standard"}
    scale = target_output_scale(config)
    output = tmp_path / "legacy-new.csv"
    frame = pd.DataFrame(
        {
            "prediction": [-0.7],
            "prediction_value_space": ["standardized_target"],
            "prediction_unit": ["dimensionless"],
            "inverse_transform_applied": [False],
        }
    )
    Path(str(output) + ".output-scale.json").write_text(json.dumps(scale))
    result = operations._prediction_scale_attachment(str(output), frame, "regression")
    assert result["output_scale"]["inverse_transform_applied"] is False
    assert result["output_scale"]["value_space"] == "standardized_target"
    assert "inverse" not in result["output_scale"]


def test_successful_mcp_result_exposes_metadata_and_rejects_csv_disagreement(prediction, monkeypatch):
    from mcp_tools.models import execution
    from mcp_tools.models.factory.runners import checkpoint
    from mcp_tools.models.factory.training import model_registry
    from mcp_tools.tools.capability_policy import CapabilityStore

    output, sidecar, frame, scale = prediction
    config = output.parent / "config.json"
    config.write_text(json.dumps({"problem_type": "regression", "num_labels": 1, "model_path": "fixture.pt"}))
    input_file = output.parent / "input.csv"
    pd.DataFrame({"aa_seq": ["AA"]}).to_csv(input_file, index=False)
    monkeypatch.setattr(CapabilityStore, "assert_call", lambda *args, **kwargs: None)
    monkeypatch.setattr(operations, "workspace_root", lambda: None)
    monkeypatch.setattr(operations, "scoped_path", lambda value: value)
    monkeypatch.setattr(operations, "validate_config", lambda value, **kwargs: value)
    monkeypatch.setattr(model_registry, "resolve_registered_config_path", lambda value: value)
    monkeypatch.setattr(operations, "build_predict_command_list", lambda *args, **kwargs: ["fixture"])
    monkeypatch.setattr(checkpoint, "prepare", lambda command: SimpleNamespace(argv=command))
    monkeypatch.setattr(execution, "execute", lambda spec: "local fixture completed")
    result = json.loads(operations.run_predict_tool(str(config), csv_file=str(input_file)))
    assert result["success"] and result["output_scale"] == scale
    assert result["output_scale_file"] == str(sidecar)
    frame["prediction_value_space"] = "standardized_target"
    frame.to_csv(output, index=False)
    result = json.loads(operations.run_predict_tool(str(config), csv_file=str(input_file)))
    assert result["success"] is False
    assert "differs from CSV" in result["error"]
