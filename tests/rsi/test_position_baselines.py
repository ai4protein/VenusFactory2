"""Numerical controls; synthetic fixtures do not represent acquired models."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest
from test_research_project_catalog import manifest as manifest
from test_research_project_catalog import save, task_of

from agent.rsi.position_baselines import IncompatibleAlignment, predict_control, run_controls


@pytest.mark.parametrize("degree", [1, 2])
def test_categorical_kernel_agrees_with_independent_encoder_and_solver(degree):
    train = [{"sequence": s, "target": y} for s, y in [("AAAA", 1.0), ("CAAA", 3.0), ("ACAA", -2.0), ("CCAA", 4.0)]]
    query = [{"sequence": s} for s in ["DCAA", "ADAA", "AACA"]]
    values, independent, support = predict_control(train, query, degree)
    np.testing.assert_allclose(values, independent, rtol=1e-10, atol=1e-10)
    assert support["training_variable_positions"] == [0, 1]
    assert support["validation_rows_with_changes_at_training_invariant_positions"] == 1


def test_training_invariant_changes_are_not_learned_effects():
    train = [{"sequence": "AAAA", "target": 1.0}, {"sequence": "CAAA", "target": 3.0}]
    a, _, support = predict_control(train, [{"sequence": "ACAA"}, {"sequence": "AADA"}], 1)
    assert a[0] == a[1]
    assert support["validation_rows_with_changes_at_training_invariant_positions"] == 2


def test_no_variable_positions_has_explicit_constant_fallback():
    train = [{"sequence": "AAAA", "target": 1.0}, {"sequence": "AAAA", "target": 3.0}]
    pred, independent, support = predict_control(train, [{"sequence": "CAAA"}], 2)
    assert pred.tolist() == independent.tolist() == [2.0]
    assert support["training_variable_positions"] == []


@pytest.mark.parametrize("degree", [1, 2])
def test_query_labels_never_change_predictions(degree):
    train = [{"sequence": "AAAA", "target": 1.0}, {"sequence": "CAAA", "target": 3.0}]
    query = [{"sequence": "ACAA", "target": 0.0}]
    first = predict_control(train, query, degree)[0]
    query[0]["target"] = 1000.0
    np.testing.assert_array_equal(first, predict_control(train, query, degree)[0])


def test_rejects_alignment_assumptions_and_invalid_targets():
    train = [{"sequence": "AAAA", "target": 1.0}, {"sequence": "CAAA", "target": 3.0}]
    with pytest.raises(IncompatibleAlignment):
        predict_control(train, [{"sequence": "AAA"}], 1)
    with pytest.raises(ValueError, match="Canonical"):
        predict_control(train, [{"sequence": "AAAX"}], 1)
    train[0]["target"] = float("nan")
    with pytest.raises(ValueError, match="Finite"):
        predict_control(train, [{"sequence": "ACAA"}], 1)
    with pytest.raises(ValueError, match="predeclared"):
        predict_control(train, [{"sequence": "ACAA"}], 3)


def test_full_task_fit_ignores_validation_and_future_labels(manifest):
    task = task_of(manifest)
    first = run_controls(task)
    for phase in ("validation", "future"):
        path = Path(task["data"][phase]["path"])
        rows = json.loads(path.read_text())
        for row in rows:
            row["target"] += 1000
        task["data"][phase] = save(path, rows)
    second = run_controls(task)
    assert not first["future_scored"] and first["new_autonomous_models"] == 0
    for left, right in zip(first["outcomes"], second["outcomes"], strict=True):
        assert left["predictions"] == right["predictions"]
        assert left["metrics"]["mae"] != right["metrics"]["mae"]


def test_non_aligned_task_is_unsupported_not_scored(manifest):
    task = task_of(manifest)
    path = Path(task["data"]["validation"]["path"])
    rows = json.loads(path.read_text())
    rows[0]["sequence"] += "A"
    task["data"]["validation"] = save(path, rows)
    report = run_controls(task)
    assert report["status"] == "unsupported" and report["outcomes"] == []


def test_classifier_is_not_silently_regressed(manifest):
    task = copy.deepcopy(task_of(manifest))
    task["plugin_id"] = "generated_classifier.v1"
    with pytest.raises(ValueError, match="sequence-regression"):
        run_controls(task)


def test_batch_retains_failure_and_never_rewrites_output(manifest, tmp_path, monkeypatch):
    import run_position_baseline_batch as batch

    path = tmp_path / "batch_manifest.json"
    save(path, manifest)

    def fail(task):
        raise RuntimeError("Synthetic numerical failure")

    monkeypatch.setattr(batch.position_baselines, "run_controls", fail)
    output = tmp_path / "batch"
    result = batch.run(path, "project_one", output)
    assert result["allocated"] == result["failed"] == 1
    assert result["completed"] == result["new_autonomous_models"] == 0
    assert (output / "protocol.json").exists()
    with pytest.raises(FileExistsError):
        batch.run(path, "project_one", output)


def test_unknown_project_rejected_before_creating_output(manifest, tmp_path):
    import run_position_baseline_batch as batch

    path = tmp_path / "batch_manifest.json"
    save(path, manifest)
    output = tmp_path / "batch"
    with pytest.raises(ValueError, match="Project"):
        batch.run(path, "unadmitted", output)
    assert not output.exists()
