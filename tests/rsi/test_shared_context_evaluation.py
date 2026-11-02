import copy
from pathlib import Path

import pytest
from test_context_v2 import make_task, save

from agent.rsi import shared_context_evaluation as evaluation
from agent.rsi.protein_model import digest


def prediction_rows(rows):
    return [
        {
            "id": r["id"],
            "protein": r["protein"],
            "group": r["group"],
            "values_sha256": digest(r["values"]),
            "prediction": r["target"],
        }
        for r in rows
    ]


def test_complete_controls_and_unregistered_target_remain_visible(tmp_path):
    task = make_task(tmp_path / "fixture")[0]
    result = evaluation.evaluate_task(task, None)
    assert result["scientist"]["status"] == "not_registered"
    assert result["scientist"]["metrics"] is result["comparison"] is None
    assert len(result["controls"]) == 4
    assert all(r["status"] == "completed" for r in result["controls"])
    assert result["fit_partition"] == "training" and result["score_partition"] == "future"
    assert result["groups"] == {"training": 4, "validation": 1, "future": 1}


def test_heldout_labels_affect_metrics_but_not_fitted_predictions(tmp_path):
    task = make_task(tmp_path / "fixture")[0]
    future = evaluation.load_task(task)["future"]
    predictions = prediction_rows(future)
    first = evaluation.evaluate_task(task, predictions)
    assert first["scientist"]["metrics"]["rmse"] == 0
    for row in future:
        row["target"] += 100
    task["data"]["future"] = save(Path(task["data"]["future"]["path"]), future)
    second = evaluation.evaluate_task(task, predictions)
    assert [c["predictions"] for c in first["controls"]] == [c["predictions"] for c in second["controls"]]
    assert second["scientist"]["metrics"]["rmse"] == pytest.approx(100)
    assert first["query_sha256"] == second["query_sha256"]


@pytest.mark.parametrize("field", ["protein", "group", "values_sha256", "id"])
def test_context_prediction_identity_is_strict(tmp_path, field):
    task = make_task(tmp_path / "fixture")[0]
    predictions = prediction_rows(evaluation.load_task(task)["future"])
    predictions[0][field] = "changed"
    with pytest.raises(ValueError):
        evaluation.evaluate_task(task, predictions)


def test_wrong_interface_fails_before_loading_labels():
    with pytest.raises(ValueError, match="Context"):
        evaluation.evaluate_task({"plugin_id": "generated_sequence.v1"}, None)


def test_failed_control_is_not_dropped(tmp_path, monkeypatch):
    task = make_task(tmp_path / "fixture")[0]
    baseline = evaluation.run_controls(task, score_partition="future")
    baseline["outcomes"][1] = {"method": "condition_ridge", "status": "failed", "error": "Synthetic failure"}
    monkeypatch.setattr(evaluation, "run_controls", lambda *a, **k: copy.deepcopy(baseline))
    result = evaluation.evaluate_task(task, None)
    assert len(result["controls"]) == 4 and result["controls"][1]["status"] == "failed"
    assert result["external_model_calls"] == result["new_autonomous_models"] == 0
