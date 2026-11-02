"""Fixed controls exercised on synthetic condition-response measurements."""

from pathlib import Path

import numpy as np
import pytest
from test_context_integration import make_task, save

from agent.rsi.context_baselines import METHODS, THERMAL_METHODS, thermal_predictions
from agent.rsi.traditional_baselines import run_controls


def prediction_map(result):
    return {r["method"]: r["predictions"] for r in result["outcomes"] if r["status"] == "completed"}


def test_six_actual_controls_and_no_future_label_or_validation_fit(tmp_path):
    task, data, _ = make_task(tmp_path)
    first = run_controls(task)
    assert set(prediction_map(first)) == set(METHODS + THERMAL_METHODS)
    assert first["groups"] == {"training": 4, "validation": 1}
    assert not first["future_scored"] and first["new_autonomous_models"] == 0
    for row in data["future"]:
        row["target"] += 1000
    task["data"]["future"] = save(Path(task["data"]["future"]["path"]), data["future"])
    second = run_controls(task)
    assert first["outcomes"][0]["metrics"] == second["outcomes"][0]["metrics"]
    assert prediction_map(first) == prediction_map(second)
    for row in data["validation"]:
        row["target"] += 10
    task["data"]["validation"] = save(Path(task["data"]["validation"]["path"]), data["validation"])
    third = run_controls(task)
    assert prediction_map(first) == prediction_map(third)
    assert first["outcomes"][0]["metrics"] != third["outcomes"][0]["metrics"]


def test_thermal_reference_conditions_are_fitted_separately():
    x = np.asarray([[t, c] for c in [0, 1] for t in range(30, 70, 5)], dtype=float)
    y = np.asarray([2 - t / 100 + c for t, c in x])
    queries = np.asarray([[42.0, 0.0], [42.0, 1.0]])
    predictions = thermal_predictions("thermal_isotonic", x, y, queries, 0)
    assert predictions == pytest.approx([1.58, 2.58])
    with pytest.raises(ValueError, match="no training observations"):
        thermal_predictions("thermal_isotonic", x, y, np.asarray([[42.0, 2.0]]), 0)


def test_unsupported_negative_response_keeps_failed_sigmoid_visible(tmp_path):
    task, data, _ = make_task(tmp_path)
    for row in data["training"]:
        row["target"] = -1.0
    task["data"]["training"] = save(Path(task["data"]["training"]["path"]), data["training"])
    outcomes = {r["method"]: r for r in run_controls(task)["outcomes"]}
    assert outcomes["thermal_sigmoid"]["status"] == "failed"
    assert all(outcomes[name]["status"] == "completed" for name in METHODS)
