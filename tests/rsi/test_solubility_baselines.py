"""Published-reference and leakage checks; synthetic strings are numerical fixtures."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from agent.rsi.engine.solubility_baselines import REFERENCE, WEIGHTS, predict_swi_control, swi


def observations():
    training = [
        {"id": "a", "sequence": "A" * 20, "target": 10.0},
        {"id": "b", "sequence": "C" * 20, "target": 20.0},
        {"id": "c", "sequence": "E" * 20, "target": 40.0},
    ]
    query = [{"id": "q1", "sequence": "AC" * 10}, {"id": "q2", "sequence": "AE" * 10}]
    return training, query


def test_weights_match_independently_archived_author_reference():
    ref = json.loads((Path(__file__).parent / "fixtures/published_swi_reference.json").read_text())
    assert WEIGHTS == ref["weights"]
    assert REFERENCE["source_sha256"] == ref["source_sha256"]
    assert REFERENCE["source_git_blob_sha1"] == ref["source_git_blob_sha1"]
    assert set(WEIGHTS) == set("ACDEFGHIKLMNPQRSTVWY")


def test_raw_swi_reproduces_mean_weight_and_preserves_composition():
    assert swi("A" * 20) == WEIGHTS["A"]
    assert swi("AC" * 10) == pytest.approx((WEIGHTS["A"] + WEIGHTS["C"]) / 2, abs=1e-15)
    assert swi("ACDEFGHIKLMNPQRSTVWY") == swi("YWVTSRQPNMLKIHGFEDCA")


def test_affine_calibration_matches_independent_linear_solver():
    training, query = observations()
    result = predict_swi_control(training, query)
    x = np.array([np.mean([WEIGHTS[a] for a in row["sequence"]]) for row in training])
    y = np.array([row["target"] for row in training])
    coefficients = np.linalg.lstsq(np.column_stack([np.ones(len(x)), x]), y, rcond=None)[0]
    z = np.array([np.mean([WEIGHTS[a] for a in row["sequence"]]) for row in query])
    expected = coefficients[0] + coefficients[1] * z
    np.testing.assert_allclose([row["prediction"] for row in result["predictions"]], expected, rtol=1e-12, atol=1e-12)
    assert result["external_model_calls"] == result["new_autonomous_models"] == 0
    assert not result["calibration"]["clipping"]


def test_query_content_and_order_cannot_change_fitted_parameters():
    training, query = observations()
    baseline = predict_swi_control(training, query)
    query.reverse()
    query.append({"id": "q3", "sequence": "W" * 20})
    changed = predict_swi_control(training, query)
    assert baseline["calibration"] == changed["calibration"]
    assert baseline["training_sha256"] == changed["training_sha256"]


def test_query_labels_are_rejected():
    training, query = observations()
    query[0]["target"] = 99.0
    with pytest.raises(ValueError, match="only id and sequence"):
        predict_swi_control(training, query)


def test_constant_scores_have_identifiable_mean_fallback():
    training, query = observations()
    for row in training:
        row["sequence"] = "A" * 20
    result = predict_swi_control(training, query)
    assert result["calibration"]["constant_score_fallback"]
    assert result["calibration"]["slope"] == 0
    assert all(r["prediction"] == np.mean([t["target"] for t in training]) for r in result["predictions"])


@pytest.mark.parametrize("invalid", ["", "acde", "AX", "AA AA", "AA\nAA", None])
def test_invalid_sequences_are_not_silently_modified(invalid):
    with pytest.raises(ValueError, match="canonical uppercase"):
        swi(invalid)


@pytest.mark.parametrize("change", ["nan", "bool", "duplicate", "empty", "missing_sequence"])
def test_invalid_training_rows_fail(change):
    training, query = observations()
    if change == "nan":
        training[0]["target"] = float("nan")
    elif change == "bool":
        training[0]["target"] = True
    elif change == "duplicate":
        training.append(copy.deepcopy(training[0]))
    elif change == "empty":
        training.clear()
    else:
        del training[0]["sequence"]
    with pytest.raises(ValueError):
        predict_swi_control(training, query)
