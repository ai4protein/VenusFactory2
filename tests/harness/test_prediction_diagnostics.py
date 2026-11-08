"""Behavioral diagnostics are evidence, not a model-quality verdict."""
import json
import pytest
from agent.harness.research_prediction_diagnostics import prediction_diagnostics
from agent.harness.research_development import scores


def test_regression_constant_matches_train_mean_without_query_labels():
    result = prediction_diagnostics("regression", [2., 2., 2.], [1., 2., 3.], split="validation")
    assert result["constant_predictions"] and result["matches_training_reference"]
    assert result["distinct_predictions"] == 1 and result["prediction_count"] == 3
    assert result["reference_method"] == "training_mean"
    assert set(result["warnings"]) == {"constant_predictions", "matches_training_reference"}
    assert not {"targets", "predictions", "approved", "passed"} & result.keys()
    assert len(json.dumps(result)) < 700


def test_equal_error_does_not_imply_equal_predictions():
    targets, train = [0., 2.], [0., 2.]
    score = scores("regression", [0., 4.], targets, train)
    assert score["score"] == score["reference"]
    result = prediction_diagnostics("regression", [0., 4.], train, split="validation")
    assert not result["constant_predictions"] and not result["matches_training_reference"]
    assert result["warnings"] == []


def test_constant_nonreference_predictor_is_distinguished():
    result = prediction_diagnostics("regression", [7., 7.], [1., 2., 3.], split="test")
    assert result["constant_predictions"] and not result["matches_training_reference"]
    assert result["warnings"] == ["constant_predictions"]
    assert result["split"] == "test"


def test_numerical_roundoff_does_not_hide_reference_behavior():
    result = prediction_diagnostics("regression", [2., 2. + 1e-12], [1., 3.], split="validation")
    assert result["constant_predictions"] and result["matches_training_reference"]
    assert result["distinct_predictions"] == 2
    changed = prediction_diagnostics("regression", [2., 2.01], [1., 3.], split="validation")
    assert not changed["constant_predictions"]


@pytest.mark.parametrize("predictions,train,constant,matches", [
    (["soluble", "soluble"], ["soluble", "soluble", "insoluble"], True, True),
    (["insoluble", "insoluble"], ["soluble", "soluble", "insoluble"], True, False),
    (["soluble", "insoluble"], ["soluble", "soluble", "insoluble"], False, False),
    ([1, 1], [1, 0, 1], True, True),
    (["1", "1"], [1, 0, 1], True, False),
])
def test_classification_behavior(predictions, train, constant, matches):
    result = prediction_diagnostics("classification", predictions, train, split="validation")
    assert result["constant_predictions"] is constant
    assert result["matches_training_reference"] is matches
    assert result["reference_method"] == "training_majority"


@pytest.mark.parametrize("task,predictions,train,split", [
    ("regression", [], [1], "validation"),
    ("regression", [1], [], "validation"),
    ("regression", [float("nan")], [1], "validation"),
    ("regression", [1], [float("inf")], "validation"),
    ("regression", ["1"], [1], "validation"),
    ("classification", [True], [1], "validation"),
    ("classification", [float("inf")], [1], "validation"),
    ("classification", [{"label":1}], [1], "validation"),
    ("unsupported", [1], [1], "validation"),
    ("regression", [1], [1], "unknown"),
])
def test_invalid_inputs_fail_before_feedback(task, predictions, train, split):
    with pytest.raises(ValueError):
        prediction_diagnostics(task, predictions, train, split=split)
