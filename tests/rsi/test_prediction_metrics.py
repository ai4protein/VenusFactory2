import copy

import numpy as np
import pytest
from scipy.stats import spearmanr
from sklearn.metrics import log_loss, mean_absolute_error, mean_squared_error

from agent.rsi.engine.prediction_metrics import measure


def rows(values, prefix):
    return [{"id": prefix + str(i), "sequence": prefix + str(i), "target": v} for i, v in enumerate(values)]


def regression_fixture():
    train, future = rows([1, 3, 4], "train"), rows([2, 2, 5, 7], "future")
    predictions = [
        {"id": r["id"], "sequence": r["sequence"], "prediction": p} for r, p in zip(future, [2, 3, 3, 6], strict=True)
    ]
    return train, future, predictions


def test_regression_matches_independent_libraries_with_ties():
    train, future, predictions = regression_fixture()
    got = measure(train, future, predictions)
    y, p = [r["target"] for r in future], [r["prediction"] for r in predictions]
    assert got["mse"] == pytest.approx(mean_squared_error(y, p))
    assert got["mae"] == pytest.approx(mean_absolute_error(y, p))
    assert got["spearman"] == pytest.approx(spearmanr(y, p).statistic)
    assert got["mse_skill_vs_training_mean"] == pytest.approx(
        1 - mean_squared_error(y, p) / mean_squared_error(y, [8 / 3] * 4)
    )


def test_constant_rank_and_degenerate_baseline_are_missing_not_zero():
    train, future = rows([2, 2], "train"), rows([2, 2], "future")
    predictions = [{"id": r["id"], "sequence": r["sequence"], "prediction": 3.0} for r in future]
    got = measure(train, future, predictions)
    assert got["spearman"] is None and got["undefined_spearman"]
    assert got["mse_skill_vs_training_mean"] is None and got["degenerate_baseline"]
    assert got["mse"] == 1


@pytest.mark.parametrize(
    "attack", ["order", "missing", "id", "overlap_id", "overlap_sequence", "nan", "bool", "target_nan"]
)
def test_regression_rejects_invalid_or_misaligned_predictions(attack):
    train, future, predictions = regression_fixture()
    if attack == "order":
        predictions.reverse()
    elif attack == "missing":
        predictions.pop()
    elif attack == "id":
        predictions[0]["id"] = "wrong"
    elif attack == "overlap_id":
        train[0]["id"] = future[0]["id"]
    elif attack == "overlap_sequence":
        train[0]["sequence"] = future[0]["sequence"]
    elif attack == "nan":
        predictions[0]["prediction"] = float("nan")
    elif attack == "bool":
        predictions[0]["prediction"] = True
    else:
        future[0]["target"] = float("nan")
    with pytest.raises(ValueError):
        measure(train, future, predictions)


def classification_fixture():
    train, future = rows([0, 0, 1, 2], "train"), rows([0, 2, 2], "future")
    probabilities = [[0.7, 0.1, 0.2], [0.2, 0.3, 0.5], [0.4, 0.5, 0.1]]
    predictions = [
        {"id": r["id"], "sequence": r["sequence"], "probabilities": p, "prediction": int(np.argmax(p))}
        for r, p in zip(future, probabilities, strict=True)
    ]
    return train, future, predictions


def test_multiclass_scores_and_absent_class_are_explicit():
    train, future, predictions = classification_fixture()
    got = measure(train, future, predictions, classes=3)
    p = np.asarray([r["probabilities"] for r in predictions])
    y = [r["target"] for r in future]
    assert got["brier"] == pytest.approx(np.mean(np.sum((p - np.eye(3)[y]) ** 2, axis=1)))
    assert got["log_loss"] == pytest.approx(log_loss(y, p, labels=[0, 1, 2]))
    assert got["class_recall"] == [1, None, 0.5]
    assert got["balanced_accuracy"] == 0.75
    assert got["training_class_prior"] == [0.5, 0.25, 0.25]


@pytest.mark.parametrize("attack", ["negative", "sum", "shape", "nan", "wrong_class", "fractional_label"])
def test_invalid_classifier_output_is_rejected(attack):
    train, future, predictions = classification_fixture()
    predictions = copy.deepcopy(predictions)
    if attack == "negative":
        predictions[0]["probabilities"] = [-0.1, 0.5, 0.6]
    elif attack == "sum":
        predictions[0]["probabilities"] = [0.1, 0.2, 0.3]
    elif attack == "shape":
        predictions[0]["probabilities"] = [0.1, 0.9]
    elif attack == "nan":
        predictions[0]["probabilities"][0] = float("nan")
    elif attack == "wrong_class":
        predictions[0]["prediction"] = 2
    else:
        future[0]["target"] = 0.5
    with pytest.raises(ValueError):
        measure(train, future, predictions, classes=3)
