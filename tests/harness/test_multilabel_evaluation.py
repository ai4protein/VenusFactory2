"""Masked probability contracts and independently cross-checked scoring formulas."""

import copy
import random

import pytest
from sklearn.metrics import average_precision_score, brier_score_loss

from agent.harness.research_evaluation import scores, training_view, validate_data
from agent.harness.research_multilabel import average_precision, parse_target, validate_prediction
from agent.harness.research_prediction_diagnostics import prediction_diagnostics


def data():
    def rows(split, count):
        return [
            {
                "id": f"{split}-{i}",
                "group": f"{split}-{i}",
                "inputs": {"x": f"{split}-{i}"},
                "target": {"A": i % 2, "B": int(i % 3 == 0) if split == "train" else None},
            }
            for i in range(count)
        ]

    return {
        "task": "multilabel",
        "labels": ["A", "B"],
        "provenance": "Synthetic software fixture",
        "grouping": "Independent synthetic examples",
        "splits": {s: rows(s, n) for s, n in [("train", 6), ("validation", 4), ("test", 3)]},
    }


def test_multilabel_targets_and_training_view_keep_masks():
    value = validate_data(data(), "supervised")
    training = training_view(value, "supervised")
    assert training["labels"] == ["A", "B"]
    assert {r["id"] for r in training["train"]} == {f"train-{i}" for i in range(6)}
    assert "validation" not in str(training) and "test" not in str(training)
    assert value["splits"]["test"][0]["target"]["B"] is None


@pytest.mark.parametrize(
    "damage",
    [
        "missing_names",
        "duplicate_names",
        "different_keys",
        "probability_target",
        "boolean",
        "unknown_all",
        "unobserved_train_label",
        "overlap",
        "invalid_name",
        "nonfinite",
    ],
)
def test_invalid_multilabel_data_fail_before_execution(damage):
    value = data()
    target = value["splits"]["train"][0]["target"]
    if damage == "missing_names":
        del value["labels"]
    elif damage == "duplicate_names":
        value["labels"] = ["A", "A"]
    elif damage == "different_keys":
        target["C"] = target.pop("B")
    elif damage == "probability_target":
        target["A"] = 0.3
    elif damage == "boolean":
        target["A"] = True
    elif damage == "unknown_all":
        target.update(A=None, B=None)
    elif damage == "unobserved_train_label":
        for row in value["splits"]["train"]:
            row["target"]["B"] = None
    elif damage == "overlap":
        value["splits"]["test"][0]["inputs"] = value["splits"]["train"][0]["inputs"]
    elif damage == "invalid_name":
        value["labels"][1] = " B "
    else:
        target["B"] = float("nan")
    with pytest.raises(ValueError):
        validate_data(value, "supervised")


@pytest.mark.parametrize(
    "text",
    [
        '{"A": 1, "A": 0, "B": 1}',
        '{"A": true, "B": 0}',
        '{"A": 0.5, "B": 1}',
        '{"A": NaN, "B": 0}',
        "[0, 1]",
        '{"A": null, "B": null}',
        '{"A": 1}',
        '{"A": 1, "B": {}}',
    ],
)
def test_invalid_csv_target_cells_do_not_become_class_labels(text):
    with pytest.raises(ValueError):
        parse_target(text)


def test_json_null_is_preserved():
    assert parse_target('{"A": 1, "B": null}') == {"A": 1, "B": None}


@pytest.mark.parametrize(
    "prediction",
    [
        {"A": 0.3},
        {"A": 0.3, "B": 0.2, "C": 0.1},
        {"A": 0.2, "B": None},
        {"A": True, "B": 0},
        {"A": -0.01, "B": 1},
        {"A": 0, "B": 1.01},
        {"A": 0, "B": float("nan")},
        {"A": "0", "B": 1},
    ],
)
def test_probability_contract_requires_all_labels_and_finite_bounds(prediction):
    with pytest.raises(ValueError):
        validate_prediction(prediction, ["A", "B"])


def test_multiple_high_probabilities_are_valid_and_unobserved_cells_are_excluded():
    train = [{"A": 0, "B": 1}, {"A": 1, "B": 0}]
    targets = [{"A": 1, "B": None}, {"A": 0, "B": None}]
    predictions = [{"A": 0.9, "B": 0.8}, {"A": 0.2, "B": 0.6}]
    measured = scores("multilabel", predictions, targets, train)
    assert measured["metric"] == "macro_brier" and measured["direction"] == "minimize"
    assert measured["score"] == pytest.approx(0.025)
    assert measured["reference"] == pytest.approx(0.25)
    details = measured["multilabel"]
    assert details["observed_cells"] == details["unobserved_cells"] == 2
    assert details["observed_label_count"] == 1
    assert details["per_label"]["B"]["brier"] is None
    assert details["macro_average_precision"] == details["macro_f1"] == 1
    changed = [{**p, "B": 0} for p in predictions]
    assert scores("multilabel", changed, targets, train) == measured


@pytest.mark.parametrize("seed", range(10))
def test_probability_metrics_agree_with_sklearn_and_handle_ties(seed):
    rng = random.Random(seed)
    y = [0, 1] + [rng.randrange(2) for _ in range(18)]
    p = [rng.randrange(5) / 4 for _ in y]
    assert average_precision(list(zip(p, y))) == pytest.approx(average_precision_score(y, p))
    targets = [{"A": a, "B": 1 - a} for a in y]
    predictions = [{"A": a, "B": 1 - a} for a in p]
    measured = scores("multilabel", predictions, targets, [{"A": 0, "B": 1}, {"A": 1, "B": 0}])
    assert measured["score"] == pytest.approx(brier_score_loss(y, p))


def test_single_class_average_precision_is_explicitly_undefined():
    assert average_precision([(0.1, 0), (0.5, 0)]) is None
    assert average_precision([(0.1, 1), (0.5, 1)]) is None
    assert average_precision([(0.5, 0), (0.5, 1)]) == 0.5
    assert average_precision([(0.4, 1), (0.7, 0)]) == 0.5


def test_macro_is_label_balanced_not_row_or_cell_weighted():
    targets = [{"A": 0, "B": 1}, {"A": 0, "B": None}, {"A": 0, "B": None}]
    predictions = [{"A": 0, "B": 0}] * 3
    result = scores("multilabel", predictions, targets, [{"A": 0, "B": 0}, {"A": 1, "B": 1}])
    assert result["score"] == 0.5
    assert result["multilabel"]["micro_brier"] == 0.25
    assert result["multilabel"]["macro_average_precision"] is None
    assert result["multilabel"]["ap_label_count"] == 0


def test_prediction_diagnostics_do_not_need_query_labels():
    train = [{"A": 0, "B": 0}, {"A": 1, "B": None}]
    result = prediction_diagnostics("multilabel", [{"A": 0.5, "B": 0}] * 4, train, split="validation")
    assert result["constant_predictions"] and result["matches_training_reference"]
    assert result["reference_matching_labels"] == ["A", "B"]
    assert result["distinct_predictions"] == 1
    changed = prediction_diagnostics("multilabel", [{"A": 0.1, "B": 0}, {"A": 0.9, "B": 0}], train, split="test")
    assert not changed["constant_predictions"]
    assert changed["constant_labels"] == ["B"]
    assert not {"predictions", "targets", "approved"} & changed.keys()


def test_episode_control_uses_support_and_training_fallback_without_query_fitting():
    train = [{"A": 0, "B": 0}, {"A": 1, "B": 1}]
    support = [{"id": "support", "inputs": {"x": "0"}, "target": {"A": 1, "B": None}}]
    targets = [{"A": 0, "B": 0}, {"A": 0, "B": 1}, {"A": 1, "B": None}]
    queries = [
        {
            "task_id": "small" if i == 2 else "large",
            "target": target,
            "inputs": {"x": str(i + 1), "support": copy.deepcopy(support)},
        }
        for i, target in enumerate(targets)
    ]
    predictions = [{"A": 1, "B": 0.5}] * 3
    result = scores("multilabel", predictions, targets, train, query_rows=queries)["episodes"]
    assert result["task_count"] == 2 and result["training_fallback_label_tasks"] == 2
    assert result["references"]["support_label_prevalence"]["matching_prediction_tasks"] == 2
    assert result["score"]["macro_score"] == pytest.approx((0.625 + 0) / 2)
    assert result["score"]["pooled_score"] == pytest.approx((2 / 3 + 0.25) / 2)
    queries[1]["inputs"]["support"][0]["target"]["A"] = 0
    with pytest.raises(ValueError, match="Support observations changed"):
        scores("multilabel", predictions, targets, train, query_rows=queries)
