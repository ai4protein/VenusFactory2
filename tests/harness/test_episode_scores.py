"""Episode metrics expose task imbalance without exposing withheld labels."""
import copy
import json
import pytest
from agent.harness.research_evaluation import scores


def rows(task, targets, support):
    return [{"task_id": task, "target": y, "inputs": {"query": str(i), "support": [
        {"id": "private-support-"+str(j), "inputs": {"x": str(j)}, "target": value}
        for j, value in enumerate(support)]}} for i, y in enumerate(targets)]


def measure(data, predictions, task="regression"):
    train = [0., 2.] if task == "regression" else ["yes", "no"]
    return scores(task, predictions, [row["target"] for row in data], train, query_rows=data)


def test_pooled_gain_can_hide_worse_equal_task_error():
    data = rows("many-queries", [0.]*9, [0., 2.])+rows("one-query", [0.], [0., 2.])
    before = measure(data, [2.]*9+[0.])
    after = measure(data, [0.]*9+[5.])
    assert after["score"] < before["score"]
    assert before["episodes"]["score"]["macro_score"] == 1.
    assert after["episodes"]["score"]["macro_score"] == 2.5
    assert after["episodes"]["score"]["pooled_score"] == after["score"]
    assert after["episodes"]["task_count"] == 2


def test_support_controls_are_task_specific_and_do_not_fit_query_labels():
    data = rows("A", [1000., 1000.], [1., 1., 10.])+rows("B", [2000.], [7., 7., 16.])
    result = measure(data, [1., 1., 7.])["episodes"]
    assert result["references"]["support_mean"]["macro_score"] == (996.+1990.)/2
    assert result["references"]["support_median"]["macro_score"] == (999.+1993.)/2
    assert result["references"]["support_median"]["matching_prediction_tasks"] == 2
    assert result["references"]["support_mean"]["matching_prediction_tasks"] == 0
    assert result["constant_within_task_count"] == 2
    text = json.dumps(result)
    assert all(secret not in text for secret in ("private-support", "2000.0", "1000.0", '"A"', '"B"'))
    assert not {"targets", "predictions", "approved", "passed"} & result.keys()
    assert len(text) < 1200


def test_classification_uses_support_majority_and_equal_task_accuracy():
    data = rows("A", ["yes"]*3, ["yes", "yes", "no"])+rows("B", ["yes"], ["no", "no", "yes"])
    value = measure(data, ["yes"]*3+["no"], "classification")
    assert value["score"] == .75
    assert value["episodes"]["score"]["macro_score"] == .5
    assert value["episodes"]["references"]["support_majority"]["macro_score"] == .5
    assert value["episodes"]["references"]["support_majority"]["matching_prediction_tasks"] == 2


def test_non_episode_scoring_contract_stays_unchanged():
    assert scores("regression", [0., 2.], [1., 3.], [1., 3.]) == {
        "metric": "mae", "direction": "minimize", "score": 1., "rmse": 1.,
        "reference": 1., "reference_method": "training_mean"}
    assert "episodes" not in scores("classification", ["x"], ["x"], ["x", "y"])


@pytest.mark.parametrize("fault", ["length", "identity", "target", "inputs", "empty_support", "changed_support", "boolean", "nonfinite"])
def test_misaligned_or_unfrozen_episode_context_is_rejected(fault):
    data = rows("A", [1., 2.], [1., 3.]);queries = copy.deepcopy(data)
    if fault == "length": queries.pop()
    if fault == "identity": queries[0].pop("task_id")
    if fault == "target": queries[0]["target"] = 10.
    if fault == "inputs": queries[0]["inputs"] = "bad"
    if fault == "empty_support": queries[0]["inputs"]["support"] = []
    if fault == "changed_support": queries[1]["inputs"]["support"][0]["target"] = 100.
    if fault == "boolean": queries[0]["inputs"]["support"][0]["target"] = True
    if fault == "nonfinite": queries[0]["inputs"]["support"][0]["target"] = float("nan")
    with pytest.raises(ValueError):
        scores("regression", [1., 2.], [1., 2.], [1., 3.], query_rows=queries)
