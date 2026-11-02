"""Final-label isolation, fixed-control equivalence and explicit missing outcomes."""

import copy
import math

import numpy as np
import pytest

from agent.rsi import position_baselines, traditional_baselines
from agent.rsi import shared_sequence_evaluation as evaluation


def rows():
    training = [
        {"id": f"train_{i}", "sequence": sequence, "target": i / 3}
        for i, sequence in enumerate(
            ["AAA", "AAC", "ACA", "CAA", "CCC", "CCA", "CAC", "ACC", "AAD", "ADA", "DAA", "DDD"]
        )
    ]
    query = [{"id": "query_0", "sequence": "ACD"}, {"id": "query_1", "sequence": "DCC"}]
    return training, query


def test_controls_match_predeclared_implementations_without_query_labels(monkeypatch):
    training, query = rows()
    controls = evaluation.predict_controls(training, query)
    validation = [{**row, "target": 1000 - i} for i, row in enumerate(query)]
    data = {"training": training, "validation": validation}
    for module in [traditional_baselines, position_baselines]:
        monkeypatch.setattr(module, "load_task", lambda task: copy.deepcopy(data))
    task = {"id": "synthetic", "plugin_id": "generated_sequence.v1"}
    old = traditional_baselines.run_controls(task)["outcomes"] + position_baselines.run_controls(task)["outcomes"]
    assert {row["method"] for row in controls} == {row["method"] for row in old}
    for current, expected in zip(controls, old, strict=True):
        np.testing.assert_allclose(
            [p["prediction"] for p in current["predictions"]],
            [p["prediction"] for p in expected["predictions"]],
            rtol=1e-12,
            atol=1e-12,
        )
    with pytest.raises(ValueError, match="unlabeled"):
        evaluation.predict_controls(training, validation)


def test_variable_length_tasks_keep_unsupported_positional_controls():
    training, query = rows()
    query[0]["sequence"] += "A"
    outcomes = evaluation.predict_controls(training, query)
    assert [row["status"] for row in outcomes] == ["completed"] * 4 + ["unsupported"] * 2
    assert all(row["predictions"] is None for row in outcomes[-2:])


def test_final_metrics_reject_missing_reordered_and_nonfinite_predictions():
    truth = [{"id": "a", "target": 1}, {"id": "b", "target": 3}]
    predicted = [{"id": "a", "prediction": 1}, {"id": "b", "prediction": 1}]
    metrics = evaluation.score_predictions(truth, predicted, 1)
    assert metrics == {"rmse": math.sqrt(2), "mae": 1.0, "spearman": None, "mse_skill_vs_training_mean": 0.0}
    for invalid in [predicted[:1], list(reversed(predicted)), [{"id": "a", "prediction": float("nan")}, predicted[1]]]:
        with pytest.raises(ValueError):
            evaluation.score_predictions(truth, invalid, 1)
    assert evaluation.score_predictions([truth[0]], [predicted[0]], 1)["mse_skill_vs_training_mean"] is None


def test_unacquired_tasks_keep_null_scientist_scores_and_all_controls(monkeypatch):
    training, query = rows()
    data = {"training": training, "validation": [], "future": [{**r, "target": i + 1.0} for i, r in enumerate(query)]}
    monkeypatch.setattr(evaluation, "load_task", lambda task: copy.deepcopy(data))
    task = {
        "id": "synthetic",
        "plugin_id": "generated_sequence.v1",
        "stage": "stability",
        "scope": {},
        "paper": {},
        "data": {},
    }
    first = evaluation.evaluate_task(task, None)
    for row in data["future"]:
        row["target"] += 500
    second = evaluation.evaluate_task(task, None)
    assert first["scientist"] == second["scientist"] == {"status": "not_acquired", "metrics": None, "predictions": None}
    assert first["comparison"] is None
    assert [r["predictions"] for r in first["controls"]] == [r["predictions"] for r in second["controls"]]
    assert first["controls"][0]["metrics"] != second["controls"][0]["metrics"]
    assert first["query_sha256"] == second["query_sha256"]


@pytest.mark.parametrize("change", ["live", "active", "paused", "owner", "inflight", "queued", "empty"])
def test_final_scoring_rejects_nonterminal_campaigns(change):
    state = {"status": "needs_tasks", "owner": None, "inflight": None, "batches": {"b": {"status": "completed"}}}
    evaluation.require_terminal(state, runner_live=False)
    live = change == "live"
    if change in {"active", "paused"}:
        state["status"] = change
    if change in {"owner", "inflight"}:
        state[change] = {"pid": 1}
    if change == "queued":
        state["batches"]["b"]["status"] = "queued"
    if change == "empty":
        state["batches"] = {}
    with pytest.raises(ValueError, match="every acquisition batch terminal"):
        evaluation.require_terminal(state, runner_live=live)
