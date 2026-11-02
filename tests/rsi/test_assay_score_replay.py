import copy

import pytest
from test_generated_assay import make_task
from test_shared_assay_evaluation import predictions

from agent.rsi import assay_score_replay, shared_assay_evaluation
from agent.rsi.protein_model import digest


def test_independent_replay_covers_scientist_and_controls(tmp_path):
    task, parts = make_task(tmp_path / "data")
    assert assay_score_replay.canonical(task) == digest(task)
    assert assay_score_replay.canonical({"condition": "温度 °C"}) == digest({"condition": "温度 °C"})
    report = shared_assay_evaluation.evaluate_task(task, predictions(parts["future"]))
    replay = assay_score_replay.replay_task(task, report)
    assert len(replay["control_methods"]) == 4
    assert replay["maximum_absolute_error"] < 1e-10
    assert replay["baseline_refit_performed"] is False


@pytest.mark.parametrize("field", ["group_balanced_rmse", "spearman", "rmse"])
def test_replay_detects_edited_scores(tmp_path, field):
    task, parts = make_task(tmp_path / "data")
    report = shared_assay_evaluation.evaluate_task(task, predictions(parts["future"]))
    report["scientist"]["metrics"][field] += 0.1
    with pytest.raises(ValueError, match="replay differs"):
        assay_score_replay.replay_task(task, report)


def test_replay_detects_prediction_change_and_missing_model(tmp_path):
    task, parts = make_task(tmp_path / "data")
    report = shared_assay_evaluation.evaluate_task(task, predictions(parts["future"]))
    changed = copy.deepcopy(report)
    changed["controls"][1]["predictions"][0]["prediction"] += 1
    with pytest.raises(ValueError):
        assay_score_replay.replay_task(task, changed)
    missing = shared_assay_evaluation.evaluate_task(task, None)
    assert assay_score_replay.replay_task(task, missing)["status"] == "prediction_scores_replayed"
