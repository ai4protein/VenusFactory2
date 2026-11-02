"""Domain references must not fit withheld labels or weight source repeats as groups."""

import copy
import json

import numpy as np
import pytest
from test_sabio_assay_tasks import synthetic_sources

from agent.rsi import model_tasks, sabio_assay_tasks
from agent.rsi import sabio_domain_controls as domain
from agent.rsi.engine.assay_baselines import run_controls


def test_family_median_weights_components_and_uses_hierarchy():
    train = [{"id": str(i), "group": "one" if i < 10 else "two", "target": 0.0 if i < 10 else 10.0} for i in range(11)]
    metadata = {r["id"]: {"ec": "1.1.1.1"} for r in train}
    metadata["score"] = {"ec": "1.1.1.2"}
    value, basis, groups = domain.family_reference(train, {"id": "score"}, metadata)
    assert value == 5 and basis == "ec_prefix_3" and groups == 2


@pytest.mark.parametrize(
    "first,second,expected",
    [([5, 7, 0, 1], [6, 8, 0, 1], 0), ([5, 6, 0, 1], [8, 8, 0, 0], 2 / 14), ([0, 0, 1, 0], [8, 8, 0, 0], 1)],
)
def test_condition_distance_preserves_intervals_and_missingness(first, second, expected):
    assert domain.interval_distance(first, second, 14.0) == expected


def test_domain_predictions_and_six_control_dispatch_are_train_only(tmp_path):
    source, sequences, _, _ = synthetic_sources(tmp_path)
    report = sabio_assay_tasks.prepare(source, [sequences], [], tmp_path / "tasks")
    task = json.loads(model_tasks.bound_file(report["tasks"][0]).read_text())
    data = model_tasks.load_task(task)
    metadata = domain.source_metadata(task, data)
    first = domain.score_rows(data["training"], data["validation"], task["scope"], metadata)
    altered = copy.deepcopy(data["validation"])
    for row in altered:
        row["target"] += 999
    second = domain.score_rows(data["training"], altered, task["scope"], metadata)
    assert [r["predictions"] for r in first] == [r["predictions"] for r in second]
    assert first[0]["metrics"] != second[0]["metrics"]
    controls = run_controls(task)
    assert len(controls["outcomes"]) == 6 and controls["declared_domain_controls_completed"] == 2
    assert not controls["future_scored"] and not controls["domain_baseline_comparison_complete"]
    assert all(np.isfinite(r["metrics"]["rmse"]) for r in controls["outcomes"])
    metadata[data["training"][0]["id"]]["target"] = 0
    with pytest.raises(ValueError, match="inputs only"):
        domain.score_rows(data["training"], data["validation"], task["scope"], metadata)
