"""Known traditional controls exercise real multilabel training; synthetic data only."""

import hashlib
import json

import numpy as np
import pytest

from agent.harness.research_reference_baselines import reference_baseline
from agent.harness.research_workspace import ResearchWorkspace
from agent.rsi.engine.multilabel_baselines import sequence_features


def portable(method, column="sequence", output="value"):
    item = reference_baseline(method, column, output, task="multilabel")
    assert hashlib.sha256(item["source"].encode()).hexdigest() == item["source_sha256"]
    module = {}
    exec(compile(item["source"], "multilabel_reference", "exec"), module)
    return item, module


def observations(start, stop, split):
    return [
        {
            "id": f"{split}-{i}",
            "group": f"{split}-{i}",
            "inputs": {"sequence": "A" * i + "K" * (40 - i)},
            "target": {"A-rich": int(i > 20), "K-rich": int(i < 20), "Unknown": None if i % 3 else 1},
        }
        for i in range(start, stop)
    ]


def test_multilabel_overview_is_compact_and_task_specific():
    result = reference_baseline(task="multilabel")
    assert len(result["references"]) == 3 and len(json.dumps(result)) < 1500
    assert "source" not in result
    with pytest.raises(ValueError, match="Reference task"):
        reference_baseline(task="classification")
    with pytest.raises(ValueError, match="multilabel"):
        reference_baseline("aac_ridge", task="multilabel")


@pytest.mark.parametrize("method", ["training_prevalence", "aac_logistic", "terminal_logistic"])
def test_reference_masks_unknown_targets_and_preserves_multilabel_probabilities(method):
    _, module = portable(method)
    train = observations(5, 36, "train")
    fitted = module["fit"]({"task": "multilabel", "labels": ["A-rich", "K-rich", "Unknown"], "train": train}, 0)
    before = module["predict"](fitted, {"sequence": "AAAKKKX"})["value"]
    assert set(before) == {"A-rich", "K-rich", "Unknown"}
    assert before["Unknown"] == 1
    assert sum(before.values()) > 1
    assert all(0 <= value <= 1 for value in before.values())
    if method == "training_prevalence":
        assert before["A-rich"] == sum(row["target"]["A-rich"] for row in train) / len(train)
    else:
        assert (
            module["predict"](fitted, {"sequence": "A" * 35 + "K" * 5})["value"]["A-rich"]
            > module["predict"](fitted, {"sequence": "A" * 5 + "K" * 35})["value"]["A-rich"]
        )
    # Unknown labels cannot become negatives, and held-out keys cannot affect fitting.
    changed = {
        "task": "multilabel",
        "labels": ["A-rich", "K-rich", "Unknown"],
        "train": train,
        "test": [{"target": "trap"}],
    }
    second = module["fit"](changed, 0)
    assert module["predict"](second, {"sequence": "AAAKKKX"})["value"] == before
    train[0]["target"]["A-rich"] = 2
    with pytest.raises(ValueError, match="binary"):
        module["fit"]({"task": "multilabel", "labels": ["A-rich"], "train": train}, 0)


def test_features_preserve_terminal_order_unknown_fraction_and_full_length():
    a = "K" * 50 + "A" * 100 + "D" * 50
    b = "D" * 50 + "A" * 100 + "K" * 50
    np.testing.assert_array_equal(sequence_features(a), sequence_features(b))
    assert sequence_features(a, True) != sequence_features(b, True)
    x = sequence_features("AXBZ")
    assert len(x) == 22 and x[-2] == 0.75 and x[0] == 0.25
    assert len(sequence_features("A", True)) == 64
    for sequence in ("", "aa", "A*", "A C", None):
        with pytest.raises(ValueError, match="amino-acid"):
            sequence_features(sequence)


def test_source_fields_remain_literal_and_unobserved_label_is_rejected():
    _, module = portable("training_prevalence", "__OUTPUT__", "score")
    training = {
        "task": "multilabel",
        "labels": ["x", "y"],
        "train": [
            {"inputs": {"__OUTPUT__": "AA"}, "target": {"x": 1, "y": None}},
            {"inputs": {"__OUTPUT__": "KK"}, "target": {"x": 0, "y": None}},
        ],
    }
    with pytest.raises(ValueError, match="observed"):
        module["fit"](training, 0)
    training["train"][0]["target"]["y"] = 1
    fitted = module["fit"](training, 0)
    assert module["predict"](fitted, {"__OUTPUT__": "AC"}) == {"score": {"x": 0.5, "y": 1.0}}


@pytest.mark.asyncio
async def test_fixed_reference_fits_and_replays_in_real_isolated_model_api(tmp_path):
    service = ResearchWorkspace(tmp_path / "work", tmp_path / "control", {"steps": []})
    item = await service.get_research_baseline("terminal_logistic", output_file="src/reference.py", task="multilabel")
    assert "source" not in item and item["kind"] == "platform_reference"
    data = {
        "task": "multilabel",
        "labels": ["A-rich", "K-rich", "Unknown"],
        "provenance": "Synthetic software fixture, not a biological experiment",
        "grouping": "Distinct synthetic inputs",
        "splits": {
            "train": observations(5, 36, "train"),
            "validation": observations(1, 4, "validation"),
            "test": observations(36, 39, "test"),
        },
    }
    await service.write_research_file("evaluation.json", json.dumps(data))
    result = await service.train_research_model(
        "Fixed terminal composition control",
        "src/reference.py",
        "evaluation.json",
        "Synthetic reference check",
        variant="baseline",
        seed=0,
    )
    assert result["experiment"]["fresh_process_replay"]
    assert not result["training"]["validation_and_test_mounted"]
    assert result["training"]["training_rows"] == 31
    assert not result["experiment"]["test_used_for_feedback"]
    assert result["experiment"]["validation"]["metric"] == "macro_brier"
    assert not service.proposals
