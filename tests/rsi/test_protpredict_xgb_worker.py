"""Source-matched comparator checks; synthetic examples never count as models."""

import copy
import importlib.util
import json
import os
import subprocess
from pathlib import Path

import pytest

WORKER = Path(__file__).resolve().parents[2] / "scripts/rsi/protpredict_xgb_worker.py"
spec = importlib.util.spec_from_file_location("protpredict_xgb_worker", WORKER)
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


@pytest.fixture(name="payload")
def make_request():
    rows = [[((i + 1) * (j + 3) % 79) / 79 for j in range(19)] for i in range(80)]
    return {
        "schema": worker.SCHEMA,
        "action": "fit_predict",
        "feature_names": [f"aa_{i}" for i in range(19)],
        "training": {
            "x": rows[:60],
            "y": [10 + 3 * x[0] - 2 * x[18] for x in rows[:60]],
            "groups": [f"gene_{i}" for i in range(60)],
        },
        "scoring": {"x": rows[60:], "ids": [f"held_{i}" for i in range(20)]},
        "parameters": {
            "gamma": 0.0,
            "max_depth": 2,
            "subsample": 1.0,
            "learning_rate": 0.1,
            "n_estimators": 10,
            "colsample_bytree": 1.0,
            "reg_lambda": 1.0,
            "reg_alpha": 0.0,
        },
    }


@pytest.mark.parametrize(
    "attack",
    ["outer_labels", "extra_fields", "missing_feature", "duplicate_group", "nan", "negative", "parameter", "boolean"],
)
def test_malformed_inputs_and_outer_labels_are_rejected(payload, attack):
    request = payload
    if attack == "outer_labels":
        request["scoring"]["y"] = [0.0] * 20
    elif attack == "extra_fields":
        request["future_labels"] = []
    elif attack == "missing_feature":
        request["training"]["x"][0].pop()
    elif attack == "duplicate_group":
        request["training"]["groups"][1] = request["training"]["groups"][0]
    elif attack in {"nan", "negative", "boolean"}:
        request["training"]["x"][0][0] = {"nan": float("nan"), "negative": -1, "boolean": True}[attack]
    else:
        request["parameters"]["n_estimators"] = 1001
    with pytest.raises(ValueError):
        worker.validate_request(request)


def run(request, root):
    runtime = os.environ.get("VENUS_PROTPREDICT_PYTHON")
    if not runtime:
        pytest.skip("Explicit isolated comparator interpreter required")
    root.mkdir()
    source = root / "request.json"
    source.write_text(json.dumps(request))
    cp = subprocess.run(
        [runtime, str(WORKER), "--request", str(source), "--output", str(root / "result")],
        text=True,
        capture_output=True,
        timeout=90,
        env=dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", PYTHONNOUSERSITE="1"),
    )
    assert cp.returncode == 0, cp.stdout + cp.stderr
    return root / "result", json.loads((root / "result/report.json").read_text())


def test_actual_versioned_training_is_deterministic_and_ignores_scoring_for_fit(payload, tmp_path):
    request = payload
    first, a = run(request, tmp_path / "first")
    request["scoring"]["x"] = [[value + 0.2 for value in row] for row in request["scoring"]["x"]]
    second, b = run(request, tmp_path / "second")
    assert a["runtime"]["packages"] == worker.VERSIONS
    assert a["model"] == b["model"] and a["selected_feature_indices"] == list(range(19))
    assert (first / "model.json").read_bytes() == (second / "model.json").read_bytes()
    assert a["outer_labels_received"] is False and a["new_autonomous_models"] == 0
    assert a["scoring_rows"] == 20


def test_duplicate_columns_use_training_only(payload, tmp_path):
    request = payload
    for row in request["training"]["x"]:
        row[18] = row[0]
    # Scoring rows deliberately break the training duplication.
    _, report = run(request, tmp_path / "duplicate")
    assert report["selected_feature_indices"] == list(range(18))


def test_actual_bayesian_search_produces_bound_parameters_and_trace(payload, tmp_path):
    request = payload
    request = {key: copy.deepcopy(value) for key, value in request.items() if key not in {"parameters", "scoring"}}
    request.update(action="tune", search={"iterations": 2, "folds": 2})
    folder, report = run(request, tmp_path / "tune")
    assert report["status"] == "completed" and report["search"] == request["search"]
    assert set(json.loads((folder / "parameters.json").read_text())) == set(worker.BOUNDS)
    trace = json.loads((folder / "search_trace.json").read_text())
    assert len(trace) == 2 and all(len(x["fold_training_mse"]) == 2 for x in trace)
    assert json.loads((folder / "progress.json").read_text())["iterations_completed"] == 2
