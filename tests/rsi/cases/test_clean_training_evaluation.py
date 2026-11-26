"""Synthetic fixtures test the evaluator; none constitute biological training evidence."""

import copy
import json

import numpy as np
import pytest
import torch

from agent.rsi import clean_annotation as annotation
from agent.rsi import clean_training_evaluation as e
from agent.rsi import clean_training_inputs as inputs


def state(out_dim):
    generator = torch.Generator().manual_seed(1234)
    result = {}
    for i, (n_in, n_out) in enumerate(((1280, 512), (512, 512), (512, out_dim)), 1):
        result[f"fc{i}.weight"] = torch.randn(n_out, n_in, generator=generator) / np.sqrt(n_in)
        result[f"fc{i}.bias"] = torch.randn(n_out, generator=generator) * 0.1
        if i < 3:
            result[f"ln{i}.weight"] = torch.ones(512)
            result[f"ln{i}.bias"] = torch.zeros(512)
    return result


@pytest.mark.parametrize("out_dim", [128, 256])
def test_independent_forward_matches_torch(out_dim):
    weights = state(out_dim)
    x = torch.randn(11, 1280, generator=torch.Generator().manual_seed(9))
    h = x
    for i in (1, 2, 3):
        h = torch.nn.functional.linear(h, weights[f"fc{i}.weight"], weights[f"fc{i}.bias"])
        if i < 3:
            h = torch.relu(
                torch.nn.functional.layer_norm(h, (512,), weights[f"ln{i}.weight"], weights[f"ln{i}.bias"], 1e-5)
            )
    audit = e.compare(h.numpy(), e.forward(weights, x.numpy(), out_dim))
    assert audit["passed"] and audit["max_absolute_error"] < 1e-5


@pytest.mark.parametrize("alteration", ["extra_key", "shape", "nan", "dtype"])
def test_invalid_weights_rejected(alteration):
    weights = state(128)
    if alteration == "extra_key":
        weights["unknown"] = torch.ones(1)
    elif alteration == "shape":
        weights["fc3.weight"] = torch.ones(127, 512)
    elif alteration == "nan":
        weights["ln1.bias"][0] = float("nan")
    else:
        weights["fc1.weight"] = weights["fc1.weight"].double()
    with pytest.raises(ValueError, match="checkpoint"):
        e.state_arrays(weights, 128)


def test_centers_use_unique_multilabel_memberships_and_first_appearance():
    rows = [{"Entry": "A", "EC number": "B;A;B"}, {"Entry": "B", "EC number": "A"}, {"Entry": "C", "EC number": "B;C"}]
    labels, centers = e.class_centers([[1.0, 3.0], [3.0, 7.0], [9.0, 11.0]], rows)
    assert labels == ["B", "A", "C"]
    np.testing.assert_array_equal(centers, [[5.0, 7.0], [2.0, 5.0], [9.0, 11.0]])
    rows[2]["Entry"] = "A"
    with pytest.raises(ValueError, match="identities"):
        e.class_centers(np.zeros((3, 2)), rows)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "bool", "nan", "negative", "timestamp", "reverse_time"])
def test_invalid_epoch_evidence_fails(mutation):
    rows = [{"epoch": 1, "loss": 0.8, "at": 100.0}, {"epoch": 2, "loss": 0.4, "at": 101.0}]
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows[1]["epoch"] = 1
    elif mutation == "bool":
        rows[0]["epoch"] = True
    elif mutation == "nan":
        rows[0]["loss"] = float("nan")
    elif mutation == "negative":
        rows[0]["loss"] = -0.1
    elif mutation == "timestamp":
        rows[0]["at"] = True
    else:
        rows[1]["at"] = 99.0
    with pytest.raises(ValueError):
        e.audit_epochs(rows, 2)


def test_sklearn_full_union_agrees_with_independent_counts():
    truth, pred = [["A", "B"], ["A"], ["C"]], [["A", "unknown"], ["A", "B"], ["unknown"]]
    actual, independent = e.sklearn_metrics(truth, pred), annotation.metrics(truth, pred)
    assert e.same_metrics(actual, independent)
    restricted = e.sklearn_metrics(truth, pred, truth_only=True)
    assert restricted["micro_precision"] > actual["micro_precision"]
    assert actual["predicted_labels_absent_from_truth"] == ["unknown"]
    assert not e.same_metrics({**actual, "micro_f1": float("nan")}, independent)
    assert not e.same_metrics({**actual, "rows": True}, independent)


def test_numerical_gate_retains_mismatches_and_rejects_shapes():
    assert e.compare(np.ones((2, 3)), np.ones((2, 3)))["passed"]
    result = e.compare(np.ones((2, 3)), np.zeros((2, 3)))
    assert not result["passed"] and result["outside_tolerance"] == 6
    with pytest.raises(ValueError, match="shape"):
        e.compare(np.ones((3,)), np.ones((2, 3)))
    with pytest.raises(ValueError, match="finiteness"):
        e.compare(np.array([float("nan")]), np.ones(1))


def test_protocol_requires_external_admission_not_self_declared_sources():
    with pytest.raises(ValueError, match="protocol"):
        e.validate_protocol({"implementation": e.PREPARATION})


@pytest.fixture
def trained_fixture(tmp_path, monkeypatch):
    """Fabricated trace/checkpoints solely for evidence-validation tests."""
    root, assets = tmp_path / "run", tmp_path / "assets"
    (root / "work/data/model").mkdir(parents=True)
    (assets / "source/app/data").mkdir(parents=True)
    (assets / "source/app/data/split10.csv").write_text("fixed synthetic training cohort")
    (root / "work/data/split10.csv").write_text("fixed synthetic training cohort")
    script = e.SCHEDULES["triplet"]["script"]
    for path in (root / script, assets / "source/app" / script):
        path.write_text("synthetic author fixture")
    (root / "clean_author_training.py").write_text("synthetic executor fixture")
    monkeypatch.setattr(e, "AUTHOR_SCRIPTS", {**e.AUTHOR_SCRIPTS, "triplet": inputs.sha(root / script)})
    monkeypatch.setattr(e, "TRAINING_EXECUTOR", inputs.sha(root / "clean_author_training.py"))
    initial = state(128)
    torch.save(initial, root / "initial.pth")
    final = copy.deepcopy(initial)
    final["fc3.bias"] += 1
    torch.save(final, root / "final.pth")
    (root / "work/data/model/split10_triplet.pth").write_bytes((root / "final.pth").read_bytes())
    with (root / "epochs.jsonl").open("w") as stream:
        for i in range(1, 2001):
            stream.write(json.dumps({"epoch": i, "loss": 1 / i, "at": 1000.0 + i}) + "\n")
    report = {
        "status": "completed",
        "stage": "trained_checkpoint",
        "algorithm": "triplet",
        "schedule": e.SCHEDULES["triplet"],
        "source_commit": inputs.COMMIT,
        "author_script_sha256": e.AUTHOR_SCRIPTS["triplet"],
        "input_protocol_sha256": e.INPUT_PROTOCOL,
        "training_completed": True,
        "evaluation_completed": False,
        "new_model_developed": False,
        "autonomous_model_developed": False,
        "full_paper_reproduced": False,
        "completed_epochs": 2000,
        "external_model_calls": 0,
        "cache_hits": 100,
        "parameter_values_changed": 128,
        "artifact_sha256": {
            name: inputs.sha(root / name)
            for name in (script, "clean_author_training.py", "initial.pth", "final.pth", "epochs.jsonl")
        },
    }
    inputs.save(root / "report.json", report)
    return root, assets, report


def test_complete_training_admission(trained_fixture):
    root, assets, _ = trained_fixture
    record = e.admit_training(root, assets, "triplet")
    assert record["status"] == "passed" and record["epochs"] == 2000
    assert record["parameter_values_changed"] == 128
    assert record["training_rerun_by_auditor"] is False


@pytest.mark.parametrize(
    "mutation",
    [
        "overclaim",
        "epochs",
        "missing_artifact",
        "extra_artifact",
        "rehashed_executor",
        "author_checkpoint",
        "changed_count",
        "train_cohort",
        "trace_rehashed",
    ],
)
def test_training_evidence_tampering_fails(trained_fixture, mutation):
    root, assets, report = trained_fixture
    if mutation == "overclaim":
        report["autonomous_model_developed"] = True
    elif mutation == "epochs":
        report["completed_epochs"] = 1999
    elif mutation == "missing_artifact":
        report["artifact_sha256"].pop("epochs.jsonl")
    elif mutation == "extra_artifact":
        report["artifact_sha256"]["extra"] = "hash"
    elif mutation == "rehashed_executor":
        (root / "clean_author_training.py").write_text("changed executor")
        report["artifact_sha256"]["clean_author_training.py"] = inputs.sha(root / "clean_author_training.py")
    elif mutation == "author_checkpoint":
        (root / "work/data/model/split10_triplet.pth").write_text("wrong checkpoint")
    elif mutation == "changed_count":
        report["parameter_values_changed"] = 0
    elif mutation == "train_cohort":
        (root / "work/data/split10.csv").write_text("changed cohort")
    else:
        trace = (root / "epochs.jsonl").read_text().splitlines()
        trace[1] = trace[0]
        (root / "epochs.jsonl").write_text("\n".join(trace) + "\n")
        report["artifact_sha256"]["epochs.jsonl"] = inputs.sha(root / "epochs.jsonl")
    inputs.save(root / "report.json", report)
    with pytest.raises(ValueError):
        e.admit_training(root, assets, "triplet")


def test_evaluation_protocol_keeps_scope_and_tolerances_fixed():
    protocol = e.evaluation_protocol({"fixture.py": "hash"})
    assert protocol["allocation"]["prediction_records"] == 1156
    assert protocol["numerical_tolerances"] == {"atol": 5e-5, "rtol": 5e-6, "metric_atol": 1e-12, "exact_labels": True}
    assert not any(protocol["limits"].values())
    assert len(e.artifact_names()) == 37
