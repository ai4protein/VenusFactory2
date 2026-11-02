import copy
from pathlib import Path

import pytest
from test_generated_assay import make_task

from agent.rsi import generated_assay
from agent.rsi import shared_assay_evaluation as evaluation
from agent.rsi.engine.storage import atomic_json, sha256_file


def predictions(rows):
    return [
        {
            "id": r["id"],
            "protein": r["protein"],
            "group": r["group"],
            "inputs_sha256": generated_assay.input_digest(r),
            "prediction": r["target"],
        }
        for r in rows
    ]


def test_missing_model_preserves_allocation_and_controls(tmp_path):
    task, _ = make_task(tmp_path / "data")
    report = evaluation.evaluate_task(task, None)
    assert report["scientist"]["status"] == "not_registered"
    assert report["comparison"] is None
    assert evaluation.summarize([report])["not_registered"] == 1
    assert evaluation.summarize([report])["acquisition_rate"] == 0
    assert len(report["controls"]) == 4


@pytest.mark.parametrize("field", ["id", "protein", "group", "inputs_sha256"])
def test_predictions_bind_complete_assay_identity(tmp_path, field):
    task, parts = make_task(tmp_path / "data")
    values = predictions(parts["future"])
    values[0][field] = "tampered"
    with pytest.raises(ValueError, match="identities"):
        evaluation.measure(parts["training"], parts["future"], values, task["scope"])


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, "1"])
def test_invalid_predictions_are_not_scored(tmp_path, value):
    task, parts = make_task(tmp_path / "data")
    values = predictions(parts["future"])
    values[0]["prediction"] = value
    with pytest.raises(ValueError, match="Finite"):
        evaluation.measure(parts["training"], parts["future"], values, task["scope"])


def test_equal_group_weight_and_label_independent_controls(tmp_path):
    task, parts = make_task(tmp_path / "data")
    scoring = [parts["future"][0], *parts["future"][8:16]]
    values = predictions(scoring)
    values[0]["prediction"] += 3
    result = evaluation.measure(parts["training"], scoring, values, task["scope"])
    assert result["mse"] == pytest.approx(1)
    assert result["group_balanced_mse"] == pytest.approx(4.5)
    first = evaluation.evaluate_task(task, predictions(parts["future"]))
    for row in parts["future"]:
        row["target"] += 100
    file = Path(task["data"]["future"]["path"])
    atomic_json(file, parts["future"])
    task["data"]["future"]["sha256"] = sha256_file(file)
    second = evaluation.evaluate_task(task, None)
    assert [c["predictions"] for c in first["controls"]] == [c["predictions"] for c in second["controls"]]


def test_protocol_rejects_mutated_source_task_policy_and_implementation(tmp_path):
    task, _ = make_task(tmp_path / "data")
    implementation = tmp_path / "evaluator.py"
    implementation.write_text("# synthetic evaluator witness\n")
    protocol = evaluation.freeze_protocol([task], {str(implementation): sha256_file(implementation)})
    assert protocol["final_labels_scored"] is False
    evaluation.verify_protocol(protocol, [task])
    changed = copy.deepcopy(task)
    changed["minimum_skill"] = 0.5
    with pytest.raises(ValueError, match="allocation"):
        evaluation.verify_protocol(protocol, [changed])
    changed_protocol = copy.deepcopy(protocol)
    changed_protocol["metrics"]["primary_quality"] = "rmse"
    with pytest.raises(ValueError, match="policy"):
        evaluation.verify_protocol(changed_protocol, [task])
    implementation.write_text("# changed\n")
    with pytest.raises(ValueError, match="changed"):
        evaluation.verify_protocol(protocol, [task])


def test_wrong_interface_rejected_before_loading():
    with pytest.raises(ValueError, match="Assay"):
        evaluation.evaluate_task({"plugin_id": "generated_context.v1"}, None)
