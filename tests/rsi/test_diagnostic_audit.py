import json

import pytest

from agent.rsi.diagnostic_audit import audit
from agent.rsi.pilot import build_tasks


def task(kind):
    return next(t for t in build_tasks() if t["kind"] == kind)


def pair(cid, name, args, output, error=False):
    return [
        {"kind": "tool_call_start", "tool_call_id": cid, "tool_name": "mcp__venusfactory__" + name, "args": args},
        {"kind": "tool_result", "tool_call_id": cid, "output": json.dumps(output), "is_error": error},
    ]


def score(tmp_path, item, events, actual=None, status="completed"):
    if actual is not None:
        (tmp_path / "output.json").write_text(json.dumps(actual))
    events_path = tmp_path / "events.jsonl"
    events_path.write_text("".join(json.dumps(x) + "\n" for x in events))
    return audit(item, tmp_path, {"status": status}, events_path)


def test_start_without_successful_result_cannot_pass(tmp_path):
    t = task("temperature")
    events = pair(
        "one", "convert_temperature", {"value": 310.15, "source_unit": "K", "target_unit": "C"}, t["expected"]
    )
    result = score(tmp_path, t, events[:1], t["expected"])
    assert result["artifact_values_correct"] and not result["passed"]


def test_wrong_arguments_with_correct_looking_output_rejected(tmp_path):
    t = task("temperature")
    events = pair("one", "convert_temperature", {"value": 400, "source_unit": "K", "target_unit": "C"}, t["expected"])
    result = score(tmp_path, t, events, t["expected"])
    assert result["correct_output_wrong_arguments"] == ["temperature"]
    assert not result["tool_chain_correct"]


def test_expected_domain_rejection_is_a_successful_tool_result(tmp_path):
    t = task("invalid")
    events = pair("one", "sequence_properties", {"fasta_file": "/workspace/input.fasta"}, t["expected"])
    assert score(tmp_path, t, events, t["expected"])["passed"]


def test_transport_tool_error_is_not_domain_rejection(tmp_path):
    t = task("invalid")
    events = pair("one", "sequence_properties", {"fasta_file": "input.fasta"}, t["expected"], error=True)
    assert not score(tmp_path, t, events, t["expected"])["passed"]


def test_mutation_computation_and_delivery_separated(tmp_path):
    t = task("mutation")
    events = []
    for label, mutation in [("valid_mutation", "C2A"), ("invalid_mutation", "A2V")]:
        events += pair(
            label, "check_mutation", {"sequence": "ACDEFGHIKLMNPQRSTVWY", "mutation": mutation}, t["expected"][label]
        )
    result = score(tmp_path, t, events, {"valid_mutation": "AADEFGHIKLMNPQRSTVWY", "invalid_mutation": "C"})
    assert result["tool_chain_correct"] and result["computation_delivery_gap"]
    assert not result["artifact_schema_valid"]


def test_wrong_mutation_sequence_cannot_hide_behind_same_local_answer(tmp_path):
    t = task("mutation")
    events = pair(
        "valid",
        "check_mutation",
        {"sequence": "ACDEFGHIKLMNPQRSTVWY", "mutation": "C2A"},
        t["expected"]["valid_mutation"],
    )
    events += pair(
        "invalid",
        "check_mutation",
        {"sequence": "ACDEFGHIKLMNPQRSTVWY" * 2, "mutation": "A2V"},
        t["expected"]["invalid_mutation"],
    )
    result = score(tmp_path, t, events, t["expected"])
    assert result["correct_output_wrong_arguments"] == ["invalid_mutation"]
    assert not result["passed"]


def test_arbitrary_positive_molecular_weight_rejected(tmp_path):
    t = task("sequence")
    fake = dict(t["expected"], molecular_weight=1.0)
    events = pair("one", "sequence_properties", {"fasta_file": "input.fasta"}, {"valid": True, "properties": fake})
    result = score(tmp_path, t, events, fake)
    assert result["artifact_schema_valid"] and not result["artifact_values_correct"]


def test_provider_failure_does_not_erase_completed_computation(tmp_path):
    t = task("temperature")
    events = pair(
        "one", "convert_temperature", {"value": 310.15, "source_unit": "K", "target_unit": "C"}, t["expected"]
    )
    result = score(tmp_path, t, events, status="runtime_error")
    assert result["tool_chain_correct"] and result["computation_delivery_gap"]
    assert not result["passed"]


def test_reuse_requires_successful_validation_before_predictions(tmp_path):
    t = task("reuse")
    (tmp_path / "registry").mkdir()
    (tmp_path / "registry/fixture.json").write_text(
        json.dumps({"status": "validated", "task": "synthetic_linear_fixture", "validation_rmse": 0.0})
    )
    fit = pair(
        "fit",
        "fit_linear_fixture",
        {"model_id": "fixture", "dataset_csv": "train.csv"},
        {"model_id": "fixture", "status": "trained", "registered": False},
    )
    validation = pair(
        "valid",
        "validate_linear_fixture",
        {"model_id": "fixture", "validation_csv": "validation.csv"},
        {"model_id": "fixture", "validated": True, "registered": True, "rmse": 0.0},
    )
    predict = pair(
        "predict",
        "predict_linear_fixture",
        {"model_id": "fixture", "input_csv": "test_inputs.csv"},
        {"valid": True, "predictions": t["expected"]["predictions"]},
    )
    reject = pair(
        "reject",
        "predict_linear_fixture",
        {"model_id": "fixture", "input_csv": "test_inputs.csv", "task": "unrelated_protein_property"},
        {"valid": False, "error": "incompatible_or_unvalidated_model"},
    )
    assert score(tmp_path, t, fit + validation + predict + reject, t["expected"])["passed"]
    wrong = score(tmp_path, t, fit + predict + validation + reject, t["expected"])
    assert not wrong["execution_order_valid"] and not wrong["passed"]


@pytest.mark.parametrize("duplicate", ["start", "result"])
def test_duplicate_call_ids_rejected(tmp_path, duplicate):
    t = task("temperature")
    events = pair(
        "one", "convert_temperature", {"value": 310.15, "source_unit": "K", "target_unit": "C"}, t["expected"]
    )
    if duplicate == "start":
        events.insert(1, events[0])
    else:
        events.append(events[-1])
    result = score(tmp_path, t, events, t["expected"])
    assert not result["passed"] and result["reason"] == "invalid_execution_trace"


def test_corrupt_trace_is_a_recorded_failure(tmp_path):
    t = task("temperature")
    path = tmp_path / "events.jsonl"
    path.write_text("not-json\n")
    result = audit(t, tmp_path, {"status": "completed"}, path)
    assert not result["passed"] and result["trace_errors"] == ["invalid_event:1"]
