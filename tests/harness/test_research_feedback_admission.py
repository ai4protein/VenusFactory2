"""Request rejections remain independent from executed training failures."""
import copy
import hashlib
import json

import pytest

from agent.harness.research_execution_feedback import failure_summary, feedback_state, repair_source_context

ATTEMPT = "a" * 32
CAUSE = "ValueError: Invalid file path or buffer object type: <class 'dict'>"
BUDGET = "DisclosureError: Execution budget reached. Save a research checkpoint or finish this iteration."


def call(message, *, identity="call", source="src/model.py", error=True, wrapped=False, options=None):
    arguments = {"name": "fixture", "source_file": source, "evaluation_file": "evaluation.json",
                 "hypothesis": "private_input never disclose", **(options or {})}
    return [
        {"kind": "tool_call_start", "tool_call_id": identity,
         "tool_name": "mcp__venusfactory__call_tool" if wrapped else "mcp__venusfactory__train_research_model",
         "args": {"name": "train_research_model", "arguments": arguments} if wrapped else arguments},
        {"kind": "tool_result", "tool_call_id": identity, "is_error": error, "output": message},
    ]


def failed(identity=ATTEMPT, **kwargs):
    return call(f"TrainingAttemptError: Training attempt {identity} failed: {CAUSE}", **kwargs)


@pytest.mark.parametrize("message", [BUDGET, "PermissionError: Tool disabled", "ValueError: Invalid arguments"])
@pytest.mark.parametrize("wrapped", [False, True])
def test_request_failure_preserves_actual_attempt_and_remains_visible(tmp_path, message, wrapped):
    state = feedback_state(failed(wrapped=wrapped) + call(message, identity="denied", wrapped=wrapped), tmp_path)
    actual, request = state["pending"]
    assert actual["training_attempt_id"] == ATTEMPT and CAUSE in actual["error"]
    assert request["failure_kind"] == "request" and request["error"] == message
    assert "training_attempt_id" not in request
    assert [r["error"] for r in state["errors"]] == [actual["error"], message]
    assert "private_input" not in json.dumps(state)
    snapshot = copy.deepcopy(state)
    assert feedback_state([], tmp_path, state) == state
    assert state == snapshot


def test_request_failure_without_attempt_does_not_invent_execution(tmp_path):
    state = feedback_state(call(BUDGET), tmp_path)
    row, = state["pending"]
    assert row["error"] == BUDGET and row["failure_kind"] == "request"
    assert "training_attempt_id" not in row and len(state["errors"]) == 1
    assert repair_source_context(tmp_path, state) is None


def test_repeated_request_is_bounded_without_evicting_real_execution(tmp_path):
    events = failed()
    for index in range(20):
        events += call(f"Denied {index} " + "x" * 3000, identity=str(index))
    state = feedback_state(events, tmp_path)
    assert len(state["pending"]) == 2
    assert state["errors"][1]["error"].startswith("Denied 19 ")
    assert all(len(row["error"]) <= 1200 for row in state["errors"])
    for index in range(20):
        events += call(BUDGET, identity=str(index), options={"evaluation_file": f"other{index}.json"})
    state = feedback_state(events, tmp_path)
    assert len(state["pending"]) == 6
    assert state["pending"][0]["training_attempt_id"] == ATTEMPT


def test_new_actual_attempt_resolves_matching_request_but_keeps_latest_fit_error(tmp_path):
    old = feedback_state(failed() + call(BUDGET, identity="denied"), tmp_path)
    current = feedback_state(failed("b" * 32), tmp_path, old)
    row, = current["pending"]
    assert row["training_attempt_id"] == "b" * 32
    assert "failure_kind" not in row and len(current["errors"]) == 1
    assert len(old["pending"]) == 2


@pytest.mark.parametrize("changed", [
    {"evaluation_file": "other.json"}, {"seed": 1}, {"prediction_key": "other"},
    {"device": "cuda"}, {"timeout_seconds": 90}, {"parent_experiment_id": "b" * 32},
])
def test_different_success_does_not_clear_rejected_request(tmp_path, changed):
    state = feedback_state(failed() + call(BUDGET, identity="denied", options=changed), tmp_path)
    current = feedback_state(call("completed", error=False), tmp_path, state)
    request, = current["pending"]
    assert request["failure_kind"] == "request" and request["error"] == BUDGET
    assert "training_attempt_id" not in request
    assert feedback_state(call("completed", error=False, options=changed), tmp_path, current)["errors"] == []


@pytest.mark.parametrize("variant", ["auto", "baseline"])
def test_defaults_and_direct_wrapped_names_have_same_request_identity(tmp_path, variant):
    state = feedback_state(call(BUDGET, wrapped=True), tmp_path)
    actual = call("completed", error=False, options={"strategy": "supervised", "parent_experiment_id": "",
        "variant": variant, "prediction_key": "value", "seed": 0, "device": "cpu", "timeout_seconds": 120})
    assert feedback_state(actual, tmp_path, state) == {"pending": [], "errors": []}


def test_success_clears_both_failures_only_for_matching_request(tmp_path):
    state = feedback_state(failed() + call(BUDGET, identity="denied"), tmp_path)
    assert feedback_state(call("completed", source="src/other.py", error=False), tmp_path, state) == state
    assert feedback_state(call("completed", error=False), tmp_path, state) == {"pending": [], "errors": []}


def test_legacy_attached_rejection_has_no_invented_request_scope(tmp_path):
    state = feedback_state(failed(), tmp_path)
    state["pending"][0]["request_failure"] = {"tool": state["pending"][0]["tool"], "error": BUDGET}
    migrated = feedback_state([], tmp_path, state)
    assert len(migrated["pending"]) == 2
    request = migrated["pending"][1]
    assert request["key"].startswith("legacy-request:") and "training_attempt_id" not in request
    current = feedback_state(call("completed", error=False), tmp_path, migrated)
    assert current["pending"] == [request]
    assert feedback_state([], tmp_path, current) == current


def test_repair_context_prioritizes_execution_over_newer_rejections(tmp_path):
    (tmp_path / "src").mkdir()
    source = b"def fit(training, seed):\n    return dict(training)\ndef predict(model, inputs):\n    return {'value': 1}\n"
    (tmp_path / "src/model.py").write_bytes(source)
    state = feedback_state(failed() + call(BUDGET, identity="denied"), tmp_path)
    repair = repair_source_context(tmp_path, state)
    assert repair["sha256"] == hashlib.sha256(source).hexdigest()
    assert CAUSE in repair["error"] and BUDGET not in repair["error"]
    assert repair["request_failures"][0]["error"] == BUDGET


@pytest.mark.parametrize("prefix", ["Execution exited with code 1: ", "Execution exited with code -15: "])
def test_controlled_runner_traceback_preserves_actual_exception(prefix):
    text = prefix + "Traceback (most recent call last):\n  File 'train.py', line 2\n    ~~~~^^^^\nRuntimeError: training failed"
    assert failure_summary(text) == "RuntimeError: training failed"


@pytest.mark.parametrize("first", [
    "ValueError: Execution exited with code 1: Traceback (most recent call last):",
    "Execution exited with code 01: Traceback (most recent call last):",
    "Execution failed: Traceback (most recent call last):",
    "Model source check failed: line 3, column 2: missing function",
])
def test_other_diagnostics_do_not_promote_later_text(first):
    assert failure_summary(first + "\n    source excerpt\nDo not replace the typed headline") == first
