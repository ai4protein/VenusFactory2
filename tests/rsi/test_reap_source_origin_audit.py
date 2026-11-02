"""Denied evaluations consume tool calls, but cannot create a third source fit."""

import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/rsi"))
from audit_reap_generated import verify_code_origin


def fixture(work):
    code = "def fit(features, targets, seed): return {}\ndef predict(model, features): return [0.0]*len(features)\n"
    sha = hashlib.sha256(code.encode()).hexdigest()
    model = {"source_sha256": sha, "recipe": {"source_code": code, "seed": 314159}}
    feedback = {"mse_skill_vs_training_mean": 0.1}
    (work / "code_attempt_2.json").write_text(json.dumps({"source_code": code, "training_feedback": feedback}))
    (work / "source_provenance.json").write_text(json.dumps({"origin": "agent_tool_call_source_code",
               "source_sha256": sha, "training_feedback_attempts": 2}))
    events = []
    def pair(cid, name, args, output, error=False):
        events.extend([{"kind": "tool_call_start", "tool_call_id": cid, "tool_name": "mcp__venusfactory__" + name, "args": args},
                       {"kind": "tool_result", "tool_call_id": cid, "is_error": error, "output": output}])
    pair("first", "evaluate_model_code", {"source_code": "first failing code"},
         "ToolError: Error calling tool 'evaluate_model_code': Out of range float values are not JSON compliant: inf", True)
    pair("second", "evaluate_model_code", {"source_code": code}, json.dumps({"source_sha256": sha,
         "training_only_feedback": feedback, "attempts_remaining": 0, "status": "tested_not_published"}))
    pair("denied", "evaluate_model_code", {"source_code": "third proposed code"},
         "ToolError: Error calling tool 'evaluate_model_code': code-development budget exhausted", True)
    pair("submit", "submit_model_code", {"source_sha256": sha}, json.dumps({"source_sha256": sha}))
    return model, events


def test_exact_denial_preserves_failed_first_attempt_and_two_admitted_attempts(tmp_path):
    model, events = fixture(tmp_path)
    result = verify_code_origin(tmp_path, events, model)
    assert result["attempts"] == 2 and result["evaluation_tool_calls"] == 3
    assert result["budget_rejected_calls"] == 1 and result["source_from_agent_tool"]
    assert not (tmp_path / "code_attempt_1.json").exists()


@pytest.mark.parametrize("error", [False, True])
def test_third_executed_or_unexplained_attempt_remains_a_violation(tmp_path, error):
    model, events = fixture(tmp_path)
    result = next(e for e in events if e["kind"] == "tool_result" and e["tool_call_id"] == "denied")
    result.update(is_error=error, output="An unrelated failure" if error else "{}")
    with pytest.raises(ValueError, match="Admitted code-attempt budget"):
        verify_code_origin(tmp_path, events, model)


def test_denial_cannot_hide_a_third_archived_execution(tmp_path):
    model, events = fixture(tmp_path)
    (tmp_path / "code_attempt_3.json").write_text("{}")
    with pytest.raises(ValueError, match="Third attempt artifact"):
        verify_code_origin(tmp_path, events, model)


def test_denial_before_two_attempts_is_not_excused(tmp_path):
    model, events = fixture(tmp_path)
    events = [e for e in events if e["tool_call_id"] != "first"]
    with pytest.raises(ValueError, match="attempt reservation|before the declared"):
        verify_code_origin(tmp_path, events, model)


def test_successful_source_cannot_reset_the_attempt_reservation(tmp_path):
    model, events = fixture(tmp_path)
    events = copy.deepcopy(events)
    event = next(e for e in events if e["kind"] == "tool_result" and e["tool_call_id"] == "second")
    value = json.loads(event["output"])
    value["attempts_remaining"] = 1
    event["output"] = json.dumps(value)
    with pytest.raises(ValueError, match="reservation"):
        verify_code_origin(tmp_path, events, model)
