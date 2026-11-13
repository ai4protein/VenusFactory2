"""Private role evidence is complete within its limit; repair prompts stay bounded."""
import hashlib
import json
from types import SimpleNamespace

import pytest

from agent.harness.autonomous_research import AutonomousResearch
from agent.harness.research_plan_format import remove_empty_step_inputs
from agent.harness.research_role_outputs import repair_feedback
from agent.harness.scientist_role_output import (
    RoleOutputError,
    output_content,
    specification,
    validate_output,
)


def plan():
    return {"summary": "Develop a model", "deliverable": "model", "needs_input": [],
            "steps": [{"id": "fit", "title": "Fit", "acceptance": "Measure held-out error",
                       "approach": "Compare a reference", "verification": "execution", "needs_input": []}]}


def envelope(raw, spec, channel):
    if spec["protocol"] == "chat_completions":
        message = {"role": "assistant", "content": raw}
        finish = "stop"
        if channel == "function":
            message = {"role": "assistant", "content": None, "tool_calls": [{"type": "function",
                       "function": {"name": spec["name"], "arguments": raw}}]}
            finish = "tool_calls"
        return {"model": "fixture", "choices": [{"message": message, "finish_reason": finish}]}
    if channel == "function":
        output = [{"type": "function_call", "status": "completed", "name": spec["name"], "arguments": raw}]
    else:
        output = [{"type": "message", "role": "assistant", "status": "completed",
                   "content": [{"type": "output_text", "text": raw}]}]
    return {"model": "fixture", "status": "completed", "output": output}


@pytest.mark.parametrize("protocol", ["chat_completions", "responses"])
@pytest.mark.parametrize("channel", ["text", "function"])
def test_invalid_content_after_12k_cannot_be_removed_by_normalization(protocol, channel):
    raw = json.dumps(plan()) + " " * 12500 + "untrusted trailing content"
    spec = specification("CB", protocol)
    with pytest.raises(RoleOutputError) as failure:
        output_content(envelope(raw, spec, channel), spec, "fixture")
    assert failure.value.raw == raw
    assert remove_empty_step_inputs(failure.value.raw, protocol) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", ["chat_completions", "responses"])
async def test_normalized_plan_saves_the_complete_original_and_tail_whitespace(tmp_path, protocol):
    raw = json.dumps(plan()) + " " * 12500
    spec = specification("CB", protocol)
    runner = object.__new__(AutonomousResearch)
    calls = []
    async def consult(*args):
        calls.append(args)
        return output_content(envelope(raw, spec, "function"), spec, "fixture")
    runner.consult = consult
    runner.activity = lambda *args: None
    profile = SimpleNamespace(model=SimpleNamespace(protocol=protocol), budget=SimpleNamespace(model_calls=16))
    result = await runner.structured("CB", {}, profile, SimpleNamespace(calls=1), tmp_path, json.loads)
    assert len(calls) == 1 and "needs_input" not in result["steps"][0]
    saved = next((tmp_path / "role_outputs").glob("*.txt"))
    receipt = json.loads(saved.with_suffix(".json").read_text())
    assert saved.read_bytes() == raw.encode()
    assert receipt["response_sha256"] == hashlib.sha256(raw.encode()).hexdigest()
    assert receipt["saved_sha256"] == receipt["response_sha256"]
    assert receipt["response_characters"] == len(raw) > 12000
    assert receipt["truncated"] is False
    record = json.loads(next((tmp_path / "role_normalizations").glob("*.json")).read_text())
    assert record["original_response"] == receipt


@pytest.mark.parametrize("protocol", ["chat_completions", "responses"])
@pytest.mark.parametrize("channel", ["text", "function"])
def test_wire_outputs_above_128k_are_not_repairable(protocol, channel):
    raw = json.dumps(plan()) + " " * 128000
    spec = specification("CB", protocol)
    with pytest.raises(ValueError) as failure:
        output_content(envelope(raw, spec, channel), spec, "fixture")
    assert not isinstance(failure.value, RoleOutputError)


def test_error_and_validation_entry_reject_oversize_without_truncation():
    raw = "x" * 128001
    for operation in (lambda: RoleOutputError("Invalid response", raw),
                      lambda: validate_output(raw, specification("CB", "chat_completions"))):
        with pytest.raises(ValueError, match="exceeds 128000") as failure:
            operation()
        assert not isinstance(failure.value, RoleOutputError)
    boundary = "x" * 128000
    assert RoleOutputError("Invalid response", boundary).raw == boundary


@pytest.mark.parametrize("length,complete", [(32000, True), (32001, False), (128000, False)])
def test_saved_full_response_does_not_expand_repair_prompt_budget(length, complete):
    raw = "x" * length
    failure = RoleOutputError("Invalid response", raw)
    feedback = repair_feedback(failure.raw, str(failure), "Repair only the format.")
    assert failure.raw == raw
    assert feedback["previous_response_complete"] is complete
    assert feedback["previous_response_sha256"] == hashlib.sha256(raw.encode()).hexdigest()
    if complete:
        assert feedback["previous_response"] == raw and "previous_response_excerpt" not in feedback
    else:
        assert "previous_response" not in feedback
        assert len(feedback["previous_response_excerpt"]) == 12000
        assert feedback["previous_response_omitted_characters"] == length - 12000
