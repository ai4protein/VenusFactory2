"""Format compatibility must not rewrite plans, hide questions, or bypass acceptance."""
import copy
import hashlib
import json
from types import SimpleNamespace

import pytest

from agent.harness.autonomous_research import AutonomousResearch
from agent.harness.research_plan_format import remove_empty_step_inputs
from agent.harness.scientist_role_output import RoleOutputError, specification, validate_output


def plan():
    return {"summary": "Develop a protein model", "deliverable": "model", "needs_input": ["Confirm assay units"],
            "steps": [{"id": "fit", "title": "Fit the baseline", "acceptance": "Measure held-out error",
                       "approach": "Compare sequence features with a conventional baseline", "verification": "execution",
                       "needs_input": []},
                      {"id": "report", "title": "Deliver", "acceptance": "Reproduce predictions", "verification": "delivery"}]}


@pytest.mark.parametrize("protocol", ["chat_completions", "responses"])
def test_only_empty_step_field_removed_all_legal_values_order_and_questions_preserved(protocol):
    original = plan()
    raw = json.dumps(original)
    with pytest.raises(RoleOutputError):
        validate_output(raw, specification("CB", protocol))
    adjusted = remove_empty_step_inputs(raw, protocol)
    expected = copy.deepcopy(original)
    del expected["steps"][0]["needs_input"]
    assert json.loads(adjusted["text"]) == expected
    assert adjusted["removed_empty_fields"] == ["/steps/0/needs_input"]
    assert original == plan()


@pytest.mark.parametrize("bad", [["Need measurements"], None, False, "", {}, 0])
def test_nonempty_or_mistyped_step_questions_are_not_discarded(bad):
    original = plan()
    original["steps"][0]["needs_input"] = bad
    assert remove_empty_step_inputs(json.dumps(original), "chat_completions") is None


def test_unknown_fields_missing_root_and_other_normalizations_still_rejected():
    for mutate in [lambda p: p.pop("needs_input"),
                   lambda p: p["steps"][0].update(comparison="unrequested"),
                   lambda p: p["steps"][0].update(status="completed"),
                   lambda p: p["steps"][0].update(verification="data"),
                   lambda p: p["steps"][0].pop("acceptance"),
                   lambda p: p.update(needs_input=[1])]:
        original = plan()
        mutate(original)
        assert remove_empty_step_inputs(json.dumps(original), "chat_completions") is None


@pytest.mark.parametrize("raw", ["{", "[]", 'null', '{"summary": "first", "summary": "last"}', '{"x":NaN}'])
def test_invalid_json_not_silently_repaired(raw):
    assert remove_empty_step_inputs(raw, "chat_completions") is None


@pytest.mark.asyncio
async def test_structured_records_original_and_lossless_adjustment_without_second_model_request(tmp_path):
    runner = object.__new__(AutonomousResearch)
    raw = json.dumps(plan())
    calls = []
    async def consult(*args):
        calls.append(args)
        raise RoleOutputError("Unexpected nested needs_input", raw)
    runner.consult = consult
    runner.activity = lambda *args: None
    profile = SimpleNamespace(model=SimpleNamespace(protocol="chat_completions"), budget=SimpleNamespace(model_calls=16))
    result = await runner.structured("CB", {}, profile, SimpleNamespace(calls=1), tmp_path, json.loads)
    assert len(calls) == 1
    assert result["needs_input"] == ["Confirm assay units"]
    assert result["steps"][0]["acceptance"] == "Measure held-out error"
    original_files = list((tmp_path / "role_outputs").glob("*.txt"))
    assert len(original_files) == 1 and original_files[0].read_text() == raw
    records = list((tmp_path / "role_normalizations").glob("*.json"))
    assert len(records) == 1
    record = json.loads(records[0].read_text())
    assert record["original_response"]["response_sha256"] == hashlib.sha256(raw.encode()).hexdigest()
    assert record["canonical_sha256"] == hashlib.sha256(record["canonical_text"].encode()).hexdigest()
    assert json.loads(record["canonical_text"]) == result
    assert record["scientific_fields_changed"] is False


@pytest.mark.asyncio
async def test_scientific_plan_validator_still_rejects_after_format_adjustment(tmp_path):
    runner = object.__new__(AutonomousResearch)
    raw = json.dumps(plan())
    async def consult(*args):
        raise RoleOutputError("Unexpected nested needs_input", raw)
    runner.consult = consult
    runner.activity = lambda *args: None
    profile = SimpleNamespace(model=SimpleNamespace(protocol="chat_completions"), budget=SimpleNamespace(model_calls=16))
    def reject(_):
        raise ValueError("Previous acceptance endpoint changed")
    with pytest.raises(ValueError, match="Previous acceptance endpoint changed"):
        await runner.structured("CB", {}, profile, SimpleNamespace(calls=99), tmp_path, reject)
    assert len(list((tmp_path / "role_outputs").glob("*.txt"))) == 1
