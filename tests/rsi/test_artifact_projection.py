import pytest

from agent.rsi.artifact_projection import project, successful_results


def ref(call_id, path=None):
    return {"op": "ref", "call_id": call_id, "path": path or []}


def test_nested_false_and_empty_values_preserved_without_mutating_evidence():
    evidence = {"call": {"valid": False, "issues": [], "nested": {"position": 2}}}
    result, used = project({"op": "object", "fields": {"audit": ref("call")}}, evidence)
    assert result == {"audit": {"valid": False, "issues": [], "nested": {"position": 2}}}
    assert used == ["call"]
    result["audit"]["nested"]["position"] = 99
    assert evidence["call"]["nested"]["position"] == 2


def test_missing_field_or_result_cannot_be_filled():
    with pytest.raises(ValueError, match="absent"):
        project(ref("missing"), {})
    with pytest.raises(ValueError, match="absent"):
        project(ref("call", ["missing"]), {"call": {}})


def test_unsuccessful_or_unpaired_tool_result_is_not_evidence():
    events = [
        {"kind": "tool_result", "tool_call_id": "unpaired", "output": "{}"},
        {"kind": "tool_call_start", "tool_call_id": "failed"},
        {"kind": "tool_result", "tool_call_id": "failed", "is_error": True, "output": "{}"},
    ]
    assert successful_results(events) == {}


def test_boolean_predicate_does_not_treat_zero_as_false():
    contract = {"op": "equals", "value": ref("call", ["valid"]), "expected_literal": False}
    assert project(contract, {"call": {"valid": False}})[0] is True
    assert project(contract, {"call": {"valid": 0}})[0] is False


def test_nonfinite_and_unknown_operators_rejected():
    with pytest.raises(ValueError):
        project(ref("call"), {"call": float("nan")})
    with pytest.raises(ValueError):
        project({"op": "execute_python", "code": "print(1)"}, {})
