import copy
import json

import pytest

from agent.harness.responses_gateway import ResponsesPolicy, inspect_responses


def declaration():
    return {
        "type": "namespace",
        "name": "mcp__venusfactory",
        "description": "Fixture",
        "tools": [
            {"type": "function", "name": "inspect_task", "parameters": {"type": "object"}},
            {"type": "function", "name": "unadmitted"},
        ],
    }


def policy():
    return ResponsesPolicy(model="fixture", scientific_tools={"inspect_task"})


def test_native_namespace_is_preserved_with_only_admitted_functions():
    original = declaration()
    sent, metadata = policy().apply(
        {
            "model": "fixture",
            "input": "fixture",
            "tools": [original],
            "tool_choice": {"type": "function", "namespace": "mcp__venusfactory", "name": "inspect_task"},
        }
    )
    assert sent["tools"] == [{**original, "tools": [original["tools"][0]]}]
    assert len(original["tools"]) == 2
    assert metadata["tools_sent"] == ["mcp__venusfactory__inspect_task"]


def test_duplicate_flat_and_namespaced_function_is_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        policy().apply(
            {
                "model": "fixture",
                "input": "fixture",
                "tools": [declaration(), {"type": "function", "name": "mcp__venusfactory__inspect_task"}],
            }
        )


@pytest.mark.parametrize("namespace,name", [("mcp", "venusfactory__inspect_task"), ("other", "inspect_task")])
def test_namespace_identity_cannot_spoof_allowed_tool(namespace, name):
    tool = declaration()
    tool["name"] = namespace
    tool["tools"] = [{"type": "function", "name": name}]
    sent, metadata = policy().apply({"model": "fixture", "input": "fixture", "tools": [tool]})
    assert "tools" not in sent and metadata["tools_sent"] == []


def test_response_call_retains_namespace_and_rejects_different_namespace():
    value = {
        "id": "r1",
        "model": "fixture",
        "status": "completed",
        "output": [
            {
                "id": "f1",
                "type": "function_call",
                "name": "inspect_task",
                "namespace": "mcp__venusfactory",
                "call_id": "c1",
                "arguments": "{}",
                "status": "completed",
            }
        ],
    }
    options = {"model": "fixture", "allowed_tools": {"mcp__venusfactory__inspect_task"}, "max_output_tokens": 100}
    assert inspect_responses(json.dumps(value).encode(), False, **options)["usage"] is None
    changed = copy.deepcopy(value)
    changed["output"][0]["namespace"] = "other"
    with pytest.raises(ValueError, match="Undeclared"):
        inspect_responses(json.dumps(changed).encode(), False, **options)
