"""Responses roles must preserve protocol identity and scientific tool validation."""

import copy
from types import SimpleNamespace

import pytest

from agent.harness.codex_adapter import validate_tool_result
from agent.harness.native_expert_team import role_content


def response():
    return {
        "status": "completed",
        "model": "bound-model",
        "output": [
            {"type": "reasoning", "encrypted_content": "opaque"},
            {
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "Observed result"}],
            },
        ],
    }


def test_responses_role_returns_text_without_opaque_reasoning():
    assert role_content(response(), "responses", "bound-model") == "Observed result"


@pytest.mark.parametrize(
    "corruption", ["model", "incomplete", "function", "refusal", "blank", "large", "missing", "role"]
)
def test_incomplete_or_nontext_roles_cannot_finish(corruption):
    value = copy.deepcopy(response())
    if corruption == "model":
        value["model"] = "different-model"
    elif corruption == "incomplete":
        value["status"] = "incomplete"
    elif corruption == "function":
        value["output"].append({"type": "function_call", "name": "unadmitted"})
    elif corruption == "refusal":
        value["output"][1]["content"] = [{"type": "refusal", "refusal": "Cannot execute"}]
    elif corruption in {"blank", "large"}:
        value["output"][1]["content"][0]["text"] = "  " if corruption == "blank" else "x" * 128001
    elif corruption == "role":
        value["output"][1]["role"] = "user"
    else:
        value.pop("output")
    with pytest.raises(ValueError):
        role_content(value, "responses", "bound-model")


@pytest.mark.parametrize("phase", ["develop", "reuse", "probe"])
def test_codex_uses_domain_validator_for_declared_phase(monkeypatch, phase):
    from agent.rsi import reproduction_tool_contracts as reproduction
    from agent.rsi import tool_contracts as generated

    seen = []
    monkeypatch.setattr(reproduction, "SUPPORTED", {"reproduce_fixture"})
    monkeypatch.setattr(
        reproduction, "validate_reproduction_result", lambda *args, **kwargs: seen.append("reproduction")
    )
    monkeypatch.setattr(generated, "validate_generated_result", lambda *args, **kwargs: seen.append("generated"))
    task = SimpleNamespace(mcp_entry="generated_tools.py", phase=phase, workspace="work", capability="cap")
    validate_tool_result(task, "reproduce_fixture", {"success": True})
    assert seen == ["reproduction"]
    validate_tool_result(task, "inspect_model_code_task", {"success": True})
    assert seen == (["reproduction", "generated"] if phase in {"develop", "reuse"} else ["reproduction"])
