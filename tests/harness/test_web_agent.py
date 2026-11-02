"""Exercise native agent orchestration and immutable Web admission."""

import asyncio
import json
from copy import deepcopy

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, messages_from_dict
from langchain_core.tools import tool
from pydantic import ValidationError

from agent.harness import web_langgraph_worker as worker
from agent.harness.catalog import ExecutionCatalog
from agent.harness.contracts import Budget, ModelBinding, Profile
from agent.session_store import persistable_state
from backend.web_v2 import agent_execution as execution
from backend.web_v2.chat_api._models import ChatStreamRequest


class ScriptedModel(GenericFakeChatModel):
    disable_streaming: bool = True

    def bind_tools(self, tools, **kwargs):
        return self


@tool
def residue_count(sequence: str) -> str:
    """Count residues in a supplied protein sequence."""
    return json.dumps({"length": len(sequence)})


def request(tmp_path, **kwargs):
    return {"text": "Count MKAA", "workspace": str(tmp_path), "budget": {"tool_calls": 2}, **kwargs}


def test_native_loop_tool_identity_and_restored_conversation(tmp_path, monkeypatch):
    events = []
    monkeypatch.setattr(worker, "emit", events.append)
    model = ScriptedModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "mcp__venusfactory__residue_count",
                            "args": {"sequence": "MKAA"},
                            "id": "real-call-42",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(content="There are four residues."),
            ]
        )
    )
    asyncio.run(worker.run(request(tmp_path), tools=[residue_count], model=model))
    start = next(e for e in events if e["kind"] == "tool_call_start")
    end = next(e for e in events if e["kind"] == "tool_result")
    assert start["tool_call_id"] == end["tool_call_id"] == "real-call-42"
    assert json.loads(end["output"]) == {"length": 4}
    assert not end["is_error"]
    completed = events[-1]
    assert completed["kind"] == "completed" and completed["tool_calls"] == 1
    events.clear()
    asyncio.run(
        worker.run(
            request(tmp_path, text="What did I provide?", messages=completed["messages"]),
            tools=[residue_count],
            model=ScriptedModel(messages=iter([AIMessage(content="MKAA.")])),
        )
    )
    history = messages_from_dict(events[-1]["messages"])
    assert [m.type for m in history] == ["human", "ai", "tool", "ai", "human", "ai"]
    assert history[0].content == "Count MKAA"
    assert events[-1]["tool_calls"] == 0


@pytest.mark.parametrize("answer", ["Please provide a sequence.", "VHH is a single antibody variable domain."])
def test_direct_answer_or_clarification_does_not_force_a_tool(tmp_path, monkeypatch, answer):
    events = []
    monkeypatch.setattr(worker, "emit", events.append)
    asyncio.run(
        worker.run(
            request(tmp_path), tools=[residue_count], model=ScriptedModel(messages=iter([AIMessage(content=answer)]))
        )
    )
    assert events[-1]["text"] == answer and events[-1]["tool_calls"] == 0


@pytest.mark.parametrize("empty", [[], [{"type": "reasoning", "reasoning": "private"}], " "])
def test_empty_final_answer_is_not_success(tmp_path, monkeypatch, empty):
    monkeypatch.setattr(worker, "emit", lambda _: None)
    with pytest.raises(ValueError, match="finish with an answer"):
        asyncio.run(
            worker.run(request(tmp_path), tools=[], model=ScriptedModel(messages=iter([AIMessage(content=empty)])))
        )


def test_tool_budget_enforced_before_second_execution(tmp_path, monkeypatch):
    events = []
    monkeypatch.setattr(worker, "emit", events.append)
    turns = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "mcp__venusfactory__residue_count",
                    "args": {"sequence": "M"},
                    "id": str(i),
                    "type": "tool_call",
                }
            ],
        )
        for i in range(2)
    ]
    with pytest.raises(RuntimeError, match="budget exceeded"):
        asyncio.run(
            worker.run(
                request(tmp_path, budget={"tool_calls": 1}),
                tools=[residue_count],
                model=ScriptedModel(messages=iter(turns)),
            )
        )
    assert len([e for e in events if e["kind"] == "tool_call_start"]) == 1


@pytest.fixture
def admitted(monkeypatch, tmp_path):
    monkeypatch.setenv("TEST_WEB_AGENT_KEY", "test-secret")
    profile = Profile(
        mode="science_agent",
        harness="langgraph",
        model=ModelBinding(
            provider="test", model="model-a", base_url="https://example.test/v1", credential_env="TEST_WEB_AGENT_KEY"
        ),
        budget=Budget(),
    )
    cat = ExecutionCatalog(tmp_path, profile)
    monkeypatch.setattr(execution, "catalog", lambda: cat)
    monkeypatch.setattr(execution, "free_model", lambda binding: binding.provider == "test")
    import sys
    monkeypatch.setattr(execution, "runtime_paths", lambda: {"python_bin": sys.executable})
    binding, key = execution.admit({}, {"harness": "langgraph", "model_id": "profile-default"})
    assert key == "test-secret"
    return binding, cat


def test_reload_retry_frozen_profile_not_current_catalog(admitted, monkeypatch):
    binding, cat = admitted
    state = {
        "session_id": "test",
        "agent_execution": binding,
        "agent_messages": [],
        "agent_retry_base": [],
        "agent_last_outcome": "completed",
    }
    persisted = persistable_state(state)
    monkeypatch.setattr(execution, "catalog", lambda: (_ for _ in ()).throw(AssertionError("must not reselect")))
    restored, key = execution.admit(persisted)
    assert restored == binding and key == "test-secret"
    public = execution.public_execution(persisted)
    assert public["model"] == "model-a"
    assert not any(s in json.dumps(public) for s in ["secret", "credential", "base_url"])
    altered = deepcopy(state)
    altered["agent_execution"]["profile"]["model"]["model"] = "different"
    with pytest.raises(ValueError, match="binding changed"):
        execution.admit(altered)


@pytest.mark.parametrize(
    "selection",
    [
        {"harness": "deepseek-harness", "model_id": "profile-default"},
        {"harness": "langgraph", "model_id": "missing"},
        {"harness": "langgraph", "model_id": "profile-default", "api_key": "injected"},
    ],
)
def test_unadmitted_selections_fail(admitted, selection):
    with pytest.raises(ValueError):
        execution.admit({}, selection)


def test_existing_context_cannot_silently_change_runtime(admitted):
    selection = {"harness": "langgraph", "model_id": "profile-default"}
    with pytest.raises(ValueError, match="new conversation"):
        execution.admit({"history": [{"role": "user", "content": "previous Kimi turn"}]}, selection)
    binding, _ = admitted
    with pytest.raises(ValueError, match="new conversation"):
        execution.admit({"agent_execution": binding}, {**selection, "model_id": "other"})
    with pytest.raises(ValidationError):
        ChatStreamRequest(execution={**selection, "base_url": "https://injected.test"})
