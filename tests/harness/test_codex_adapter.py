from dataclasses import replace
from types import SimpleNamespace

import pytest

from agent.harness import Budget, ModelBinding, Profile, builtin_registry
from agent.harness.codex_adapter import CodexEvents, configuration
from agent.harness.responses_gateway import ResponsesPolicy


def profile():
    return Profile(
        "science_agent",
        "codex",
        ModelBinding("fixture", "fixture-model", "http://localhost:1234/v1", protocol="responses"),
        Budget(thinking=None, temperature=None, reasoning_effort="low"),
    )


def test_profile_explicit_protocol_and_roundtrip_preserve_legacy_budget_shape(tmp_path):
    p = profile()
    assert Profile.from_dict(p.public()) == p
    legacy = Profile("science_agent", "kimi-code", ModelBinding("fixture", "m", "http://localhost:1234/v1"))
    assert "reasoning_effort" not in legacy.public()["budget"]
    with pytest.raises(ValueError, match="thinking=null"):
        ResponsesPolicy.from_profile(replace(p, budget=Budget()), set())
    with pytest.raises(ValueError):
        Budget(reasoning_effort="unknown")
    executable = tmp_path / "codex"
    executable.write_text("fixture")
    assert builtin_registry().preflight(p, {"codex_bin": str(executable)})["ready"]
    assert not builtin_registry().preflight(replace(p, mode="science_expert"), {"codex_bin": str(executable)})["ready"]


def test_config_uses_only_host_admitted_tools_and_gateway():
    p = profile()
    task = SimpleNamespace(tools={"inspect_task"}, mcp_entry="fixture_tools.py")
    ctx = SimpleNamespace(profile=p, task=task, relay=SimpleNamespace(base_url="http://127.0.0.1:12345/v1"))
    import tomllib

    cfg = tomllib.loads(configuration(ctx))
    assert cfg["model"] == p.model.model
    assert cfg["model_providers"]["venus_research"]["base_url"] == ctx.relay.base_url
    assert cfg["model_providers"]["venus_research"]["stream_max_retries"] == 0
    assert cfg["features"]["multi_agent"] is False and cfg["features"]["shell_tool"] is False
    assert cfg["mcp_servers"]["venusfactory"]["enabled_tools"] == ["inspect_task"]
    assert cfg["mcp_servers"]["venusfactory"]["tools"]["inspect_task"]["approval_mode"] == "approve"


def initialized(limit=1):
    value = CodexEvents({"inspect_task"}, limit)
    value.consume({"type": "thread.started", "thread_id": "fixture-thread"})
    value.consume({"type": "turn.started"})
    return value


def tool_event(kind="item.started", **changes):
    item = {
        "id": "item1",
        "type": "mcp_tool_call",
        "server": "venusfactory",
        "tool": "inspect_task",
        "arguments": {"x": 1},
        "status": "in_progress",
        "result": None,
        "error": None,
    }
    if kind == "item.completed":
        item.update(status="completed", result={"structured_content": {"x": 1}})
    item.update(changes)
    return {"type": kind, "item": item}


def test_native_tool_identity_arguments_and_structured_result_are_retained():
    state = initialized()
    start = state.consume(tool_event())[0]
    end = state.consume(tool_event("item.completed"))[0]
    assert start["tool_call_id"] == end["tool_call_id"] == "item1"
    assert start["tool_name"] == "mcp__venusfactory__inspect_task"
    assert start["args"] == {"x": 1} and end["is_error"] is False
    state.consume({"type": "item.completed", "item": {"type": "agent_message", "text": "done"}})
    state.consume({"type": "turn.completed"})
    assert state.turn_completed and not state.failed


@pytest.mark.parametrize("change", [{"tool": "other"}, {"server": "other"}, {"arguments": []}, {"id": ""}])
def test_unadmitted_or_malformed_native_call_rejected(change):
    with pytest.raises(ValueError):
        initialized().consume(tool_event(**change))


def test_unpaired_changed_duplicate_or_over_budget_calls_rejected():
    state = initialized()
    with pytest.raises(ValueError):
        state.consume(tool_event("item.completed"))
    state.consume(tool_event())
    with pytest.raises(ValueError):
        state.consume(tool_event())
    with pytest.raises(ValueError):
        state.consume(tool_event(id="second"))
    with pytest.raises(ValueError):
        state.consume(tool_event("item.completed", arguments={"x": 2}))
    with pytest.raises(ValueError):
        state.consume({"type": "turn.completed"})
    state.consume(tool_event("item.completed"))
    with pytest.raises(ValueError):
        state.consume(tool_event("item.completed"))


def test_native_failures_and_unadmitted_actions_cannot_complete():
    state = initialized()
    state.consume(tool_event())
    state.consume(tool_event("item.completed", status="failed", result=None, error={"message": "fixture failure"}))
    assert state.tool_errors == 1
    state.consume({"type": "turn.failed"})
    with pytest.raises(ValueError):
        state.consume({"type": "turn.completed"})
    with pytest.raises(ValueError):
        initialized().consume({"type": "item.started", "item": {"type": "command_execution"}})
