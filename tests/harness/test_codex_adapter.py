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
    assert cfg["model_providers"]["venus_research"]["capabilities"]["remote_compaction"] == "unsupported"
    assert cfg["model_context_window"] == p.model.context_tokens
    assert "model_auto_compact_token_limit" not in cfg  # Preserve native thresholds.
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


@pytest.mark.parametrize("error_source", ["failed_status", "isError", "is_error", "transport", "structured"])
def test_native_failure_preserves_repair_evidence_and_attempt_identity(tmp_path, error_source):
    from agent.harness.research_execution_feedback import feedback_state

    identity = "a" * 32
    diagnostic = f"TrainingAttemptError: Training attempt {identity} failed: fit is an unfinished placeholder"
    state = CodexEvents({"train_research_model"}, 1, workspace=tmp_path)
    state.consume({"type": "thread.started", "thread_id": "fixture-thread"})
    state.consume({"type": "turn.started"})
    arguments = {"source_file": "src/model.py"}
    events = state.consume(tool_event(tool="train_research_model", arguments=arguments))
    result = {"content": [{"type": "text", "text": diagnostic}], "structured_content": None}
    changes = {"tool": "train_research_model", "arguments": arguments, "result": result}
    if error_source == "failed_status":
        changes["status"] = "failed"
    elif error_source in {"isError", "is_error"}:
        result[error_source] = True
    elif error_source == "transport":
        changes.update(result=None, error={"message": diagnostic, "private_metadata": "must not be forwarded"})
    else:
        changes.update(status="failed", result={"structured_content": {"error": diagnostic, "private_metadata": "must not be forwarded"}})
    events += state.consume(tool_event("item.completed", **changes))
    assert events[-1]["is_error"] is True
    assert events[-1]["output"] == diagnostic
    feedback = feedback_state(events, tmp_path)
    assert feedback["pending"][0]["training_attempt_id"] == identity
    assert feedback["pending"][0]["source_file"] == "src/model.py"
    assert "unfinished placeholder" in feedback["errors"][0]["error"]
    assert state.tool_errors == 1


def test_native_error_redacts_credentials_and_host_paths_before_bounding(tmp_path):
    state = CodexEvents({"inspect_task"}, 1, workspace=tmp_path)
    state.consume({"type": "thread.started", "thread_id": "fixture-thread"})
    state.consume({"type": "turn.started"})
    state.consume(tool_event())
    diagnostic = (
        "ValueError: invalid input " + str(tmp_path / "src/model.py") + "\n"
        "Other file: /home/another-user/private/model.py\n"
        "Authorization: Bearer " + "fixture-secret-" * 20 + "\n"
        "API_KEY=sk-" + "fixture-key-" * 20 + "\n"
        "-----BEGIN " "RSA PRIVATE KEY-----\nprivate-body\n-----END RSA PRIVATE KEY-----\n"
        + "padding " * 2000 + "\nRepair the missing expected_sha256 argument."
    )
    output = state.consume(tool_event("item.completed", status="failed", result=None, error={"message": diagnostic}))[0]["output"]
    assert len(output) <= 1600
    assert output.startswith("ValueError: invalid input /workspace/src/model.py")
    assert output.endswith("Repair the missing expected_sha256 argument.")
    assert all(value not in output for value in (str(tmp_path), "another-user", "fixture-secret", "fixture-key", "private-body"))


@pytest.mark.parametrize("result", [None, {}, {"content": [None, {"type": "image", "data": "private"}]}, {"structured_content": {"other": "private"}}, {"content": None}])
def test_native_error_without_message_does_not_forward_metadata(result):
    state = initialized()
    state.consume(tool_event())
    output = state.consume(tool_event("item.completed", status="failed", result=result, error={"code": 42}))[0]["output"]
    assert output == "Native MCP tool failed without a diagnostic."


def test_failed_output_is_not_success_evidence_and_success_still_requires_validation(tmp_path):
    validated = []

    def validate(name, data):
        validated.append((name, data))
        raise ValueError("Invalid success evidence")

    state = CodexEvents({"inspect_task"}, 2, validate, workspace=tmp_path)
    state.consume({"type": "thread.started", "thread_id": "fixture-thread"})
    state.consume({"type": "turn.started"})
    state.consume(tool_event())
    state.consume(tool_event("item.completed", status="failed", result={"content": [{"type": "text", "text": "Bad input"}]}))
    assert validated == []
    state.consume(tool_event(id="second"))
    with pytest.raises(ValueError, match="Invalid success evidence"):
        state.consume(tool_event("item.completed", id="second"))
    assert validated == [("inspect_task", {"x": 1})]


@pytest.mark.parametrize("diagnostic, hidden", [
    ("Authorization: Bearer fixtureBearerCredential123456789", "fixtureBearerCredential123456789"),
    ("API_KEY=sk-" + "fixture" * 8, "sk-" + "fixture" * 8),
    ("-----BEGIN " "RSA PRIVATE KEY-----\nfixture-private-body\n-----END RSA PRIVATE KEY-----", "fixture-private-body"),
    ("Cannot read /home/other-user/private/file.py", "/home/other-user"),
])
def test_short_native_errors_are_redacted_without_relying_on_truncation(diagnostic, hidden):
    state = initialized()
    state.consume(tool_event())
    output = state.consume(tool_event("item.completed", status="failed", result={"content": [{"type": "text", "text": diagnostic}]}))[0]["output"]
    assert hidden not in output
    assert "trace shortened" not in output
    assert output


def test_native_compaction_advisory_preserves_thread_and_tool_continuation():
    import hashlib
    import json
    from agent.harness.codex_adapter import COMPACTION_NOTICE

    state = initialized(limit=2)
    state.consume(tool_event())
    state.consume(tool_event("item.completed"))
    raw = {"type": "item.completed", "item": {"id": "compact-notice", "type": "error", "message": COMPACTION_NOTICE}}
    row = state.consume(raw)[0]
    encoded = json.dumps(raw, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    assert row == {"kind": "context_compaction", "phase": "notice",
                   "native_type": "item.error.compaction_notice", "session_id": "fixture-thread",
                   "native_event_path": "native_events.jsonl",
                   "native_event_sha256": hashlib.sha256(encoded.encode()).hexdigest()}
    start = state.consume(tool_event(id="after-compaction"))[0]
    end = state.consume(tool_event("item.completed", id="after-compaction"))[0]
    assert start["session_id"] == end["session_id"] == row["session_id"]
    state.consume({"type": "turn.completed"})
    assert state.turn_completed and not state.failed and state.compaction_notices == 1


@pytest.mark.parametrize("variant", ["prefix", "suffix", "real_error", "started"])
def test_only_exact_completed_native_compaction_advisory_is_nonfatal(variant):
    from agent.harness.codex_adapter import COMPACTION_NOTICE

    state = initialized()
    message = {"prefix": "Error: " + COMPACTION_NOTICE, "suffix": COMPACTION_NOTICE + " failed",
               "real_error": "Context compaction failed", "started": COMPACTION_NOTICE}[variant]
    state.consume({"type": "item.started" if variant == "started" else "item.completed",
                   "item": {"type": "error", "message": message}})
    assert state.failed and state.compaction_notices == 0
    with pytest.raises(ValueError, match="Tool outside"):
        state.consume(tool_event())


@pytest.mark.parametrize("phase", ["before", "after", "failed"])
def test_native_compaction_advisory_requires_active_turn(phase):
    from agent.harness.codex_adapter import COMPACTION_NOTICE

    state = CodexEvents({"inspect_task"}, 1) if phase == "before" else initialized()
    if phase == "after":
        state.consume({"type": "turn.completed"})
    elif phase == "failed":
        state.consume({"type": "turn.failed"})
    with pytest.raises(ValueError, match="outside active"):
        state.consume({"type": "item.completed", "item": {"type": "error", "message": COMPACTION_NOTICE}})
