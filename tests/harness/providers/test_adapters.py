import asyncio
import json
from dataclasses import replace

import httpx
import pytest

from agent.harness import Budget, ModelBinding, Profile, Task, builtin_registry
from agent.harness.events import audit_requests, audit_trace, normalize_dsh
from agent.harness.gateway import EvidenceGateway, inspect_response
from agent.research_modes import get_research_mode
from agent.rsi.engine.model_gateway import GatewayPolicy


def model():
    return ModelBinding("provider-a", "glm-exact-alias", "https://example.org/v1")


def test_mode_is_scientific_semantics_and_preserves_legacy_web_routing():
    assert get_research_mode("science_agent").workflow == "adaptive_execution"
    assert get_research_mode("science_expert").roles == ("PI", "CB", "MLS", "SC")
    assert get_research_mode("science_agent").engine == "kimi-code"


def test_swap_adapter_keeps_model_and_budget_exact_and_does_not_import_sdk(tmp_path):
    executable = tmp_path / "runtime"
    executable.write_text("pinned executable")
    runtimes = {"kimi_bin": str(executable), "dsh_bin": str(executable)}
    registry = builtin_registry()
    p = Profile("science_agent", "kimi-code", model())
    a = registry.preflight(p, runtimes)
    b = registry.preflight(replace(p, harness="deepseek-harness"), runtimes)
    assert a["ready"] and b["ready"]
    assert a["profile"]["model"] == b["profile"]["model"]
    assert a["profile"]["budget"] == b["profile"]["budget"]
    assert a["runtime"]["sha256"] == b["runtime"]["sha256"]


def test_unsupported_combinations_are_explicit_not_fallback(tmp_path):
    exe = tmp_path / "runtime"
    exe.touch()
    registry = builtin_registry()
    p = Profile("science_agent", "kimi-code", replace(model(), protocol="responses"))
    result = registry.preflight(p, {"kimi_bin": str(exe)})
    assert not result["ready"]
    assert "model_protocol_incompatible" in result["reasons"]
    assert "adapter_not_implemented" not in result["reasons"]
    with pytest.raises(ValueError, match="rejected"):
        registry.resolve(p, {"kimi_bin": str(exe)})
    assert (
        "mode_not_implemented_for_harness"
        in registry.preflight(replace(p, mode="science_expert", harness="deepseek-harness"), {"dsh_bin": str(exe)})[
            "reasons"
        ]
    )


@pytest.mark.parametrize(
    "endpoint", ["https://key@example.org/v1", "https://example.org/v1?api_key=secret", "file:///secret"]
)
def test_credentials_cannot_be_embedded_in_public_endpoint(endpoint):
    with pytest.raises(ValueError):
        replace(model(), base_url=endpoint)


def test_unrecognized_inline_secret_field_is_rejected():
    public = Profile("science_agent", "kimi-code", model()).public()
    public["model"]["api_key"] = "should-not-be-accepted"
    with pytest.raises(TypeError):
        Profile.from_dict(public)


@pytest.mark.parametrize("budget", [{"model_calls": True}, {"wall_seconds": float("nan")}, {"tool_calls": 0}])
def test_nonfinite_or_ambiguous_budgets_rejected(budget):
    with pytest.raises(ValueError):
        Budget(**budget)


def test_source_and_input_hashes_detect_changes(tmp_path):
    work, caps = tmp_path / "work", tmp_path / "caps"
    work.mkdir()
    caps.mkdir()
    (work / "train.json").write_text("[]")
    (caps / "generated_tools.py").write_text("pass")
    task = Task(work, caps, tmp_path / "out", "prompt", "instructions", frozenset({"fit"}))
    before = task.binding()
    (work / "train.json").write_text("[1]")
    assert task.binding()["inputs"] != before["inputs"]
    assert task.binding()["capabilities"] == before["capabilities"]


def test_real_dsh_tool_events_normalize_without_losing_identity(tmp_path):
    call = {"type": "tool/call", "data": {"callId": "abc", "name": "mcp__venusfactory__fit", "arguments": '{"seed":3}'}}
    result = {
        "type": "tool/result",
        "data": {
            "message": {
                "content": [
                    {
                        "type": "tool-result",
                        "toolCallId": "abc",
                        "isError": False,
                        "content": [{"type": "text", "text": '{"success":true}'}],
                    }
                ]
            }
        },
    }
    rows = normalize_dsh(call) + normalize_dsh(result)
    assert rows[0]["args"] == {"seed": 3}
    path = tmp_path / "events.jsonl"
    path.write_text("\n".join(json.dumps(x) for x in rows))
    assert audit_trace(path, {"fit"}, 1)["passed"]
    path.write_text("\n".join(json.dumps(x) for x in rows[::-1]))
    assert not audit_trace(path, {"fit"}, 1)["passed"]
    path.write_text("\n".join(json.dumps(x) for x in rows + rows))
    assert not audit_trace(path, {"fit"}, 1)["passed"]


@pytest.mark.parametrize(
    "raw",
    [
        b'data: {"choices":[{"finish_reason":"stop"}]}\n\n',
        b"data: [DONE]\n\n",
        b'data: {"error": {"message":"private"}}\n\n',
    ],
)
def test_incomplete_http200_is_not_a_completed_response(raw):
    with pytest.raises(ValueError):
        inspect_response(raw, True)


class Heartbeats(httpx.AsyncByteStream):
    async def __aiter__(self):
        for _ in range(30):
            await asyncio.sleep(0.01)
            yield b": keepalive\n\n"


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [True, False])
async def test_shared_total_deadline_prevents_heartbeat_extension(tmp_path, stream):
    policy = GatewayPolicy(model="m", mode="science_agent", scientific_tools={"fit"})
    log = tmp_path / "log.jsonl"
    async with EvidenceGateway(
        policy=policy,
        upstream_url="https://example.org/v1",
        api_key="upstream-secret",
        output=log,
        timeout_s=0.05,
        transport=httpx.MockTransport(lambda r: httpx.Response(200, stream=Heartbeats())),
    ) as gateway:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                gateway.base_url + "/chat/completions",
                headers={"Authorization": "Bearer " + gateway.token},
                json={"model": "m", "stream": stream},
            )
            assert response.status_code == 502
    assert not audit_requests(log)["passed"]
    assert json.loads(log.read_text().splitlines()[-1])["error_type"] == "TimeoutError"
    assert "upstream-secret" not in log.read_text()


@pytest.mark.asyncio
async def test_buffered_sse_complete_before_downstream_disconnect_and_budget(tmp_path):
    raw = b'data: {"choices":[{"finish_reason":"stop"}],"usage":{"prompt_tokens":7}}\n\ndata: [DONE]\n\n'
    seen = []

    def upstream(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, content=raw)

    policy = GatewayPolicy(model="m", mode="science_agent", scientific_tools={"fit"}, max_calls=1)
    log = tmp_path / "log.jsonl"
    async with EvidenceGateway(
        policy=policy,
        upstream_url="https://example.org/v1",
        api_key="upstream-secret",
        output=log,
        transport=httpx.MockTransport(upstream),
    ) as gateway:
        async with httpx.AsyncClient() as client:
            url = gateway.base_url + "/chat/completions"
            payload = {"model": "m", "stream": True, "messages": [{"role": "user", "content": "private-prompt"}]}
            assert (await client.post(url, json=payload)).status_code == 401
            h = {"Authorization": "Bearer " + gateway.token}
            async with client.stream("POST", url, headers=h, json=payload) as response:
                assert response.status_code == 200
                # Exit without consuming its body, as an SDK may do after a finish event.
            assert audit_requests(log)["passed"]
            rejected = await client.post(url, headers=h, json=payload)
            assert rejected.status_code == 400
            assert rejected.json()["error"]["code"] == "model_call_budget_exhausted"
            assert gateway.calls == 1
    assert len(seen) == 1
    assert "private-prompt" not in log.read_text() and "upstream-secret" not in log.read_text()
    assert not audit_requests(log)["passed"]  # budget overrun remains a failed protocol


def test_same_harness_supports_both_modes_without_changing_model_or_budget(tmp_path):
    exe = tmp_path / "python"
    exe.touch()
    registry = builtin_registry()
    profile = Profile("science_agent", "langgraph", model())
    left = registry.preflight(profile, {"python_bin": str(exe)})
    right = registry.preflight(replace(profile, mode="science_expert"), {"python_bin": str(exe)})
    assert left["ready"] and right["ready"]
    assert left["adapter"] == right["adapter"]
    assert left["profile"]["model"] == right["profile"]["model"]
    assert left["profile"]["budget"] == right["profile"]["budget"]


@pytest.mark.parametrize("arguments", ['{"code":"unterminated', "[]", "null", "[", None])
def test_bad_dsh_arguments_are_observed_and_require_a_tool_error(tmp_path, arguments):
    event = {"type": "tool/call", "data": {
        "callId": "bad-input", "name": "mcp__venusfactory__fit", "arguments": arguments,
    }}
    original = json.loads(json.dumps(event))
    start = normalize_dsh(event)[0]
    assert event == original
    assert start["args"] is None and start["arguments_error"]
    path = tmp_path / "events.jsonl"
    for is_error in (True, False):
        result = {"kind": "tool_result", "tool_call_id": "bad-input",
                  "is_error": is_error, "output": "Argument validation"}
        path.write_text("\n".join(json.dumps(row) for row in (start, result)))
        audit = audit_trace(path, {"fit"}, 1)
        assert audit["passed"] is is_error
        assert audit["tool_calls"] == 1 and audit["tool_errors"] == int(is_error)
    path.write_text(json.dumps(start))
    assert "unpaired_tool_calls" in audit_trace(path, {"fit"}, 1)["problems"]


def test_recovery_after_bad_arguments_does_not_hide_budget_or_identity(tmp_path):
    rows = normalize_dsh({"type": "tool/call", "data": {
        "callId": "bad", "name": "mcp__venusfactory__fit", "arguments": '{"seed":',
    }})
    rows.append({"kind": "tool_result", "tool_call_id": "bad",
                 "is_error": True, "output": "Invalid JSON"})
    rows += normalize_dsh({"type": "tool/call", "data": {
        "callId": "retry", "name": "mcp__venusfactory__fit", "arguments": '{"seed":1}',
    }})
    rows.append({"kind": "tool_result", "tool_call_id": "retry",
                 "is_error": False, "output": "Fitted"})
    path = tmp_path / "events.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows))
    assert audit_trace(path, {"fit"}, 2)["passed"]
    assert "tool_budget_exceeded" in audit_trace(path, {"fit"}, 1)["problems"]
    assert "undeclared_tool" in audit_trace(path, {"other"}, 2)["problems"]
