"""Provider failures retain bounded signals without echoed prompts or keys."""

import json

import httpx
import pytest

from agent.harness.gateway import EvidenceGateway
from agent.harness.context_limit_error import context_limit_error
from agent.harness.failure_summary import public_failure
from agent.harness.messages_bridge import MessagesBridge
from agent.harness.responses_bridge import ResponsesChatBridge
from agent.harness.responses_gateway import ResponsesGateway, ResponsesPolicy
from agent.rsi.engine.model_gateway import GatewayPolicy, ModelGateway, classify_upstream_error


def test_untrusted_error_fields_never_enter_logs():
    raw = json.dumps(
        {
            "error": {
                "code": "private-code",
                "type": "private-type",
                "param": "private-param",
                "message": "private prompt and secret; missing reasoning_content for tool_calls",
            }
        }
    ).encode()
    result = classify_upstream_error(raw)
    assert result["reported_signals"] == ["tool_protocol", "reasoning_protocol"]
    assert result["mentioned_fields"] == ["reasoning_content", "tool_calls"]
    assert not any(x in json.dumps(result) for x in ["private", "secret"])
    assert classify_upstream_error(b"not JSON")["format"] == "unparsed"
    assert classify_upstream_error(b"[]")["reported_signals"] == []


@pytest.mark.asyncio
@pytest.mark.parametrize("gateway", [ModelGateway, EvidenceGateway])
@pytest.mark.parametrize("oversized", [False, True])
async def test_rejected_request_keeps_status_bounded_diagnostics_and_failed_outcome(tmp_path, gateway, oversized):
    raw = (
        b"private-prompt " * 10000
        if oversized
        else json.dumps({"error": {"code": "UnavailableModel", "message": "private-prompt upstream-secret"}}).encode()
    )
    path = tmp_path / "requests.jsonl"
    async with gateway(
        policy=GatewayPolicy(model="test", mode="science_agent", scientific_tools=set()),
        upstream_url="https://provider.example/v1",
        api_key="upstream-secret",
        output=path,
        transport=httpx.MockTransport(lambda request: httpx.Response(400, content=raw)),
    ) as relay:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                relay.base_url + "/chat/completions",
                headers={"Authorization": "Bearer " + relay.token},
                json={"model": "test", "messages": [{"role": "user", "content": "private-prompt"}]},
            )
        assert response.status_code == 400
    records = [json.loads(line) for line in path.read_text().splitlines()]
    diagnostic = next(x for x in records if x["event"] == "upstream_rejection")
    assert diagnostic["inspected_bytes"] <= 65536 and diagnostic["truncated"] is oversized
    assert diagnostic["reported_signals"] == ([] if oversized else ["model_unavailable"])
    assert records[-1]["status"] == 400 and records[-1]["completed"] is False
    assert not any(x in path.read_text() for x in ["private-prompt", "upstream-secret", relay.token])


def overflow_gateway(kind, calls=2):
    policy = GatewayPolicy(model="test", mode="science_agent", scientific_tools=set(),
                           max_calls=calls, max_tokens=4096, thinking="disabled")
    request = {"model": "test", "stream": False,
               "messages": [{"role": "user", "content": "private-prompt context_length_exceeded"}]}
    if kind == "chat":
        return EvidenceGateway, policy, "/chat/completions", request
    if kind == "messages":
        return MessagesBridge, policy, "/messages", {**request, "max_tokens": 4096}
    request = {"model": "test", "stream": False, "store": False,
               "input": "private-prompt context_length_exceeded"}
    if kind == "responses_bridge":
        return ResponsesChatBridge, policy, "/responses", request
    return (ResponsesGateway, ResponsesPolicy(model="test", scientific_tools=set(), max_calls=calls),
            "/responses", request)


def success_response(kind):
    if kind == "responses":
        return {"id": "resp1", "model": "test", "status": "completed",
                "output": [{"id": "msg1", "type": "message", "role": "assistant", "status": "completed",
                            "content": [{"type": "output_text", "text": "fixture answer", "annotations": []}]}],
                "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}}
    return {"id": "chat1", "model": "test",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "fixture answer"},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}


def assert_generic_error_body(body, kind, status):
    error = {"type": "api_error", "message": "upstream_http_" + str(status)}
    if kind == "messages":
        assert body == {"type": "error", "error": error}
    elif kind == "responses_bridge":
        assert body == {"error": error}
    else:
        assert body == {"error": "upstream_http_" + str(status)}


def assert_overflow_body(body, kind):
    if kind == "messages":
        assert body == {"type": "error", "error": {
            "type": "invalid_request_error", "message": "prompt is too long"}}
    else:
        assert body == {"error": {"type": "invalid_request_error", "code": "context_length_exceeded",
            "message": "Your input exceeds the context window of this model.", "param": None}}


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["chat", "messages", "responses", "responses_bridge"])
@pytest.mark.parametrize("status", [400, 413, 429, 500])
async def test_native_overflow_error_is_canonical_redacted_and_preserves_accounting(tmp_path, kind, status):
    gateway, policy, route, payload = overflow_gateway(kind)
    path = tmp_path / "requests.jsonl"
    seen = []
    def upstream(request):
        seen.append(request)
        return httpx.Response(status, json={"error": {"code": "context_length_exceeded",
            "message": "maximum context; private-prompt upstream-secret /private/source"}})
    async with gateway(policy=policy, upstream_url="https://fixture.invalid/v1", api_key="upstream-secret",
                       output=path, transport=httpx.MockTransport(upstream), stop_on_rate_limit=True) as relay:
        async with httpx.AsyncClient(trust_env=False) as client:
            kwargs = {"headers": {"Authorization": "Bearer " + relay.token}, "json": payload}
            response = await client.post(relay.base_url + route, **kwargs)
            assert response.status_code == (400 if status == 429 else status)
            if status in (400, 413):
                assert_overflow_body(response.json(), kind)
                assert public_failure(path)["code"] == "context_limit"
            else:
                assert_generic_error_body(response.json(), kind, status)
                assert "context_length_exceeded" not in response.text
                assert public_failure(path)["code"] == ("upstream_rate_limited" if status == 429 else "model_unavailable")
            if status == 429:
                assert (await client.post(relay.base_url + route, **kwargs)).status_code == 400
            assert relay.calls == len(seen) == 1
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rejection = next(row for row in rows if row["event"] == "upstream_rejection")
    assert rejection["status"] == status and "context_limit" in rejection["reported_signals"]
    final = next(row for row in reversed(rows) if row["event"] == "response")
    assert final["call"] == rejection["call"] == 1
    assert final["status"] == status and final["completed"] is False
    for secret in ["private-prompt", "upstream-secret", "/private/source", relay.token]:
        assert secret not in response.text + path.read_text() + json.dumps(public_failure(path))


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["chat", "messages", "responses", "responses_bridge"])
@pytest.mark.parametrize("later_status", [200, 400, 413, 500])
async def test_later_call_is_not_contaminated_by_prior_overflow(tmp_path, kind, later_status):
    gateway, policy, route, payload = overflow_gateway(kind)
    path = tmp_path / "requests.jsonl"
    calls = []
    def upstream(request):
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(400, json={"error": {"code": "context_length_exceeded"}})
        if later_status == 200:
            return httpx.Response(200, json=success_response(kind))
        return httpx.Response(later_status, json={"error": {"message": "Unrelated rejection upstream-secret"}})
    async with gateway(policy=policy, upstream_url="https://fixture.invalid/v1", api_key="upstream-secret",
                       output=path, transport=httpx.MockTransport(upstream)) as relay:
        async with httpx.AsyncClient(trust_env=False) as client:
            kwargs = {"headers": {"Authorization": "Bearer " + relay.token}, "json": payload}
            first = await client.post(relay.base_url + route, **kwargs)
            assert_overflow_body(first.json(), kind)
            assert public_failure(path)["code"] == "context_limit"
            second = await client.post(relay.base_url + route, **kwargs)
            assert second.status_code == later_status, second.text
            assert "context_length_exceeded" not in second.text
            if later_status != 200:
                assert_generic_error_body(second.json(), kind, later_status)
            expected = "runtime_error" if later_status == 200 else "model_unavailable"
            assert public_failure(path)["code"] == expected
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            final = next(row for row in reversed(rows) if row["event"] == "response")
            assert final["call"] == 2 and final["completed"] is (later_status == 200)
            assert relay.calls == len(calls) == 2
            exhausted = await client.post(relay.base_url + route, **kwargs)
            assert exhausted.status_code == 400
            assert relay.calls == len(calls) == 2
            assert public_failure(path)["code"] == "model_call_budget_exhausted"
    assert "upstream-secret" not in path.read_text()


@pytest.mark.parametrize("evidence,status", [
    ({"event": "request", "status": 400, "reported_signals": ["context_limit"]}, 400),
    ({"event": "upstream_rejection", "status": 413, "reported_signals": ["context_limit"]}, 400),
    ({"event": "upstream_rejection", "status": 400, "reported_signals": "context_limit"}, 400),
    ({"event": "upstream_rejection", "status": 400, "message": "context_length_exceeded"}, 400),
    ({"event": "upstream_rejection", "status": 200, "reported_signals": ["context_limit"]}, 200),
])
def test_only_matching_classified_rejection_can_produce_overflow(evidence, status):
    assert context_limit_error(evidence, status) is None


@pytest.mark.parametrize("call,status", [(2, 400), (1, 413), (None, 400)])
def test_public_failure_requires_same_call_and_status(tmp_path, call, status):
    path = tmp_path / "requests.jsonl"
    rows = [{"event": "upstream_rejection", "call": 1, "status": 400, "reported_signals": ["context_limit"]},
            {"event": "response", "call": call, "status": status, "completed": False}]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    assert public_failure(path)["code"] == "model_unavailable"
