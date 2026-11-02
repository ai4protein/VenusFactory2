import asyncio
import json
import time

import httpx
import pytest

from agent.rsi.model_gateway import GatewayPolicy, ModelGateway


def function(name):
    return {"type": "function", "function": {"name": name, "parameters": {"type": "object"}}}


def policy(mode="science_agent", max_calls=2):
    return GatewayPolicy(model="test-model", mode=mode, scientific_tools={"inspect", "submit"}, max_calls=max_calls)


def test_agent_exposure_and_sampling_are_applied_to_actual_payload():
    original = {
        "model": "test-model",
        "messages": [],
        "tools": [function("bash"), function("mcp__venusfactory__inspect")],
        "temperature": 1.8,
        "max_completion_tokens": 99,
        "top_p": 0.3,
        "seed": 9,
    }
    sent, audit = policy().apply(original)
    assert sent["tools"] == [function("mcp__venusfactory__inspect")]
    assert sent["temperature"] == 0.1 and sent["max_tokens"] == 4096
    assert not {"top_p", "seed", "max_completion_tokens"} & sent.keys()
    assert len(original["tools"]) == 2
    assert "messages" not in audit
    with pytest.raises(ValueError, match="model identity"):
        policy().apply({"model": "wrong"})
    with pytest.raises(ValueError, match="selected tool"):
        policy().apply({**original, "tool_choice": {"type": "function", "function": {"name": "bash"}}})


def test_scientist_role_output_schemas_are_not_mistaken_for_executed_tools():
    payload = {"model": "test-model", "tools": [function("submitArguments")]}
    sent, audit = policy("science_expert").apply(payload)
    assert sent["tools"] == payload["tools"]
    assert audit["tools_sent"] == ["submitArguments"]


@pytest.mark.asyncio
async def test_relay_authentication_budget_and_secret_free_logs(tmp_path):
    calls = []

    def upstream(request):
        calls.append(json.loads(request.content))
        assert request.headers["Authorization"] == "Bearer upstream-secret"
        return httpx.Response(
            200,
            json={
                "id": "sample",
                "choices": [{"finish_reason": "stop"}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 1},
            },
        )

    log = tmp_path / "requests.jsonl"
    async with ModelGateway(
        policy=policy(max_calls=1),
        upstream_url="https://provider.example/v1",
        api_key="upstream-secret",
        output=log,
        transport=httpx.MockTransport(upstream),
    ) as relay:
        async with httpx.AsyncClient() as client:
            endpoint = relay.base_url + "/chat/completions"
            request = {
                "model": "test-model",
                "messages": [{"role": "user", "content": "private-prompt"}],
                "tools": [function("bash"), function("mcp__venusfactory__inspect")],
            }
            assert (await client.post(endpoint, json=request)).status_code == 401
            headers = {"Authorization": "Bearer " + relay.token}
            response = await client.post(endpoint, json=request, headers=headers)
            assert response.status_code == 200
            assert (await client.post(endpoint, json=request, headers=headers)).status_code == 429
    assert len(calls) == 1 and len(calls[0]["tools"]) == 1
    records = log.read_text()
    assert all(secret not in records for secret in ["upstream-secret", "private-prompt", relay.token])
    assert json.loads(records.splitlines()[1])["usage"]["prompt_tokens"] == 3


@pytest.mark.asyncio
async def test_sse_is_forwarded_and_usage_is_audited(tmp_path):
    raw = b'data: {"choices":[{"index":0,"finish_reason":"stop"}],"usage":{"prompt_tokens":7}}\n\ndata: [DONE]\n\n'

    def upstream(request):
        return httpx.Response(200, content=raw, headers={"content-type": "text/event-stream"})

    log = tmp_path / "stream.jsonl"
    async with ModelGateway(
        policy=policy(),
        upstream_url="https://provider.example/v1",
        api_key="secret",
        output=log,
        transport=httpx.MockTransport(upstream),
    ) as relay:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                relay.base_url + "/chat/completions",
                headers={"Authorization": "Bearer " + relay.token},
                json={"model": "test-model", "messages": [], "stream": True},
            )
            assert response.content == raw
    assert json.loads(log.read_text().splitlines()[-1])["usage"]["prompt_tokens"] == 7


def test_explicit_shared_thinking_overrides_runtime_hints():
    controlled = GatewayPolicy(
        model="test-model", mode="science_agent", scientific_tools=set(), max_tokens=12288, thinking="enabled"
    )
    sent, audit = controlled.apply(
        {"model": "test-model", "thinking": {"type": "disabled"}, "reasoning_effort": "medium"}
    )
    assert sent["max_tokens"] == 12288
    assert sent["thinking"] == {"type": "enabled", "clear_thinking": True}
    assert "reasoning_effort" not in sent
    assert audit["thinking_sent"] == sent["thinking"]


class SlowBody(httpx.AsyncByteStream):
    def __init__(self, streaming):
        self.streaming = streaming
        self.closed = False

    async def __aiter__(self):
        for _ in range(10):
            await asyncio.sleep(0.02)
            yield b": heartbeat\n\n" if self.streaming else b" "
        yield b"data: [DONE]\n\n" if self.streaming else b"{}"

    async def aclose(self):
        self.closed = True


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_absolute_deadline_expires_even_when_chunks_keep_arriving(tmp_path, streaming):
    body = SlowBody(streaming)

    async def upstream(request):
        return httpx.Response(200, stream=body)

    log = tmp_path / "deadline.jsonl"
    async with ModelGateway(
        policy=policy(),
        upstream_url="https://provider.example/v1",
        api_key="secret",
        output=log,
        transport=httpx.MockTransport(upstream),
        timeout_s=0.07,
    ) as relay:
        started = time.monotonic()
        async with httpx.AsyncClient(timeout=2) as client:
            response = await client.post(
                relay.base_url + "/chat/completions",
                headers={"Authorization": "Bearer " + relay.token},
                json={"model": "test-model", "stream": streaming},
            )
        assert time.monotonic() - started < 1
        assert response.status_code == (200 if streaming else 502)
    records = [json.loads(line) for line in log.read_text().splitlines()]
    assert records[-1]["completed"] is False
    assert records[-1]["error_type"] == "TimeoutError"
    assert records[-1]["total_deadline_s"] == 0.07
    assert body.closed


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "raw",
    [
        b'data: {"choices":[{"index":0,"finish_reason":"stop"}]}\n\n',
        b"data: [DONE]\n\n",
        b'data: {"error":{"message":"private-error"}}\n\n',
    ],
)
async def test_http_200_truncated_or_error_stream_is_not_completed(tmp_path, raw):
    log = tmp_path / "truncated.jsonl"
    async with ModelGateway(
        policy=policy(),
        upstream_url="https://provider.example/v1",
        api_key="secret",
        output=log,
        transport=httpx.MockTransport(lambda r: httpx.Response(200, content=raw)),
    ) as relay:
        async with httpx.AsyncClient() as client:
            await client.post(
                relay.base_url + "/chat/completions",
                headers={"Authorization": "Bearer " + relay.token},
                json={"model": "test-model", "stream": True},
            )
    record = json.loads(log.read_text().splitlines()[-1])
    assert record["status"] == 200 and record["completed"] is False
    assert record["error_type"] == "ValueError"
    assert "private-error" not in log.read_text()


def test_campaign_audit_rejects_errors_and_unmatched_or_incomplete_responses(tmp_path):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/rsi"))
    from run_paired_generated import request_audit

    request = {"event": "request", "call": 1, "tools_sent": []}
    response = {"event": "response", "call": 1, "status": 200, "completed": True}
    path = tmp_path / "audit.jsonl"
    for events, passed in [
        ([request, response], True),
        ([request, {**response, "completed": False}], False),
        ([request, {**response, "call": 2}], False),
        ([request, response, {"event": "transport_error", "call": 1}], False),
        ([request, response, {"event": "budget_rejected"}], False),
        ([request, response, response], False),
    ]:
        path.write_text("\n".join(json.dumps(event) for event in events))
        assert request_audit(path, "science_agent", set())["passed"] is passed
