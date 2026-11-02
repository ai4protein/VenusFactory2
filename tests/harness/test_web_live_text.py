"""Real incremental text, with native tool responses withheld until validation."""

import asyncio
import json

import httpx
import pytest

from agent.harness.events import audit_requests
from agent.harness.gateway import EvidenceGateway
from agent.harness.live_text import PublicTextStream
from agent.harness.messages_bridge import MessagesBridge
from agent.harness.responses_gateway import ResponsesGateway
from agent.rsi.model_gateway import GatewayPolicy


def chunk(content=None, *, end=False, tools=None, reasoning=None):
    delta = {"role": "assistant"}
    if content is not None:
        delta["content"] = content
    if tools is not None:
        delta["tool_calls"] = tools
    if reasoning is not None:
        delta["reasoning_content"] = reasoning
    return {
        "id": "fixture1", "model": "unisound-u2-flash", "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": delta, "finish_reason": "stop" if end else None}],
        **({"usage": {"prompt_tokens": 31, "completion_tokens": 10, "total_tokens": 41}} if end else {}),
    }


def sse(value):
    return ("data: " + json.dumps(value, ensure_ascii=False) + "\r\n\r\n").encode()


@pytest.mark.asyncio
async def test_fragmented_utf8_only_emits_public_text_and_keeps_tool_arguments():
    seen = []
    async def observe(call, text):
        seen.append((call, text))
    parser = PublicTextStream(observe, 7)
    raw = b"".join([
        sse(chunk("蛋白", reasoning="private reasoning")),
        sse(chunk(tools=[{"index": 0, "id": "call1", "type": "function", "function": {"name": "inspect", "arguments": '{"x":'}}])),
        sse(chunk("质", tools=[{"index": 0, "function": {"arguments": "1}"}}], end=True)),
        b"data: [DONE]\r\n\r\n",
    ])
    for byte in raw:
        await parser.feed(bytes([byte]))
    assert seen == [(7, "蛋白"), (7, "质")]
    answer = json.loads(parser.completion())
    assert answer["choices"][0]["message"]["content"] == "蛋白质"
    assert answer["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] == '{"x":1}'
    assert "private reasoning" not in json.dumps(answer)


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", ["chat", "messages", "responses"])
@pytest.mark.parametrize("client_stream", [False, True])
async def test_public_text_arrives_before_upstream_finishes(tmp_path, protocol, client_stream):
    received, release = asyncio.Event(), asyncio.Event()
    seen = []
    async def observe(call, text):
        seen.append(text)
        received.set()

    if protocol == "responses":
        from test_responses_gateway import events, stream, policy, response
        rows = events()
        first, rest = stream(rows[:3]), stream(rows[3:])
        gateway_type, route = ResponsesGateway, "/responses"
        config = policy()
        body = {"model": "fixture-model", "input": "fixture", "stream": client_stream}
        expected = "fixture answer"
    else:
        first, rest = sse(chunk("第一段")), sse(chunk("第二段", end=True)) + b"data: [DONE]\n\n"
        expected = "第一段第二段"
        if protocol == "messages":
            from test_messages_bridge import policy, request
            config, body, gateway_type, route = policy(), request(), MessagesBridge, "/messages"
            body["stream"] = client_stream
        else:
            config = GatewayPolicy(model="unisound-u2-flash", mode="science_agent", scientific_tools=set())
            body = {"model": "unisound-u2-flash", "messages": [{"role": "user", "content": "fixture"}], "stream": client_stream}
            gateway_type, route = EvidenceGateway, "/chat/completions"

    class Delayed(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield first
            await release.wait()
            yield rest

    async def upstream(request):
        assert json.loads(request.content)["stream"] is True
        return httpx.Response(200, stream=Delayed())

    log = tmp_path / "requests.jsonl"
    async with gateway_type(policy=config, upstream_url="https://fixture.invalid/v1", api_key="test", output=log, transport=httpx.MockTransport(upstream), on_text=observe) as gateway:
        async with httpx.AsyncClient(trust_env=False) as client:
            pending = asyncio.create_task(client.post(gateway.base_url + route, json=body, headers={"Authorization": "Bearer " + gateway.token}))
            try:
                await asyncio.wait_for(received.wait(), 3)
                assert not pending.done()  # UI already has text; native completion is still pending.
                assert "".join(seen) != expected
            finally:
                release.set()
            result = await pending
            assert result.status_code == 200, result.text
            assert "".join(seen) == expected
            if not client_stream:
                data = result.json()
                if protocol == "chat":
                    assert data["choices"][0]["message"]["content"] == expected
                elif protocol == "messages":
                    assert data["content"][0]["text"] == expected
                else:
                    assert data == response()
            else:
                assert result.headers["content-type"].startswith("text/event-stream")
    assert audit_requests(log)["passed"]


@pytest.mark.asyncio
async def test_partial_text_does_not_turn_a_truncated_response_into_success(tmp_path):
    seen = []
    async def observe(call, text):
        seen.append(text)
    async def upstream(request):
        return httpx.Response(200, content=sse(chunk("unfinished")))
    log = tmp_path / "requests.jsonl"
    async with EvidenceGateway(
        policy=GatewayPolicy(model="unisound-u2-flash", mode="science_agent", scientific_tools=set()),
        upstream_url="https://fixture.invalid/v1", api_key="test", output=log,
        transport=httpx.MockTransport(upstream), on_text=observe,
    ) as gateway:
        async with httpx.AsyncClient(trust_env=False) as client:
            result = await client.post(gateway.base_url + "/chat/completions", json={"model":"unisound-u2-flash","messages":[]}, headers={"Authorization":"Bearer " + gateway.token})
    assert seen == ["unfinished"] and result.status_code == 502
    assert not audit_requests(log)["passed"]
    assert json.loads(log.read_text().splitlines()[-1])["completed"] is False


@pytest.mark.asyncio
async def test_messages_preview_still_requires_the_complete_provider_stream(tmp_path):
    from test_messages_bridge import policy, request
    seen = []
    async def observe(call, text):
        seen.append(text)
    async def upstream(req):
        return httpx.Response(200, content=sse(chunk("partial", end=True)))  # Missing [DONE].
    log = tmp_path / "requests.jsonl"
    async with MessagesBridge(policy=policy(), upstream_url="https://fixture.invalid/v1", api_key="test", output=log, transport=httpx.MockTransport(upstream), on_text=observe) as gateway:
        async with httpx.AsyncClient(trust_env=False) as client:
            result = await client.post(gateway.base_url + "/messages", json=request(), headers={"Authorization": "Bearer " + gateway.token})
    assert seen == ["partial"] and result.status_code == 502
    assert not audit_requests(log)["passed"]
