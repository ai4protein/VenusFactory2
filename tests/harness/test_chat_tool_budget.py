import json
import httpx
import pytest
from agent.harness.chat_tool_budget import ChatToolBudget
from agent.harness.gateway import EvidenceGateway
from agent.harness.live_text import PublicTextStream
from agent.harness.events import audit_requests
from agent.rsi.engine.model_gateway import GatewayPolicy

def calls(n):
    return [{"id": f"call_{i}", "type": "function", "function": {
        "name": "mcp__venusfactory__search_tools", "arguments": '{"query":"protein"}'
    }} for i in range(n)]

def answer(tool_calls=None):
    return {"id": "fixture", "model": "fixture", "object": "chat.completion",
            "choices": [{"index": 0, "message": {"role": "assistant",
                "content": None if tool_calls else "Verified platform capabilities.",
                **({"tool_calls": tool_calls} if tool_calls else {})},
                "finish_reason": "tool_calls" if tool_calls else "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}

def stream(obj):
    message = obj["choices"][0]["message"]
    rows = []
    for i, call in enumerate(message.get("tool_calls", [])):
        for first, part in enumerate((call["function"]["arguments"][:8], call["function"]["arguments"][8:])):
            rows.append({"choices": [{"index": 0, "delta": {"tool_calls": [{
                "index": i, **({"id": call["id"], "type": "function"} if first == 0 else {}),
                "function": {**({"name": call["function"]["name"]} if first == 0 else {}),
                             "arguments": part}}]}, "finish_reason": None}]})
    rows.append({**obj, "choices": [{"index": 0, "delta": {"content": message["content"]},
                                  "finish_reason": obj["choices"][0]["finish_reason"]}]})
    return b"".join(b"data: " + json.dumps(r).encode() + b"\n\n" for r in rows) + b"data: [DONE]\n\n"

@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_parallel_tool_batch_fits_limit_and_preserves_arguments(streaming):
    budget = ChatToolBudget(2)
    raw = stream(answer(calls(5))) if streaming else json.dumps(answer(calls(5))).encode()
    delivered, info = budget.deliver(raw, streaming)
    if streaming:
        async def noop(*args): pass
        parser = PublicTextStream(noop, 1)
        await parser.feed(delivered)
        delivered = parser.completion()
    assert json.loads(delivered)["choices"][0]["message"]["tool_calls"] == calls(2)
    assert info == {"admitted": 2, "not_dispatched": 3, "used": 2}
    payload, status = budget.prepare({"messages": [], "tools": ["fixture"], "tool_choice": "auto",
                                     "parallel_tool_calls": True})
    assert not payload.get("tools") and "tool_choice" not in payload
    assert status["answer_only"]
    assert "unperformed" in payload["messages"][0]["content"]

@pytest.mark.asyncio
@pytest.mark.parametrize("client_stream", [False, True])
@pytest.mark.parametrize("last_request", [False, True])
async def test_gateway_stops_discovery_and_produces_final_answer(tmp_path, client_stream, last_request):
    received = []
    async def upstream(request):
        payload = json.loads(request.content)
        received.append(payload)
        obj = answer(calls(13)) if len(received) == 1 else answer()
        return httpx.Response(200, content=stream(obj) if payload.get("stream") else json.dumps(obj).encode())
    async def observe(*args): pass
    policy = GatewayPolicy(model="fixture", mode="science_agent", scientific_tools={"search_tools"}, max_calls=2 if last_request else 24)
    log = tmp_path / "requests.jsonl"
    async with EvidenceGateway(policy=policy, upstream_url="https://fixture.invalid/v1", api_key="test",
                               output=log, transport=httpx.MockTransport(upstream),
                               tool_budget=20 if last_request else 12, on_text=observe) as gateway:
        body = {"model": "fixture", "messages": [{"role": "system", "content": "Host instructions"},
                {"role": "user", "content": "What can you do?"}], "stream": client_stream,
                "tools": [{"type": "function", "function": {"name": "mcp__venusfactory__search_tools",
                           "parameters": {"type": "object", "properties": {}}}}]}
        async with httpx.AsyncClient(trust_env=False) as client:
            for _ in range(2):
                response = await client.post(gateway.base_url + "/chat/completions", json=body,
                                             headers={"Authorization": "Bearer " + gateway.token})
                assert response.status_code == 200, response.text
            assert "Verified platform capabilities." in response.text
    assert received[0]["tools"] and not received[1].get("tools")
    assert received[1]["messages"][-1] == body["messages"][-1]
    assert "final answer now" in received[1]["messages"][0]["content"]
    assert audit_requests(log)["passed"]
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    budget_rows = [r for r in rows if r["event"] == "chat_tool_budget"]
    assert budget_rows[0]["not_dispatched"] == (0 if last_request else 1)
    assert budget_rows[-1]["used"] <= (20 if last_request else 12)
