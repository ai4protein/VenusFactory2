"""A provider quota rejection must not consume the remaining Scientist budget."""
import json

import httpx
import pytest

from agent.harness.gateway import EvidenceGateway
from agent.harness.messages_bridge import MessagesBridge
from agent.harness.responses_gateway import ResponsesGateway, ResponsesPolicy
from agent.rsi.engine.model_gateway import GatewayPolicy


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", ["chat", "messages", "responses"])
@pytest.mark.parametrize("stop", [False, True])
async def test_buffered_quota_rejection_limits_physical_requests(tmp_path, protocol, stop):
    upstream_calls = []

    async def upstream(request):
        upstream_calls.append(request)
        return httpx.Response(429, json={"error": {"message": "Provider rate limit"}})

    policy = GatewayPolicy(model="fixture", mode="science_agent", scientific_tools=set(),
                           max_tokens=4096, thinking="disabled")
    cls = EvidenceGateway
    route = "/chat/completions"
    body = {"model": "fixture", "messages": [{"role": "user", "content": "Public fixture"}]}
    if protocol == "messages":
        cls, route = MessagesBridge, "/messages"
        body["max_tokens"] = 4096
    elif protocol == "responses":
        cls, route = ResponsesGateway, "/responses"
        policy = ResponsesPolicy(model="fixture", scientific_tools=set())
        body = {"model": "fixture", "input": "Public fixture", "stream": False}
    evidence = tmp_path / "requests.jsonl"
    options = {"stop_on_rate_limit": True} if stop else {}
    async with cls(policy=policy, upstream_url="https://fixture.invalid/v1",
                   api_key="fixture", output=evidence,
                   transport=httpx.MockTransport(upstream), **options) as gateway:
        async with httpx.AsyncClient(trust_env=False) as client:
            for _ in range(2):
                response = await client.post(gateway.base_url + route, json=body,
                    headers={"Authorization": "Bearer " + gateway.token})
                assert response.status_code == (400 if stop else 429), response.text
                assert "upstream_http_429" in response.text
        expected_calls = 1 if stop else 2
        assert gateway.calls == len(upstream_calls) == expected_calls
    rows = [json.loads(line) for line in evidence.read_text().splitlines()]
    actual = [row for row in rows if row["event"] == "response"]
    assert len(actual) == expected_calls
    assert all(row["status"] == 429 and not row["completed"] for row in actual)
    assert sum(row["event"] == "upstream_retry_suppressed" for row in rows) == int(stop)
