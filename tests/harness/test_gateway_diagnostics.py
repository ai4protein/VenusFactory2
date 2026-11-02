"""Provider failures retain bounded signals without echoed prompts or keys."""

import json

import httpx
import pytest

from agent.harness.gateway import EvidenceGateway
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
