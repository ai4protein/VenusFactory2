"""Planning/review calls cannot accidentally become executable tool requests."""
import json
from types import SimpleNamespace

import httpx
import pytest

from agent.harness import autonomous_research as ar
from agent.harness.gateway import inspect_response
from agent.harness.responses_gateway import ResponsesPolicy
from agent.harness.tool_response_names import normalize
from agent.harness.scientist_gateway_policy import ScientistGatewayPolicy


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", ["chat_completions", "responses"])
@pytest.mark.parametrize("role", ["PI", "CB", "SC", "REVIEW"])
async def test_roles_keep_no_tool_choice_through_relay_policy(monkeypatch, tmp_path, protocol, role):
    original = httpx.AsyncClient
    seen = []
    model = "fixture"
    def handler(request):
        payload = json.loads(request.content)
        policy = (ResponsesPolicy(model=model, scientific_tools=set())
                  if protocol == "responses" else
                  ScientistGatewayPolicy(model=model, mode="science_expert", scientific_tools=set()))
        sent, _ = policy.apply(payload)
        assert sent["tool_choice"] == "none"
        assert not sent.get("tools")
        seen.append(sent)
        if protocol == "responses":
            return httpx.Response(200, json={
                "status": "completed", "model": model,
                "output": [{"type": "message", "role": "assistant", "status": "completed",
                            "content": [{"type": "output_text", "text": "Direction checked"}]}],
            })
        chunk = {"model": model, "choices": [{"index": 0,
                 "delta": {"content": "Direction checked"}, "finish_reason": "stop"}]}
        return httpx.Response(200, text="data: "+json.dumps(chunk)+"\n\ndata: [DONE]\n\n",
                              headers={"content-type": "text/event-stream"})
    monkeypatch.setattr(ar.httpx, "AsyncClient",
                        lambda **kw: original(transport=httpx.MockTransport(handler), **kw))
    profile = SimpleNamespace(
        model=SimpleNamespace(protocol=protocol, model=model),
        budget=SimpleNamespace(request_seconds=1, model_calls=2))
    relay = SimpleNamespace(base_url="http://fixture", token="fixture", calls=0)
    assert await ar.AutonomousResearch(None, "fixture", {}).consult(
        role, {}, profile, relay, tmp_path) == "Direction checked"
    assert len(seen) == 1


@pytest.mark.parametrize("streaming", [False, True])
def test_provider_error_finish_is_not_a_complete_response(streaming):
    choice = {"index": 0, "finish_reason": "error"}
    choice["delta" if streaming else "message"] = {"content": "partial"}
    raw = json.dumps({"choices": [choice]}).encode()
    if streaming:
        raw = b"data: " + raw + b"\n\ndata: [DONE]\n\n"
    with pytest.raises(ValueError):
        inspect_response(raw, streaming)


def test_unadvertised_pi_tool_is_still_rejected():
    raw = json.dumps({"choices": [{"message": {"tool_calls": [{
        "type": "function", "id": "call1",
        "function": {"name": "inspect", "arguments": "{}"},
    }]}}]}).encode()
    with pytest.raises(ValueError, match="outside the advertised"):
        normalize(raw, False, [])


def test_historical_policy_unchanged_and_scientist_request_hash_matches_wire():
    import hashlib
    from agent.rsi.engine.model_gateway import GatewayPolicy
    payload = {"model": "fixture", "messages": [], "tool_choice": "none"}
    args = dict(model="fixture", mode="science_agent", scientific_tools=set())
    historical, _ = GatewayPolicy(**args).apply(payload)
    assert "tool_choice" not in historical
    current, evidence = ScientistGatewayPolicy(**args).apply(payload)
    assert current == {**historical, "tool_choice": "none"}
    encoded = json.dumps(current, sort_keys=True, ensure_ascii=False,
                         separators=(",", ":")).encode()
    assert evidence["request_sha256"] == hashlib.sha256(encoded).hexdigest()
    assert evidence["request_bytes"] == len(encoded)
    assert evidence["role_tool_policy"] == "explicit_no_tools.v1"
