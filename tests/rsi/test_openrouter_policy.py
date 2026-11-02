"""Free-only routing is tested against an in-memory upstream, with no paid calls."""

import copy
import hashlib
import json

import httpx
import pytest

from agent.harness.gateway import EvidenceGateway
from agent.rsi import model_access
from agent.rsi.live_model_gateway import ModelGateway
from agent.rsi.model_gateway import GatewayPolicy
from agent.rsi.openrouter_policy import (
    BASE_URL,
    FREE_ROUTER,
    PROVIDER,
    enforce_free_request,
    free_catalog,
    free_model_id,
    verify_free_response,
)

MODEL = "cohere/north-mini-code:free"


@pytest.fixture
def access(tmp_path, monkeypatch):
    path = tmp_path / "policy.json"
    policy = json.loads(model_access.POLICY_PATH.read_bytes())
    if policy.get("schema") == 2:
        catalog = model_access.POLICY_PATH.parent / policy["catalog"]
        (tmp_path / "catalog.json").write_bytes(catalog.read_bytes())
        policy["catalog"] = "catalog.json"
    path.write_text(json.dumps(policy))
    monkeypatch.setattr(model_access, "POLICY_PATH", path)
    return path


def example(**updates):
    return {"model": MODEL, "messages": [{"role": "user", "content": "Reply OK"}], **updates}


@pytest.mark.parametrize(
    "value", ["cohere/north-mini-code", "openrouter/auto", MODEL + ":online", "other/free", "x:free", None]
)
def test_paid_or_ambiguous_ids_are_not_free(value):
    assert not free_model_id(value)


def test_catalog_requires_every_price_to_be_zero_and_function_support():
    base = {"id": MODEL, "pricing": {"prompt": "0", "completion": "0"}, "supported_parameters": ["tools"]}
    assert free_catalog({"data": [base]}) == [base]
    for pricing in [
        {"prompt": "0"},
        {"prompt": "0", "completion": "0", "request": "0.001"},
        {"prompt": "NaN", "completion": "0"},
        {"prompt": "-1", "completion": "0"},
        {"prompt": True, "completion": "0"},
    ]:
        assert free_catalog({"data": [{**base, "pricing": pricing}]}) == []
    assert free_catalog({"data": [{**base, "supported_parameters": []}]}) == []
    with pytest.raises(ValueError, match="Duplicate"):
        free_catalog({"data": [base, base]})


def test_request_rewrites_thinking_and_binds_actual_zero_price_payload():
    original = example(thinking={"type": "disabled"})
    original_copy = copy.deepcopy(original)
    payload, witness = enforce_free_request(original, {"request_sha256": "obsolete"})
    assert original == original_copy
    assert payload["provider"] == PROVIDER
    assert payload["reasoning"] == {"enabled": False} and "thinking" not in payload
    assert payload["usage"] == {"include": True}
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()
    assert witness["request_sha256"] == hashlib.sha256(encoded).hexdigest()
    assert witness["request_bytes"] == len(encoded)


@pytest.mark.parametrize(
    "extension",
    [
        {"models": ["paid/model"]},
        {"plugins": [{"id": "web"}]},
        {"route": "fallback"},
        {"transforms": ["middle-out"]},
        {"service_tier": "priority"},
        {"provider": {"allow_fallbacks": True}},
        {"provider": {"max_price": {"prompt": 1}}},
        {"modalities": ["audio"]},
        {
            "messages": [
                {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "https://example.org/a.png"}}]}
            ]
        },
    ],
)
def test_cost_extensions_and_multimodal_requests_are_rejected(extension):
    with pytest.raises(ValueError):
        enforce_free_request(example(**extension), {})


def body(*, model=MODEL.removesuffix(":free"), cost=0):
    return {
        "id": "synthetic",
        "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "OK"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4, "cost": cost},
    }


def test_free_router_keeps_actual_returned_model_identity():
    result = verify_free_response(json.dumps(body(model="actual/free-model")).encode(), False, FREE_ROUTER)
    assert result["reported_models"] == ["actual/free-model"] and result["billed_cost"] == 0


class Upstream(httpx.AsyncBaseTransport):
    def __init__(self, response=None):
        self.requests = []
        self.response = response or body()

    async def handle_async_request(self, request):
        payload = json.loads(request.content)
        self.requests.append(payload)
        if payload.get("stream"):
            raw = ("data: " + json.dumps(self.response) + "\n\ndata: [DONE]\n\n").encode()
            return httpx.Response(200, content=raw, headers={"Content-Type": "text/event-stream"})
        return httpx.Response(200, json=self.response)


def gateway(tmp_path, transport, model=MODEL, kind=EvidenceGateway):
    return kind(
        policy=GatewayPolicy(
            model=model, mode="science_agent", scientific_tools=set(), max_tokens=128, thinking="disabled"
        ),
        upstream_url=BASE_URL,
        api_key="not-a-real-key",
        output=tmp_path / "requests.jsonl",
        transport=transport,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_zero_cost_chat_and_stream_use_actual_gateway(access, tmp_path, stream):
    upstream = Upstream()
    async with gateway(tmp_path, upstream) as relay:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                relay.base_url + "/chat/completions",
                headers={"Authorization": "Bearer " + relay.token},
                json=example(stream=stream),
            )
        assert response.status_code == 200
    assert upstream.requests[0]["provider"] == PROVIDER
    assert upstream.requests[0]["max_tokens"] == 128
    records = [json.loads(line) for line in (tmp_path / "requests.jsonl").read_text().splitlines()]
    request = next(row for row in records if row["event"] == "request")
    response = next(row for row in records if row["event"] == "response")
    assert request["free_route"]["provider"] == PROVIDER
    assert response["completed"] and response["provider_evidence"]["billed_cost"] == 0
    assert "not-a-real-key" not in (tmp_path / "requests.jsonl").read_text()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response", [body(cost=0.01), body(cost=None), body(cost="NaN"), body(model="unexpected/model")]
)
async def test_cost_or_identity_failure_blocks_later_requests(access, tmp_path, response):
    upstream = Upstream(response)
    async with gateway(tmp_path, upstream) as relay:
        async with httpx.AsyncClient() as client:

            async def send():
                return await client.post(
                    relay.base_url + "/chat/completions",
                    headers={"Authorization": "Bearer " + relay.token},
                    json=example(),
                )

            assert (await send()).status_code == 502
            assert (await send()).status_code == 400
    assert len(upstream.requests) == 1
    events = [json.loads(line) for line in (tmp_path / "requests.jsonl").read_text().splitlines()]
    assert any(row["event"] == "free_pricing_rejected" for row in events)
    assert not next(row for row in events if row["event"] == "response")["completed"]


@pytest.mark.asyncio
async def test_paid_and_legacy_streaming_routes_fail_before_upstream(access, tmp_path):
    for number, (model, kind) in enumerate([("cohere/north-mini-code", EvidenceGateway), (MODEL, ModelGateway)]):
        transport = Upstream()
        relay = gateway(tmp_path / str(number), transport, model=model, kind=kind)
        with pytest.raises(model_access.ModelAccessDenied):
            async with relay:
                pytest.fail("Unsupported route opened")
        assert not transport.requests and relay.client is None


def test_openrouter_paid_entry_cannot_be_enabled_by_allowlist_edit(access):
    value = json.loads(access.read_text())
    if value.get("schema") == 2:
        # Exercise the supported legacy allowlist as well as the current
        # server-catalog policy used by the gateway tests above.
        from agent.harness.free_catalog import free_routes
        routes, _ = free_routes(access.parent / value["catalog"])
        value = {"schema": 1, "policy_id": "test-edited-policy",
                 "pricing_source": "test", "allowed": routes}
    value["allowed"].append(
        {
            "model": "cohere/north-mini-code",
            "base_url": BASE_URL,
            "protocol": "chat_completions",
            "max_output_tokens": 8192,
        }
    )
    access.write_text(json.dumps(value))
    with pytest.raises(model_access.ModelAccessDenied):
        model_access.authorize_model(
            model="cohere/north-mini-code", base_url=BASE_URL, protocol="chat_completions", output_tokens=128
        )
