"""Authorization must stop paid requests before any upstream transport call."""

import json
from dataclasses import replace

import httpx
import pytest

from agent.harness.contracts import ModelBinding, Profile
from agent.harness.gateway import EvidenceGateway
from agent.harness.responses_gateway import ResponsesGateway, ResponsesPolicy
from agent.rsi import model_access
from agent.rsi.live_model_gateway import ModelGateway
from agent.rsi.model_gateway import GatewayPolicy


@pytest.fixture
def access_file(tmp_path, monkeypatch):
    path = tmp_path / "model_access.json"
    # Copy both parts of the server configuration so the relative route path
    # remains valid after moving the policy into an isolated test directory.
    policy = json.loads(model_access.POLICY_PATH.read_bytes())
    if policy.get("schema") == 2:
        catalog = model_access.POLICY_PATH.parent / policy["catalog"]
        (tmp_path / "catalog.json").write_bytes(catalog.read_bytes())
        policy["catalog"] = "catalog.json"
    path.write_text(json.dumps(policy))
    monkeypatch.setattr(model_access, "POLICY_PATH", path)
    return path


def admit(model="glm-4-flash", **overrides):
    args = {
        "model": model,
        "base_url": "https://www.dmxapi.cn/v1",
        "protocol": "chat_completions",
        "output_tokens": 4096,
    }
    return model_access.authorize_model(**(args | overrides))


@pytest.mark.parametrize("model", ["glm-4-flash", "unisound-u2-flash"])
def test_only_explicit_free_text_bindings_are_authorized(access_file, model):
    assert admit(model)["policy_sha256"]
    for bad in [
        {"model": "glm-5.2"},
        {"model": "mimo-v2.5-tts"},
        {"model": model.upper()},
        {"base_url": "https://www.dmxapi.cn.evil/v1"},
        {"base_url": "http://www.dmxapi.cn/v1"},
        {"base_url": "https://www.dmxapi.cn/v1?redirect=1"},
        {"base_url": "https://secret@www.dmxapi.cn/v1"},
        {"protocol": "responses"},
        {"output_tokens": True},
        {"output_tokens": 0},
    ]:
        with pytest.raises(model_access.ModelAccessDenied):
            admit(**({"model": model} | bad))


def test_limits_and_missing_or_malformed_policy_fail_closed(access_file):
    for count in [4097, 131072]:
        with pytest.raises(model_access.ModelAccessDenied):
            admit(output_tokens=count)
    assert admit("unisound-u2-flash", output_tokens=131072)
    original = access_file.read_bytes()
    for value in ["invalid", "[]", "null", '{"schema":1}']:
        access_file.write_text(value)
        with pytest.raises(model_access.ModelAccessDenied):
            admit()
    access_file.write_bytes(original)
    access_file.unlink()
    with pytest.raises(model_access.ModelAccessDenied):
        admit()


class RecordingTransport(httpx.AsyncBaseTransport):
    def __init__(self):
        self.requests = []

    async def handle_async_request(self, request):
        self.requests.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"finish_reason": "stop"}]})


def relay(kind, tmp_path, transport, model="glm-4-flash", output=4096):
    policy = (
        ResponsesPolicy(model=model, scientific_tools=set(), max_output_tokens=output)
        if kind is ResponsesGateway
        else GatewayPolicy(model=model, mode="science_agent", scientific_tools=set(), max_tokens=output)
    )
    return kind(
        policy=policy,
        upstream_url="https://www.dmxapi.cn/v1",
        api_key="test-secret",
        output=tmp_path / "requests.jsonl",
        transport=transport,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", [ModelGateway, EvidenceGateway, ResponsesGateway])
@pytest.mark.parametrize("transport", [None, RecordingTransport])
async def test_paid_configuration_rejected_before_client_creation(access_file, tmp_path, kind, transport):
    upstream = transport() if transport else None
    gateway = relay(kind, tmp_path, upstream, model="glm-5.2")
    with pytest.raises(model_access.ModelAccessDenied):
        async with gateway:
            pytest.fail("Paid configuration entered the live gateway")
    assert gateway.client is None and gateway.calls == 0
    assert upstream is None or upstream.requests == []
    assert "test-secret" not in (tmp_path / "requests.jsonl").read_text()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", [ModelGateway, EvidenceGateway])
@pytest.mark.parametrize("model", ["glm-4-flash", "unisound-u2-flash"])
async def test_free_request_retains_exact_identity_and_policy_evidence(access_file, tmp_path, kind, model):
    upstream = RecordingTransport()
    async with relay(kind, tmp_path, upstream, model=model) as gateway:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                gateway.base_url + "/chat/completions",
                headers={"Authorization": "Bearer " + gateway.token},
                json={"model": model, "messages": [], "max_tokens": 99999},
            )
        assert response.status_code == 200 and gateway.calls == 1
    assert len(upstream.requests) == 1
    assert upstream.requests[0]["model"] == model and upstream.requests[0]["max_tokens"] == 4096
    records = [json.loads(line) for line in (tmp_path / "requests.jsonl").read_text().splitlines()]
    request = next(row for row in records if row["event"] == "request")
    assert request["model_access"]["policy_sha256"] == admit(model)["policy_sha256"]


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", [ModelGateway, EvidenceGateway])
async def test_policy_is_rechecked_before_each_request(access_file, tmp_path, kind):
    upstream = RecordingTransport()
    async with relay(kind, tmp_path, upstream) as gateway:
        access_file.unlink()
        async with httpx.AsyncClient() as client:
            response = await client.post(
                gateway.base_url + "/chat/completions",
                headers={"Authorization": "Bearer " + gateway.token},
                json={"model": "glm-4-flash", "messages": []},
            )
        assert response.status_code == 400 and gateway.calls == 0
    assert upstream.requests == []


def test_historical_profiles_still_deserialize_without_authorizing_execution(access_file):
    historical = Profile(
        mode="science_expert",
        harness="expert-kimi-code",
        model=ModelBinding(provider="dmx", model="glm-5.2", base_url="https://www.dmxapi.cn/v1"),
    )
    assert Profile.from_dict(historical.public()) == historical
    with pytest.raises(model_access.ModelAccessDenied):
        model_access.authorize_profile(historical)
    free = replace(
        historical,
        model=replace(historical.model, model="glm-4-flash"),
        budget=replace(historical.budget, output_tokens=4096, thinking=None),
    )
    assert model_access.authorize_profile(free)
