"""A single transient recovery preserves payloads, physical costs and deadlines."""

import asyncio
import hashlib
import json

import httpx
import pytest

from agent.harness import Budget, ModelBinding, Profile
from agent.harness.contracts import digest
from agent.harness.events import audit_requests
from agent.harness.gateway import EvidenceGateway
from agent.harness.request_recovery import POLICY, REJECTION_SHA256
from agent.rsi.model_gateway import GatewayPolicy

ERROR = b'{"error":{"message":"The request is invalid: the request was rejected by an internal MaaS component. Please check the request body, required fields, and request format.","type":"invalid_request_error","param":"","code":"400001"}}'
OK = {
    "choices": [{"message": {"role": "assistant", "content": "OK"}, "finish_reason": "stop"}],
    "usage": {"total_tokens": 4},
}


def bound_profile(tmp_path, policy, limit):
    profile = Profile(
        "science_agent",
        "langgraph",
        ModelBinding("dmx", "glm-5.1", "https://www.dmxapi.cn/v1"),
        Budget(model_calls=limit),
        request_recovery=policy,
    )
    lock = {"profile": profile.public()}
    lock["sha256"] = digest(lock)
    (tmp_path / "profile.lock.json").write_text(json.dumps(lock))
    return profile


async def fixture(tmp_path, *, policy=POLICY, limit=4, bad=None, always_bad=False, timeout=2):
    bound_profile(tmp_path, policy, limit)
    bodies = []

    def upstream(request):
        bodies.append(request.content)
        if len(bodies) == 1 or always_bad:
            return httpx.Response(400, content=ERROR if bad is None else bad)
        return httpx.Response(200, json=OK)

    log = tmp_path / "model_requests.jsonl"
    async with EvidenceGateway(
        policy=GatewayPolicy(model="glm-5.1", mode="science_agent", scientific_tools=set(), max_calls=limit),
        upstream_url="https://www.dmxapi.cn/v1",
        api_key="fixture-key",
        output=log,
        request_recovery=policy,
        timeout_s=timeout,
        transport=httpx.MockTransport(upstream),
    ) as relay:
        async with httpx.AsyncClient(trust_env=False) as client:
            response = await client.post(
                relay.base_url + "/chat/completions",
                headers={"Authorization": "Bearer " + relay.token},
                json={"model": "glm-5.1", "messages": [{"role": "user", "content": "synthetic fixture"}]},
            )
    return response, bodies, log


@pytest.mark.asyncio
async def test_one_recovery_is_charged_and_not_called_uninterrupted(tmp_path):
    assert len(ERROR) == 229 and hashlib.sha256(ERROR).hexdigest() == REJECTION_SHA256
    response, bodies, log = await fixture(tmp_path)
    assert response.status_code == 200 and len(bodies) == 2 and bodies[0] == bodies[1]
    result = audit_requests(log)
    assert result["passed"] and not result["uninterrupted_passed"]
    assert result["model_calls"] == 2 and result["logical_calls"] == 1 and result["failed_attempts"] == 1
    assert result["recovered_calls"] == [{"logical_call": 1, "failed_attempt": 1, "successful_attempt": 2}]
    rows = [json.loads(s) for s in log.read_text().splitlines()]
    assert [x["status"] for x in rows if x["event"] == "response"] == [400, 200]
    assert "fixture-key" not in log.read_text()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "options,expected,count",
    [
        ({"policy": "none"}, 400, 1),
        ({"policy": "empty_length_once_v1"}, 400, 1),
        ({"always_bad": True}, 400, 2),
        ({"bad": b'{"error":"invalid credential"}'}, 400, 1),
        ({"limit": 1}, 429, 1),
        ({"timeout": 0.05}, 502, 1),
    ],
)
async def test_recovery_never_bypasses_policy_error_budget_or_deadline(tmp_path, options, expected, count):
    response, bodies, log = await fixture(tmp_path, **options)
    assert response.status_code == expected and len(bodies) == count
    assert not audit_requests(log)["passed"]


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["payload", "failure", "grant", "policy", "ancestry", "schedule"])
async def test_forged_recovery_evidence_fails_audit(tmp_path, change):
    _, _, log = await fixture(tmp_path)
    rows = [json.loads(s) for s in log.read_text().splitlines()]
    if change in {"grant", "policy"}:
        path = tmp_path / "profile.lock.json"
        lock = json.loads(path.read_text())
        if change == "grant":
            lock["profile"]["budget"]["model_calls"] = 1
        else:
            lock["profile"].pop("request_recovery")
        lock["sha256"] = digest({k: v for k, v in lock.items() if k != "sha256"})
        path.write_text(json.dumps(lock))
    else:
        if change == "payload":
            next(x for x in rows if x["event"] == "request" and x["call"] == 2)["request_sha256"] = "changed"
        elif change == "failure":
            next(x for x in rows if x["event"] == "upstream_rejection")["body_prefix_sha256"] = "changed"
        elif change == "ancestry":
            next(x for x in rows if x["event"] == "request" and x["call"] == 2)["retry_of"] = 9
        else:
            rows = [x for x in rows if x["event"] != "retry_scheduled"]
        log.write_text("".join(json.dumps(x) + "\n" for x in rows))
    assert not audit_requests(log)["passed"]


def test_legacy_profile_serialization_is_unchanged_and_policy_is_endpoint_bound():
    profile = Profile("science_agent", "langgraph", ModelBinding("test", "m", "https://example.org/v1"))
    assert "request_recovery" not in profile.public()
    with pytest.raises(ValueError, match="endpoint"):
        Profile("science_agent", "langgraph", profile.model, request_recovery=POLICY)


@pytest.mark.asyncio
async def test_concurrent_requests_share_physical_attempt_budget(tmp_path):
    bound_profile(tmp_path, POLICY, 3)
    calls = []

    def upstream(request):
        calls.append(request.content)
        return httpx.Response(400, content=ERROR) if len(calls) <= 2 else httpx.Response(200, json=OK)

    log = tmp_path / "model_requests.jsonl"
    async with EvidenceGateway(
        policy=GatewayPolicy(model="glm-5.1", mode="science_agent", scientific_tools=set(), max_calls=3),
        upstream_url="https://www.dmxapi.cn/v1",
        api_key="fixture",
        output=log,
        request_recovery=POLICY,
        transport=httpx.MockTransport(upstream),
    ) as relay:
        async with httpx.AsyncClient(trust_env=False) as client:

            async def call():
                return await client.post(
                    relay.base_url + "/chat/completions",
                    headers={"Authorization": "Bearer " + relay.token},
                    json={"model": "glm-5.1", "messages": []},
                )

            results = await asyncio.gather(call(), call())
    assert sorted(r.status_code for r in results) == [200, 429]
    assert len(calls) == 3 and audit_requests(log)["model_calls"] == 3
    assert not audit_requests(log)["passed"]


@pytest.mark.asyncio
async def test_mixed_live_pattern_lists_only_successfully_recovered_calls(tmp_path):
    bound_profile(tmp_path, POLICY, 6)
    statuses = iter([400, 400, 400, 200, 200])

    def upstream(request):
        status = next(statuses)
        return httpx.Response(status, content=ERROR) if status == 400 else httpx.Response(200, json=OK)

    log = tmp_path / "model_requests.jsonl"
    async with EvidenceGateway(
        policy=GatewayPolicy(model="glm-5.1", mode="science_agent", scientific_tools=set(), max_calls=6),
        upstream_url="https://www.dmxapi.cn/v1",
        api_key="fixture",
        output=log,
        request_recovery=POLICY,
        transport=httpx.MockTransport(upstream),
    ) as relay:
        async with httpx.AsyncClient(trust_env=False) as client:
            statuses_seen = []
            for _ in range(3):
                response = await client.post(
                    relay.base_url + "/chat/completions",
                    headers={"Authorization": "Bearer " + relay.token},
                    json={"model": "glm-5.1", "messages": []},
                )
                statuses_seen.append(response.status_code)
    assert statuses_seen == [400, 200, 200]
    result = audit_requests(log)
    assert not result["passed"] and not result["uninterrupted_passed"]
    assert result["recovered_calls"] == [{"logical_call": 3, "failed_attempt": 3, "successful_attempt": 4}]
    assert result["model_calls"] == 5 and result["failed_attempts"] == 3
