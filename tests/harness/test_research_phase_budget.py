"""Real loopback requests enforce physical native budgets across every relay route."""
import asyncio
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace

import httpx
import pytest
from aiohttp import web

from agent.harness import research_handoff as handoff
from agent.harness.empty_response_recovery import POLICY as EMPTY_RECOVERY
from agent.harness.gateway import EvidenceGateway
from agent.harness.messages_bridge import MessagesBridge
from agent.harness.responses_bridge import ResponsesChatBridge
from agent.harness.responses_gateway import ResponsesGateway, ResponsesPolicy
from agent.rsi.engine.model_gateway import GatewayPolicy

KINDS = ("chat", "responses", "messages", "responses_bridge")


@pytest.fixture(autouse=True)
def local_queue(tmp_path, monkeypatch):
    from urllib.parse import urlsplit

    from agent.rsi.engine.live_model_gateway import ModelGateway

    def synthetic_loopback_authorization(relay):
        origin = urlsplit(relay.upstream_url)
        assert origin.scheme == "http" and origin.hostname == "127.0.0.1"
        assert relay.policy.model == "fixture-model"
        return {"scope": "synthetic_loopback_test"}

    monkeypatch.setattr(ModelGateway, "authorize_model_access", synthetic_loopback_authorization)
    monkeypatch.setenv("VENUS_USAGE_DB", str(tmp_path / "usage.sqlite"))
    monkeypatch.setenv("VENUS_JOB_QUEUE_ROOT", str(tmp_path / "queue"))
    monkeypatch.setenv("VENUS_WORKER_AUTOSTART", "0")
    for key in ("VENUS_QUEUE_JOB_ID", "VENUS_QUEUE_OWNER", "VENUS_QUEUE_LEASE_TOKEN"):
        monkeypatch.delenv(key, raising=False)


def completed(kind, text="Synthetic response"):
    if kind == "responses":
        return {"id": "resp-fixture", "model": "fixture-model", "status": "completed", "output": [
            {"id": "item1", "type": "message", "role": "assistant", "status": "completed",
             "content": [{"type": "output_text", "text": text, "annotations": []}]}],
                "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}}
    return {"id": "chat-fixture", "object": "chat.completion", "model": "fixture-model", "choices": [
        {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}


@asynccontextmanager
async def relay_fixture(tmp_path, kind, *, limit=5, timeout=2, recovery="none", upstream=None):
    seen = []

    async def provider(request):
        seen.append(await request.json())
        if upstream is not None:
            return await upstream(len(seen))
        await asyncio.sleep(0.01)
        return web.json_response(completed(kind))

    app = web.Application()
    app.router.add_post("/v1/{route:.*}", provider)
    runner = web.AppRunner(app, access_log=None, shutdown_timeout=0.1)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    origin = "http://127.0.0.1:" + str(site._server.sockets[0].getsockname()[1]) + "/v1"
    gateway_class = {"chat": EvidenceGateway, "responses": ResponsesGateway,
                     "messages": MessagesBridge, "responses_bridge": ResponsesChatBridge}[kind]
    if kind == "responses":
        policy = ResponsesPolicy(model="fixture-model", scientific_tools=set(), max_calls=limit)
    else:
        policy = GatewayPolicy(model="fixture-model", mode="science_agent", scientific_tools=set(), max_calls=limit)
        policy.thinking = "disabled"
    payload = {"model": "fixture-model", "stream": False}
    if kind in {"responses", "responses_bridge"}:
        payload.update(input="Review synthetic evidence", store=False)
    else:
        payload["messages"] = [{"role": "user", "content": "Review synthetic evidence"}]
        if kind == "messages":
            payload["max_tokens"] = policy.max_tokens
    route = {"chat": "/chat/completions", "responses": "/responses",
             "messages": "/messages", "responses_bridge": "/responses"}[kind]
    output = tmp_path / "model_requests.jsonl"
    try:
        async with gateway_class(policy=policy, upstream_url=origin, api_key="fixture-upstream-secret",
                                 output=output, request_recovery=recovery, timeout_s=timeout) as relay:
            async with httpx.AsyncClient(trust_env=False, timeout=5) as client:
                async def send(token=None, *, body=None, key_header=False):
                    credential = relay.token if token is None else token
                    headers = {"x-api-key": credential} if key_header else {"Authorization": "Bearer " + credential}
                    if body is None:
                        return await client.post(relay.base_url + route, headers=headers, json=payload)
                    return await client.post(relay.base_url + route,
                                             headers={**headers, "Content-Type": "application/json"}, content=body)
                yield SimpleNamespace(relay=relay, send=send, seen=seen, output=output, payload=payload, policy=policy)
    finally:
        await runner.cleanup()


def error_code(response):
    assert response.status_code == 400, response.text
    return response.json()["error"]["code"]


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", KINDS)
async def test_concurrent_native_calls_leave_exact_review_reserve(tmp_path, kind):
    async with relay_fixture(tmp_path, kind) as case:
        relay = case.relay
        with handoff.native_phase(case.policy, relay=relay, reserve_calls=3) as phase:
            assert phase.relay_token != relay.token
            responses = await asyncio.gather(*(case.send(phase.relay_token) for _ in range(8)))
            assert sum(row.status_code == 200 for row in responses) == 2
            assert all(error_code(row) == "research_phase_budget_exhausted"
                       for row in responses if row.status_code != 200)
            assert relay.calls == 2 and len(case.seen) == 2
        assert error_code(await case.send(phase.relay_token)) == "research_phase_closed"
        for _ in range(3):
            assert (await case.send()).status_code == 200
        assert error_code(await case.send()) == "model_call_budget_exhausted"
        assert relay.calls == 5 and len(case.seen) == 5
        text = case.output.read_text()
        assert relay.token not in text and phase.relay_token not in text and "fixture-upstream-secret" not in text


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", KINDS)
async def test_transport_timeout_and_sdk_retry_share_physical_native_limit(tmp_path, kind):
    release_provider = asyncio.Event()

    async def provider(number):
        if number == 1:
            await release_provider.wait()
        return web.json_response(completed(kind))

    async with relay_fixture(tmp_path, kind, limit=4, timeout=0.25, upstream=provider) as case:
        relay = case.relay
        with handoff.native_phase(relay.policy, relay=relay) as phase:
            first = await case.send(phase.relay_token)
            assert first.status_code == 502
            assert error_code(await case.send(phase.relay_token)) == "research_phase_budget_exhausted"
            assert relay.calls == 1 and len(case.seen) == 1
        # The injected native timeout is not a latency target for the reviewer.
        release_provider.set()
        relay.timeout_s = 2
        assert (await case.send()).status_code == 200
        assert relay.calls == 2 and len(case.seen) == 2
        rows = [json.loads(line) for line in case.output.read_text().splitlines()]
        assert any(row["event"] == "transport_error" and row["call"] == 1 for row in rows)


@pytest.mark.asyncio
@pytest.mark.parametrize("end_phase", [False, True])
async def test_internal_response_recovery_rechecks_original_scope(tmp_path, end_phase):
    async def provider(number):
        value = completed("chat", "" if number == 1 else "Synthetic response")
        if number == 1:
            value["choices"][0]["finish_reason"] = "length"
        return web.json_response(value)

    async with relay_fixture(tmp_path, "chat", limit=4, recovery=EMPTY_RECOVERY, upstream=provider) as case:
        relay = case.relay
        retry_scheduled = asyncio.Event()
        original_record = relay.record

        def record(row):
            original_record(row)
            if row["event"] == "retry_scheduled":
                retry_scheduled.set()

        relay.record = record
        with handoff.native_phase(relay.policy, relay=relay, reserve_calls=3 if not end_phase else 2) as phase:
            pending = asyncio.create_task(case.send(phase.relay_token))
            await asyncio.wait_for(retry_scheduled.wait(), 2)
            if not end_phase:
                response = await pending
        if end_phase:
            response = await pending
        expected = "research_phase_closed" if end_phase else "research_phase_budget_exhausted"
        assert error_code(response) == expected
        assert relay.calls == 1 and len(case.seen) == 1
        assert (await case.send()).status_code == 200
        assert relay.calls == 2 and len(case.seen) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", KINDS)
async def test_authenticated_request_waiting_for_body_cannot_escape_its_closed_phase(tmp_path, monkeypatch, kind):
    captured = asyncio.Event()
    release = asyncio.Event()
    original_scope = handoff.request_scope

    def capture(relay, credential):
        scope = original_scope(relay, credential)
        if scope is not None:
            captured.set()
        return scope

    monkeypatch.setattr(handoff, "request_scope", capture)
    async with relay_fixture(tmp_path, kind) as case:
        relay = case.relay
        raw = json.dumps(case.payload).encode()

        async def delayed_body():
            yield raw[:10]
            await release.wait()
            yield raw[10:]

        with handoff.native_phase(relay.policy, relay=relay) as old:
            pending = asyncio.create_task(case.send(old.relay_token, body=delayed_body()))
            await asyncio.wait_for(captured.wait(), 2)
        try:
            with handoff.native_phase(relay.policy, relay=relay) as new:
                assert old.relay_token != new.relay_token
                release.set()
                assert error_code(await pending) == "research_phase_closed"
                assert relay.calls == 0 and not case.seen
                assert (await case.send(new.relay_token)).status_code == 200
        finally:
            release.set()
            await asyncio.gather(pending, return_exceptions=True)
        assert error_code(await case.send(new.relay_token)) == "research_phase_closed"
        assert (await case.send()).status_code == 200
        assert relay.calls == 2


@pytest.mark.asyncio
async def test_request_waiting_for_resource_slot_keeps_its_original_scope(tmp_path, monkeypatch):
    from backend.web_v2 import job_queue

    entered = asyncio.Event()
    release = asyncio.Event()
    original_slot = job_queue.resource_slot

    @asynccontextmanager
    async def delayed_slot(*args, **kwargs):
        async with original_slot(*args, **kwargs):
            entered.set()
            await release.wait()
            yield

    monkeypatch.setattr(job_queue, "resource_slot", delayed_slot)
    async with relay_fixture(tmp_path, "chat") as case:
        relay = case.relay
        with handoff.native_phase(relay.policy, relay=relay) as phase:
            pending = asyncio.create_task(case.send(phase.relay_token))
            await asyncio.wait_for(entered.wait(), 2)
        with handoff.native_phase(case.policy, relay=relay) as newer:
            newer.save({"summary": "A different phase checkpoint"}, {"checkpointed": True, "artifacts": []})
            release.set()
            assert error_code(await pending) == "research_phase_closed"
            assert relay.calls == 0 and not case.seen
        assert (await case.send()).status_code == 200


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", KINDS)
async def test_checkpoint_gate_and_master_identity_survive_scoped_native_shutdown(tmp_path, kind):
    async with relay_fixture(tmp_path, kind) as case:
        relay = case.relay
        with handoff.native_phase(relay.policy, relay=relay) as phase:
            invalid = await case.send("unknown-phase-token")
            assert invalid.status_code == 401
            phase.save({"summary": "Saved synthetic results"}, {"checkpointed": True, "artifacts": []})
            assert error_code(await case.send(phase.relay_token)) == "research_checkpoint_handoff"
            assert relay.calls == 0 and not case.seen
        assert error_code(await case.send(phase.relay_token)) == "research_phase_closed"
        assert (await case.send()).status_code == 200
        assert relay.calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["messages", "responses_bridge"])
async def test_api_key_header_never_upgrades_native_scope_to_master(tmp_path, kind):
    async with relay_fixture(tmp_path, kind, limit=4) as case:
        relay = case.relay
        with handoff.native_phase(relay.policy, relay=relay) as phase:
            assert (await case.send(phase.relay_token, key_header=True)).status_code == 200
            assert error_code(await case.send(phase.relay_token, key_header=True)) == "research_phase_budget_exhausted"
        assert error_code(await case.send(phase.relay_token, key_header=True)) == "research_phase_closed"
        assert (await case.send(key_header=True)).status_code == 200
        assert relay.calls == 2


@pytest.mark.parametrize("reserve", [-1, True, None, 5])
def test_invalid_reserve_never_creates_or_changes_an_active_phase(reserve):
    policy = SimpleNamespace(max_calls=5)
    relay = SimpleNamespace(policy=policy, calls=0)
    with pytest.raises(ValueError, match="reserve"):
        with handoff.native_phase(policy, relay=relay, reserve_calls=reserve):
            pytest.fail("Invalid scope must not open")
    assert not getattr(policy, "_research_handoff", None)
    assert not hasattr(relay, "_research_request_scopes")


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", KINDS)
async def test_cancelled_native_context_revokes_token_without_charging_a_request(tmp_path, kind):
    async with relay_fixture(tmp_path, kind) as case:
        relay = case.relay
        with pytest.raises(asyncio.CancelledError):
            with handoff.native_phase(case.policy, relay=relay) as phase:
                raise asyncio.CancelledError
        assert not getattr(case.policy, "_research_handoff", None)
        assert error_code(await case.send(phase.relay_token)) == "research_phase_closed"
        assert relay.calls == 0 and not case.seen
        assert (await case.send()).status_code == 200
        assert relay.calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("wrapper_first", [False, True])
async def test_original_and_wrapped_policy_cannot_open_overlapping_phases(tmp_path, wrapper_first):
    async with relay_fixture(tmp_path, "chat") as case:
        relay = case.relay
        first, second = (relay.policy, case.policy) if wrapper_first else (case.policy, relay.policy)
        with handoff.native_phase(first, relay=relay) as phase:
            with pytest.raises(ValueError, match="already active"):
                with handoff.native_phase(second, relay=relay):
                    pytest.fail("The same relay cannot open overlapping native scopes")
            assert (await case.send(phase.relay_token)).status_code == 200
        assert error_code(await case.send(phase.relay_token)) == "research_phase_closed"


@pytest.mark.asyncio
async def test_planning_requests_remain_counted_before_native_reserve(tmp_path):
    async with relay_fixture(tmp_path, "chat", limit=16) as case:
        relay = case.relay
        for _ in range(2):
            assert (await case.send()).status_code == 200
        with handoff.native_phase(case.policy, relay=relay) as phase:
            results = await asyncio.gather(*(case.send(phase.relay_token) for _ in range(12)))
            assert sum(row.status_code == 200 for row in results) == 11
            assert relay.calls == 13 and len(case.seen) == 13
        for _ in range(3):
            assert (await case.send()).status_code == 200
        assert relay.calls == 16 and len(case.seen) == 16
        assert error_code(await case.send()) == "model_call_budget_exhausted"
