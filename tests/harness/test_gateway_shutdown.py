"""Regression for late model evidence after gateway / host timeout shutdown."""

import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest

from agent.harness import Budget, ModelBinding, Profile, Task
from agent.harness.events import audit_requests
from agent.harness.gateway import EvidenceGateway
from agent.harness.host import execute
from agent.rsi.engine.model_gateway import GatewayPolicy, ModelGateway


class PendingUpstream:
    def __init__(self):
        self.started = asyncio.Event()
        self.cleaned = asyncio.Event()

    async def __call__(self, request):
        self.started.set()
        try:
            await asyncio.Future()
        finally:
            # Asynchronous transport cleanup must finish before the terminal snapshot.
            await asyncio.sleep(0.08)
            self.cleaned.set()


@pytest.mark.asyncio
@pytest.mark.parametrize("gateway_type", [ModelGateway, EvidenceGateway])
@pytest.mark.parametrize("cancel_exit", [False, True])
async def test_context_exit_drains_inflight_request_and_its_final_records(tmp_path, gateway_type, cancel_exit):
    upstream = PendingUpstream()
    log = tmp_path / "model_requests.jsonl"
    gateway = gateway_type(
        policy=GatewayPolicy(model="m", mode="science_agent", scientific_tools=set()),
        upstream_url="https://example.org/v1",
        api_key="test-private-key",
        output=log,
        timeout_s=20,
        transport=httpx.MockTransport(upstream),
    )
    await gateway.__aenter__()
    async with httpx.AsyncClient(trust_env=False, timeout=5) as client:
        request = asyncio.create_task(
            client.post(
                gateway.base_url + "/chat/completions",
                json={"model": "m", "messages": []},
                headers={"Authorization": "Bearer " + gateway.token},
            )
        )
        await asyncio.wait_for(upstream.started.wait(), 2)
        closing = asyncio.create_task(gateway.__aexit__(None, None, None))
        if cancel_exit:
            await asyncio.sleep(0.01)
            closing.cancel()
            with pytest.raises(asyncio.CancelledError):
                await closing
        else:
            await closing
        try:
            assert upstream.cleaned.is_set()
            snapshot = log.read_bytes()
            rows = [json.loads(line) for line in snapshot.splitlines()]
            assert [row["event"] for row in rows] == ["request", "transport_error", "response"]
            assert rows[-1]["completed"] is False
            assert rows[-1]["error_type"] == "CancelledError"
            assert [row["call"] for row in rows] == [1, 1, 1]
            assert not audit_requests(log)["passed"]
            await asyncio.sleep(0.12)
            assert log.read_bytes() == snapshot
            assert b"test-private-key" not in snapshot
        finally:
            request.cancel()
            await asyncio.gather(request, return_exceptions=True)


@pytest.mark.asyncio
async def test_host_timeout_snapshot_equals_terminal_log(tmp_path, monkeypatch):
    import agent.harness.gateway as gateway_module

    upstream = PendingUpstream()

    class Gateway(EvidenceGateway):
        def __init__(self, **kwargs):
            super().__init__(**kwargs, transport=httpx.MockTransport(upstream))

    class WaitingAdapter:
        async def run(self, context):
            (context.task.output / "events.jsonl").write_text("")
            async with httpx.AsyncClient(trust_env=False) as client:
                await client.post(
                    context.relay.base_url + "/chat/completions",
                    headers={"Authorization": "Bearer " + context.relay.token},
                    json={"model": "m", "messages": []},
                )
            raise AssertionError("The host should time out first")

    monkeypatch.setattr(gateway_module, "EvidenceGateway", Gateway)
    registry = SimpleNamespace(resolve=lambda *_: (WaitingAdapter(), {"ready": True}))
    profile = Profile(
        "science_agent",
        "test",
        ModelBinding("test", "m", "https://example.org/v1"),
        Budget(wall_seconds=0.3, request_seconds=20),
    )
    task = Task(
        tmp_path / "workspace", tmp_path / "capability", tmp_path / "output", "hello", "", frozenset(), phase="chat"
    )
    task.workspace.mkdir()
    task.capability.mkdir()
    result = await execute(profile, task, {}, registry=registry, credential="test-private-key")
    assert upstream.started.is_set() and upstream.cleaned.is_set()
    assert result["status"] == "timeout"
    assert not result["audit"]["passed"]
    saved = json.loads((task.output / "run.json").read_text())
    before = (task.output / "model_requests.jsonl").read_bytes()
    await asyncio.sleep(0.12)
    assert saved == result
    assert before == (task.output / "model_requests.jsonl").read_bytes()
    assert saved["audit"]["requests"] == audit_requests(task.output / "model_requests.jsonl")
    assert saved["audit"]["requests"]["errors"] == ["transport_error"]
