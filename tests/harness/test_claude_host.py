import asyncio
import json
import sys
from dataclasses import replace
from types import SimpleNamespace

import httpx
import pytest

from agent.harness import Budget, ModelBinding, Profile, Task
from agent.harness import claude_adapter as adapter
from agent.harness.host import RuntimeContext
from agent.harness.messages_bridge import MessagesBridge
from agent.rsi.engine.model_gateway import GatewayPolicy


def response():
    return {
        "id": "test_1",
        "model": "fixture",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "Observed"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
    }


@pytest.mark.asyncio
async def test_expert_chat_and_native_messages_share_physical_budget(tmp_path):
    observed = []

    async def upstream(request):
        observed.append(json.loads(request.content))
        return httpx.Response(200, json=response())

    policy = GatewayPolicy(
        model="fixture",
        mode="science_agent",
        scientific_tools={"protein_probe"},
        max_calls=2,
        max_tokens=8192,
        thinking="disabled",
    )
    async with MessagesBridge(
        policy=policy,
        upstream_url="https://fixture.invalid/v1",
        api_key="fixture",
        output=tmp_path / "requests.jsonl",
        transport=httpx.MockTransport(upstream),
    ) as gateway:
        async with httpx.AsyncClient(trust_env=False) as client:
            chat = {"model": "fixture", "messages": [{"role": "user", "content": "Plan"}]}
            native = {**chat, "max_tokens": 8192, "stream": True}
            results = await asyncio.gather(
                client.post(
                    gateway.base_url + "/chat/completions",
                    json=chat,
                    headers={"Authorization": "Bearer " + gateway.token},
                ),
                client.post(gateway.base_url + "/messages", json=native, headers={"x-api-key": gateway.token}),
            )
            assert [r.status_code for r in results] == [200, 200]
            assert "event: message_stop" in results[1].text
            denied = await client.post(
                gateway.base_url + "/messages", json=native, headers={"x-api-key": gateway.token}
            )
            assert denied.status_code == 400
            denied = await client.post(
                gateway.base_url + "/chat/completions", json=chat, headers={"Authorization": "Bearer " + gateway.token}
            )
            assert denied.status_code == 400
            assert gateway.calls == len(observed) == 2
    events = [json.loads(line) for line in (tmp_path / "requests.jsonl").read_text().splitlines()]
    assert sorted(row["call"] for row in events if row["event"] == "request") == [1, 2]
    assert {row["transport"] for row in events if row["event"] == "request"} == {
        "buffered_complete_response_v1",
        "messages_to_free_chat_buffered_v1",
    }


@pytest.mark.asyncio
async def test_both_routes_reject_invalid_credentials_without_an_upstream_call(tmp_path):
    async def upstream(request):
        raise AssertionError("No request may reach upstream")

    policy = GatewayPolicy(
        model="fixture", mode="science_agent", scientific_tools=set(), max_tokens=8192, thinking="disabled"
    )
    async with MessagesBridge(
        policy=policy,
        upstream_url="https://fixture.invalid/v1",
        api_key="fixture",
        output=tmp_path / "requests.jsonl",
        transport=httpx.MockTransport(upstream),
    ) as gateway:
        async with httpx.AsyncClient(trust_env=False) as client:
            for path in ["/messages", "/chat/completions"]:
                reply = await client.post(
                    gateway.base_url + path, json={}, headers={"x-api-key": "wrong", "Authorization": "Bearer wrong"}
                )
                assert reply.status_code == 401 and gateway.calls == 0


def profile():
    return Profile(
        "science_agent",
        "claude-code",
        ModelBinding("fixture", "fixture", "https://fixture.invalid/v1"),
        Budget(
            wall_seconds=30, request_seconds=10, model_calls=2, tool_calls=1, output_tokens=8192, thinking="disabled"
        ),
    )


@pytest.mark.parametrize(
    "change",
    [
        {"thinking": "enabled"},
        {"thinking": None},
        {"reasoning_effort": "low"},
        {"temperature": None},
    ],
)
def test_unsupported_native_semantics_rejected(change):
    p = profile()
    with pytest.raises(ValueError):
        adapter.validate_profile(replace(p, budget=replace(p.budget, **change)))


def context(tmp_path):
    work, cap, out = [tmp_path / name for name in ["work", "cap", "out"]]
    for p in [work, cap, out]:
        p.mkdir()
    task = Task(work, cap, out, "Measure the protein", "Use tools", frozenset({"protein_probe"}), phase="probe")
    return RuntimeContext(
        profile(),
        task,
        SimpleNamespace(base_url="http://127.0.0.1:9/v1", token="ephemeral-fixture"),
        {"python_root": sys.prefix, "claude_bin": "/fixture/claude"},
    )


class PreparedFixture:
    def __init__(self, **kwargs):
        self.stopped = False

    async def start(self):
        return {"stage": "fixture_loaded"}

    def connection(self):
        return {"command": "unused", "args": []}

    async def stop(self):
        self.stopped = True


@pytest.mark.asyncio
async def test_cancel_preserves_native_checkpoint_and_stops_process(tmp_path, monkeypatch):
    ctx = context(tmp_path)
    monkeypatch.setattr(adapter, "PreparedMcp", PreparedFixture)
    script = "import json,time;print(json.dumps({'type':'system','subtype':'init','session_id':'s1'}),flush=True);time.sleep(90)"
    monkeypatch.setattr(adapter, "sandbox_command", lambda *args: [sys.executable, "-u", "-c", script])
    task = asyncio.create_task(adapter.ClaudeAdapter(ctx.runtimes).run(ctx))
    try:
        async with asyncio.timeout(20):
            while (
                not (ctx.task.output / "native_events.jsonl").exists()
                or not (ctx.task.output / "native_events.jsonl").stat().st_size
            ):
                if task.done():
                    task.result()
                await asyncio.sleep(0.02)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    checkpoint = json.loads((ctx.task.output / "native_checkpoint.json").read_text())
    assert (
        checkpoint["status"] == "interrupted"
        and checkpoint["sandbox_stopped"]
        and checkpoint["capability_worker_stopped"]
    )
    assert checkpoint["session_id"] == "s1" and checkpoint["tool_calls"] == 0


@pytest.mark.asyncio
async def test_truncated_native_stream_is_not_a_success(tmp_path, monkeypatch):
    ctx = context(tmp_path)
    monkeypatch.setattr(adapter, "PreparedMcp", PreparedFixture)
    script = 'import sys;print(\'{"type":"system","subtype":"init","session_id":"s1"}\');sys.stdout.write(\'{"type":"assistant"\')'
    monkeypatch.setattr(adapter, "sandbox_command", lambda *args: [sys.executable, "-u", "-c", script])
    result = await adapter.ClaudeAdapter(ctx.runtimes).run(ctx)
    assert result["status"] == "protocol_failed" and result["error_type"] == "JSONDecodeError"
    assert result["sandbox_stopped"] and result["capability_worker_stopped"]


@pytest.mark.asyncio
async def test_eof_without_native_terminal_record_is_not_success(tmp_path, monkeypatch):
    ctx = context(tmp_path)
    monkeypatch.setattr(adapter, "PreparedMcp", PreparedFixture)
    script = 'print(\'{"type":"system","subtype":"init","session_id":"s1"}\')'
    monkeypatch.setattr(adapter, "sandbox_command", lambda *args: [sys.executable, "-u", "-c", script])
    result = await adapter.ClaudeAdapter(ctx.runtimes).run(ctx)
    assert result["returncode"] == 0 and result["status"] == "runtime_error"


def test_command_contains_only_declared_model_and_tools(tmp_path):
    ctx = context(tmp_path)
    prepared = SimpleNamespace(mounts=lambda: [])
    command = adapter.sandbox_command(ctx, ctx.task.output, prepared)
    assert command[command.index("--tools") + 1] == ""
    assert command[command.index("--allowedTools") + 1] == "mcp__venusfactory__protein_probe"
    assert command[command.index("--model") + 1] == "fixture"
    assert "--clearenv" in command and "--strict-mcp-config" in command
    assert not any("DMXAPI_API_KEY" in item for item in command)


@pytest.mark.asyncio
async def test_host_uses_one_bridge_for_expert_and_native_calls(tmp_path, monkeypatch):
    from agent.harness import host, messages_bridge

    calls = []

    async def upstream(request):
        calls.append(json.loads(request.content))
        return httpx.Response(200, json=response())

    class OfflineBridge(MessagesBridge):
        def __init__(self, **kwargs):
            super().__init__(transport=httpx.MockTransport(upstream), **kwargs)

    class RoutingProbe:
        async def run(self, ctx):
            async with httpx.AsyncClient(trust_env=False) as client:
                body = {"model": "fixture", "messages": [{"role": "user", "content": "Plan"}]}
                a = await client.post(
                    ctx.relay.base_url + "/chat/completions",
                    json=body,
                    headers={"Authorization": "Bearer " + ctx.relay.token},
                )
                b = await client.post(
                    ctx.relay.base_url + "/messages",
                    json={**body, "max_tokens": 8192},
                    headers={"x-api-key": ctx.relay.token},
                )
                assert a.status_code == b.status_code == 200
            (ctx.task.output / "events.jsonl").write_text("")
            return {"status": "routing_fixture", "observed_requests": ctx.relay.calls}

    class Registry:
        def resolve(self, *args):
            return RoutingProbe(), {"ready": True, "fixture": True}

    monkeypatch.setattr(messages_bridge, "MessagesBridge", OfflineBridge)
    ctx = context(tmp_path)
    task = replace(ctx.task, output=tmp_path / "host_output")
    result = await host.execute(ctx.profile, task, {}, registry=Registry(), credential="offline-fixture")
    assert result["observed_requests"] == len(calls) == result["audit"]["requests"]["model_calls"] == 2
    lock = json.loads((task.output / "profile.lock.json").read_text())
    assert lock["transport"] == "shared_chat_messages_free_bridge.v1"
