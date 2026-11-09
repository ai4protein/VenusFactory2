"""Opt-in actual native Kimi SDK; every model response is generated locally."""
import asyncio
import json
import os
from pathlib import Path

import httpx
import pytest

from agent.harness import Budget, ModelBinding, Profile, Task
from agent.harness.kimi_adapter import KimiAdapter
from agent.harness.kimi_web_client import KimiWebClient
from agent.harness.installation import default_runtimes
from agent.harness.gateway import EvidenceGateway
from agent.harness.host import RuntimeContext
from agent.rsi.engine.model_gateway import GatewayPolicy

ROOT = Path(__file__).resolve().parents[2]


def response(payload, tool=False):
    if tool:
        deltas = [{"role": "assistant", "tool_calls": [{"index": 0, "id": "protein-probe",
            "type": "function", "function": {"name": "mcp__venusfactory__protein_probe", "arguments": '{"sequence":"ACDEFGHIKC"}'}}]}]
        finish = "tool_calls"
    else:
        deltas = [{"role": "assistant", "content": "Protein \U0001f9ec"}, {"content": " length: 10."}]
        finish = "stop"
    usage = {"prompt_tokens": 50, "completion_tokens": 20, "total_tokens": 70}
    if not payload.get("stream"):
        message = {"role": "assistant", "content": None if tool else "Protein \U0001f9ec length: 10."}
        if tool: message["tool_calls"] = [{k: v for k, v in deltas[0]["tool_calls"][0].items() if k != "index"}]
        return httpx.Response(200, json={"model": "offline-fixture", "choices": [{"index": 0, "message": message, "finish_reason": finish}], "usage": usage})
    chunks = [{"model": "offline-fixture", "choices": [{"index": 0, "delta": d, "finish_reason": None}]} for d in deltas]
    chunks.append({"model": "offline-fixture", "choices": [{"index": 0, "delta": {}, "finish_reason": finish}], "usage": usage})
    return httpx.Response(200, content="".join("data: " + json.dumps(c) + "\n\n" for c in chunks) + "data: [DONE]\n\n", headers={"content-type": "text/event-stream"})


def fixture_case(tmp_path, binary):
    runtimes = default_runtimes(ROOT)
    runtimes["kimi_bin"] = binary
    work, caps, output = [tmp_path / n for n in ["workspace", "capabilities", "execution"]]
    for path in (work, caps, output): path.mkdir()
    (caps / "generated_tools.py").write_text(
        "from fastmcp import FastMCP\nfrom pathlib import Path\nimport json\n"
        "mcp = FastMCP('venusfactory')\n@mcp.tool()\ndef protein_probe(sequence: str) -> dict:\n"
        "    result = {'length': len(sequence)}\n"
        "    with Path('/workspace/calls.jsonl').open('a') as f: f.write(json.dumps(result)+'\\n')\n"
        "    return result\nif __name__ == '__main__': mcp.run()\n"
    )
    instructions = "NATIVE_SYSTEM_BINDING_FIXTURE. Use the declared protein tools to answer the question."
    task = Task(work, caps, output, "Measure ACDEFGHIKC with protein_probe.", instructions,
                frozenset({"protein_probe"}), mcp_entry="generated_tools.py", phase="chat")
    profile = Profile("science_agent", "kimi-code", ModelBinding("fixture", "offline-fixture", "https://example.org/v1"),
        Budget(wall_seconds=90, model_calls=3, tool_calls=1, output_tokens=1024, thinking="disabled"))
    return runtimes, work, caps, output, instructions, task, profile


@pytest.mark.asyncio
@pytest.mark.parametrize("version,disconnect", [("2.5.2", False), ("2.5.2", True), ("0.19.2", False)])
async def test_actual_kimi_system_binding_tool_use_and_cleanup(tmp_path, monkeypatch, version, disconnect):
    interrupted = []
    original = KimiWebClient._wire_frames
    async def unstable(self, sid, since=0):
        stream = original(self, sid, since)
        try:
            async for item in stream:
                yield item
                if disconnect and not interrupted and item.get("type") == "transcript.ops" and any(
                        op.get("op") == "append" for op in item.get("payload", {}).get("ops", [])):
                    interrupted.append(True)
                    return
        finally:
            await stream.aclose()
    monkeypatch.setattr(KimiWebClient, "_wire_frames", unstable)
    binary = os.environ.get("VENUS_TEST_KIMI_BIN" if version == "2.5.2" else "VENUS_TEST_KIMI_LEGACY_BIN")
    if not binary: pytest.skip("Native Kimi binary is opt-in")
    runtimes, work, caps, output, instructions, task, profile = fixture_case(tmp_path, binary)
    calls = []
    def upstream(request):
        payload = json.loads(request.content); calls.append(payload)
        system = "\n".join(str(m.get("content", "")) for m in payload["messages"] if m["role"] == "system")
        user = "\n".join(str(m.get("content", "")) for m in payload["messages"] if m["role"] == "user")
        if version == "2.5.2":
            assert instructions in system and instructions not in user
        else:
            assert instructions in user
        if len(calls) > 1:
            assert any(m["role"] == "tool" and "10" in str(m["content"]) for m in payload["messages"])
        return response(payload, tool=len(calls) == 1)
    async with EvidenceGateway(policy=GatewayPolicy(model="offline-fixture", mode="science_agent", scientific_tools=task.tools,
            max_calls=3, thinking="disabled"), upstream_url=profile.model.base_url, api_key="unused-fixture",
            output=output / "model_requests.jsonl", transport=httpx.MockTransport(upstream)) as relay:
        result = await KimiAdapter(runtimes).run(RuntimeContext(profile, task, relay, runtimes))
    assert result["status"] == "completed", result
    assert result["text"] == "Protein \U0001f9ec length: 10."
    assert len(calls) == 2 and result["tool_calls"] == 1
    assert bool(interrupted) == disconnect
    assert result["native_version"] == version
    if version == "2.5.2":
        assert result["instruction_binding"] == "private SYSTEM.md"
        snapshot = result["runtime_session_usage"]
        if not (snapshot.get("input_tokens") or snapshot.get("output_tokens")):
            assert result["usage"].get("output") == 20
            assert not result["usage_status"].startswith("runtime_session_rollup")
        receipts = [json.loads(line) for line in (output / "model_requests.jsonl").read_text().splitlines()]
        usage = [row["usage"] for row in receipts if row["event"] == "response"]
        assert sum(row["prompt_tokens"] for row in usage) == 100
        assert sum(row["completion_tokens"] for row in usage) == 40
    else:
        assert result["native_prompt_retained"] is True
    assert result["sandbox_stopped"]
    assert (work / "calls.jsonl").read_text().splitlines() == ['{"length": 10}']
    events = [json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()]
    assert len([e for e in events if e["kind"] == "tool_result"]) == 1
    assert not (output / "runtime-private.toml").exists()
    assert not (output / "control/private_home/.kimi-code/config.toml").exists()
    assert not (output / "control/private_home/.kimi-code/server.token").exists()


@pytest.mark.asyncio
async def test_actual_native_cancellation_reaps_children_and_removes_credentials(tmp_path, monkeypatch):
    from agent.harness.kimi_web_sandbox import KimiWebSandbox
    binary = os.environ.get("VENUS_TEST_KIMI_BIN")
    if not binary: pytest.skip("Native Kimi binary is opt-in")
    runtimes, work, caps, output, instructions, task, profile = fixture_case(tmp_path, binary)
    ready, release = asyncio.Event(), asyncio.Event()
    instances = []
    original = KimiWebSandbox.start
    async def capture(self):
        instances.append(self)
        return await original(self)
    monkeypatch.setattr(KimiWebSandbox, "start", capture)
    async def upstream(request):
        ready.set()
        await release.wait()
        return response(json.loads(request.content))
    async with EvidenceGateway(policy=GatewayPolicy(model="offline-fixture", mode="science_agent",
            scientific_tools=task.tools, max_calls=3, thinking="disabled"),
            upstream_url=profile.model.base_url, api_key="unused-fixture", output=output / "model_requests.jsonl",
            transport=httpx.MockTransport(upstream)) as relay:
        execution = asyncio.create_task(KimiAdapter(runtimes).run(RuntimeContext(profile, task, relay, runtimes)))
        try:
            await asyncio.wait_for(ready.wait(), 30)
            sandbox = instances[0]
            async with httpx.AsyncClient(trust_env=False) as http:
                assert (await http.get(f"http://127.0.0.1:{sandbox.port}/api/v1/meta")).status_code == 401
            token = sandbox.client._token
            execution.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(execution, 15)
            assert sandbox.stopped and sandbox.proc.returncode is not None and sandbox.prepared_mcp.stopped
            assert sandbox.prepared_mcp.process.returncode is not None
            assert not (output / "runtime-private.toml").exists()
            for name in ("config.toml", "server.token"):
                assert not (sandbox.private_home / ".kimi-code" / name).exists()
            log = (output / "control/server.log").read_text()
            assert token not in log and relay.token not in log
        finally:
            release.set()
            if not execution.done():
                execution.cancel()
                try: await execution
                except asyncio.CancelledError: pass
