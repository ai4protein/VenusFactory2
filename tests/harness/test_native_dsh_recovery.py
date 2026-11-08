"""Opt-in native SDK regression with local scripted model responses only.

Run with VENUS_TEST_NATIVE_DSH=1 when the pinned SDK and bwrap are installed.
"""
import json
import os
import shutil
from pathlib import Path

import httpx
import pytest

import agent.harness
from agent.harness import Budget, ModelBinding, Profile, Task, adapters
from agent.harness.events import audit_requests, audit_trace
from agent.harness.gateway import EvidenceGateway
from agent.harness.host import RuntimeContext
from agent.rsi.engine.model_gateway import GatewayPolicy
from scripts.rsi.run_harness_probe import default_runtimes

pytestmark = pytest.mark.skipif(
    os.environ.get("VENUS_TEST_NATIVE_DSH") != "1", reason="Native SDK verification is opt-in"
)


def model_response(payload, call_id, arguments=None):
    message = {"role": "assistant", "content": "Offline probe completed."}
    finish = "stop"
    if arguments is not None:
        message = {"role": "assistant", "content": None, "tool_calls": [{
            "id": call_id, "type": "function", "function": {
                "name": "mcp__venusfactory__protein_probe", "arguments": arguments,
            },
        }]}
        finish = "tool_calls"
    common = {"id": call_id, "model": "offline-fixture"}
    usage = {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20}
    if not payload.get("stream"):
        return httpx.Response(200, json={**common, "object": "chat.completion", "usage": usage,
            "choices": [{"index": 0, "message": message, "finish_reason": finish}]})
    delta = dict(message)
    if "tool_calls" in delta:
        delta["tool_calls"] = [{"index": 0, **call} for call in delta["tool_calls"]]
    chunks = [
        {**common, "object": "chat.completion.chunk",
         "choices": [{"index": 0, "delta": delta, "finish_reason": None}]},
        {**common, "choices": [{"index": 0, "delta": {}, "finish_reason": finish}], "usage": usage},
    ]
    body = "".join("data: " + json.dumps(chunk) + "\n\n" for chunk in chunks) + "data: [DONE]\n\n"
    return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})


def prepare(tmp_path, monkeypatch):
    runtimes = default_runtimes()
    assert shutil.which("bwrap") and Path(runtimes["dsh_bin"]).is_file()
    work, capability, output, runner = [tmp_path / name for name in ("work", "caps", "output", "runner")]
    for folder in (work, capability, output, runner):
        folder.mkdir()
    for filename in ("dsh_worker.py", "events.py", "mcp_config.py", "dsh_argument_guard.mjs"):
        source = next(Path(base) / filename for base in agent.harness.__path__ if (Path(base) / filename).is_file())
        shutil.copyfile(source, runner / filename)
    original = adapters.dsh_command

    def command(ctx, controls):
        args = original(ctx, controls)
        source = str(Path(adapters.__file__).parent)
        assert source in args
        return [str(runner) if value == source else value for value in args]

    monkeypatch.setattr(adapters, "dsh_command", command)
    (capability / "generated_tools.py").write_text(
        "from fastmcp import FastMCP\nfrom pathlib import Path\nimport json\n"
        "mcp = FastMCP('venusfactory')\n@mcp.tool()\n"
        "def protein_probe(sequence: str = 'DEFAULT') -> dict:\n"
        "    result = {'length': len(sequence), 'cysteines': sequence.count('C')}\n"
        "    with Path('/workspace/executions.jsonl').open('a') as f:\n"
        "        f.write(json.dumps(result) + '\\n')\n"
        "    return result\nif __name__ == '__main__': mcp.run()\n"
    )
    task = Task(work, capability, output, "Run the protein probe.", "Offline integration test.",
                frozenset({"protein_probe"}), phase="probe")
    return task, runtimes


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid", ['{"sequence":"truncated', "[]", "null", ""])
async def test_invalid_arguments_never_invoke_default_and_can_be_corrected(tmp_path, monkeypatch, invalid):
    task, runtimes = prepare(tmp_path, monkeypatch)
    calls = []

    def upstream(request):
        payload = json.loads(request.content)
        calls.append(payload)
        if len(calls) == 1:
            return model_response(payload, "invalid-call", invalid)
        if len(calls) == 2:
            feedback = [m for m in payload["messages"] if m.get("role") == "tool"
                        and m.get("tool_call_id") == "invalid-call"]
            assert len(feedback) == 1 and feedback[0]["content"]
            assert not (task.workspace / "executions.jsonl").exists()
            return model_response(payload, "valid-call", '{"sequence":"ACDEFGHIKC"}')
        assert len(calls) == 3
        return model_response(payload, "final")

    result, requests = await run(task, runtimes, upstream, max_calls=4)
    assert result["status"] == "completed" and result["sandbox_stopped"]
    assert result["tool_calls"] == 2 and len(calls) == 3
    events = audit_trace(task.output / "events.jsonl", task.tools, 4)
    assert events["passed"] and events["tool_errors"] == 1
    assert audit_requests(requests)["passed"]
    assert [json.loads(line) for line in (task.workspace / "executions.jsonl").read_text().splitlines()] == [
        {"length": 10, "cysteines": 2},
    ]


@pytest.mark.asyncio
async def test_native_budget_rejection_is_terminal_and_not_retried(tmp_path, monkeypatch):
    task, runtimes = prepare(tmp_path, monkeypatch)
    calls = []

    def upstream(request):
        payload = json.loads(request.content)
        calls.append(payload)
        return model_response(payload, "valid-call", '{"sequence":"ACDEFGHIKC"}')

    result, requests = await run(task, runtimes, upstream, max_calls=1)
    assert result["status"] == "runtime_error" and result["sandbox_stopped"]
    assert result["tool_calls"] == 1 and len(calls) == 1
    rows = [json.loads(line) for line in requests.read_text().splitlines()]
    rejected = [row for row in rows if row["event"] == "budget_rejected"]
    assert len(rejected) == 1 and rejected[0]["code"] == "model_call_budget_exhausted"
    assert audit_requests(requests)["errors"] == ["budget_rejected"]


async def run(task, runtimes, upstream, max_calls):
    profile = Profile("science_agent", "deepseek-harness",
        ModelBinding("offline", "offline-fixture", "http://127.0.0.1:1/v1"),
        Budget(wall_seconds=45, request_seconds=5, model_calls=max_calls, tool_calls=4,
               output_tokens=512, thinking="disabled"))
    policy = GatewayPolicy(model=profile.model.model, mode=profile.mode, scientific_tools=task.tools,
                           max_calls=max_calls, max_tokens=512, thinking="disabled")
    requests = task.output / "requests.jsonl"
    async with EvidenceGateway(policy=policy, upstream_url=profile.model.base_url, api_key="offline-fixture",
        output=requests, timeout_s=5, transport=httpx.MockTransport(upstream)) as relay:
        result = await adapters.DeepSeekAdapter(runtimes).run(RuntimeContext(profile, task, relay, runtimes))
    return result, requests
