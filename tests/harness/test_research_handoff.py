"""Checkpoint handoff uses real MCP receipts and never fabricates an LLM answer."""

import asyncio
import copy
import json
from types import SimpleNamespace

import httpx
import pytest

from agent.harness import web_native_worker
from agent.harness.contracts import Budget, ModelBinding, Profile
from agent.harness.research_handoff import ResearchHandoff, closed_response, native_phase
from agent.harness.research_workspace import ResearchWorkspace

ARGS = {
    "summary": "Inspected the supplied data",
    "completed_steps": [],
    "artifacts": ["result.json"],
    "needs_input": [],
}
RESULT = {"checkpointed": True, "artifacts": [{"path": "result.json", "sha256": "a" * 64, "bytes": 2}]}


def trace_rows(broker=False):
    name = "call_tool" if broker else "submit_research_checkpoint"
    args = {"name": "submit_research_checkpoint", "arguments": copy.deepcopy(ARGS)} if broker else copy.deepcopy(ARGS)
    return [
        {
            "kind": "tool_call_start",
            "tool_call_id": "checkpoint",
            "tool_name": "mcp__venusfactory__" + name,
            "args": args,
        },
        {
            "kind": "tool_result",
            "tool_call_id": "checkpoint",
            "is_error": False,
            "output": json.dumps(RESULT)
            + "\nCheckpoint saved. End this iteration with a short summary; do not call further tools.",
        },
    ]


def write_trace(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


@pytest.mark.parametrize("broker", [False, True])
def test_handoff_requires_exact_checkpoint_evidence(tmp_path, broker):
    boundary = ResearchHandoff()
    assert not boundary.closed and boundary.fingerprint is None
    boundary.save(ARGS, RESULT)
    path = tmp_path / "trace.jsonl"
    write_trace(path, trace_rows(broker))
    audit = boundary.audit(path, {"submit_research_checkpoint", "call_tool"}, 4)
    assert audit["passed"] and audit["checkpoint_call_id"] == "checkpoint"
    event = boundary.event(audit, executions=1, lookups=0, checkpoints=1)
    assert event["text"] == ARGS["summary"] and not event["research_complete"]
    assert event["completion_source"] == "research_checkpoint_handoff" and "messages" not in event
    with pytest.raises(ValueError):
        boundary.save(ARGS, RESULT)


@pytest.mark.parametrize(
    "fault",
    [
        "unpaired",
        "wrong_arguments",
        "failed",
        "wrong_result",
        "duplicate",
        "undeclared",
        "limit",
        "missing",
        "incomplete",
        "unsaved",
    ],
)
def test_incomplete_or_changed_traces_cannot_trigger_acceptance(tmp_path, fault):
    boundary = ResearchHandoff()
    if fault != "unsaved":
        boundary.save(ARGS, RESULT)
    rows = trace_rows()
    limit = 4
    if fault == "unpaired":
        rows.pop()
    elif fault == "wrong_arguments":
        rows[0]["args"]["summary"] = "Unobserved work"
    elif fault == "failed":
        rows[1]["is_error"] = True
    elif fault == "wrong_result":
        rows[1]["output"] = json.dumps({**RESULT, "artifacts": []})
    elif fault == "duplicate":
        rows.extend(copy.deepcopy(rows))
    elif fault == "undeclared":
        rows[0]["tool_name"] = "mcp__venusfactory__shell"
    elif fault == "limit":
        limit = 0
    path = tmp_path / "trace.jsonl"
    if fault != "missing":
        write_trace(path, rows)
    if fault == "incomplete":
        path.write_text('{"kind":')
    audit = boundary.audit(path, {"submit_research_checkpoint"}, limit)
    assert not audit["passed"]
    with pytest.raises(ValueError):
        boundary.event(audit, executions=1, lookups=0, checkpoints=1)


def test_gate_ends_only_the_native_phase_and_does_not_charge_requests():
    policy = SimpleNamespace()
    logs = []
    assert closed_response(policy, logs.append) is None
    with native_phase(policy) as boundary:
        assert closed_response(policy, logs.append) is None
        with pytest.raises(ValueError):
            with native_phase(policy):
                pass
        boundary.save(ARGS, RESULT)
        response = closed_response(policy, logs.append)
        assert response.status == 400
        assert json.loads(response.body)["error"]["code"] == "research_checkpoint_handoff"
        assert logs == [{"event": "research_checkpoint_handoff", "checkpoint_sha256": boundary.fingerprint}]
    assert closed_response(policy, logs.append) is None
    assert not any(row.get("event") in {"request", "policy_rejected", "budget_rejected"} for row in logs)


@pytest.mark.asyncio
@pytest.mark.parametrize("harness", ["deepseek-harness", "kimi-code", "codex", "claude-code"])
@pytest.mark.parametrize("exit_kind", ["wait", "error"])
async def test_native_checkpoint_handoff_stops_the_worker_with_paired_receipts(
    tmp_path, monkeypatch, harness, exit_kind
):
    monkeypatch.setenv("VENUS_JOB_QUEUE_ROOT", str(tmp_path / "queue"))
    monkeypatch.setattr("mcp_tools.scheduler.catalog.native_tools", lambda: {})
    service = ResearchWorkspace(tmp_path / "work", tmp_path / "tools", {"steps": [{"id": "inspect"}]})
    (tmp_path / "control").mkdir()
    events, stopped = [], []
    boundary = ResearchHandoff()

    class Client:
        async def run(self, context):
            trace = context.task.output / ("control/events.jsonl" if harness == "deepseek-harness" else "events.jsonl")
            rows = []
            connection = json.loads((context.task.capability / "connection.json").read_text())
            try:
                async with httpx.AsyncClient(trust_env=False) as client:

                    async def invoke(name, args):
                        identity = str(len(rows))
                        rows.append(
                            {
                                "kind": "tool_call_start",
                                "tool_call_id": identity,
                                "tool_name": "mcp__venusfactory__" + name,
                                "args": args,
                            }
                        )
                        write_trace(trace, rows)
                        response = await client.post(
                            connection["url"],
                            headers={"Authorization": "Bearer " + connection["token"]},
                            json={
                                "jsonrpc": "2.0",
                                "id": identity,
                                "method": "tools/call",
                                "params": {"name": name, "arguments": args},
                            },
                        )
                        result = response.json()["result"]
                        rows.append(
                            {
                                "kind": "tool_result",
                                "tool_call_id": identity,
                                "is_error": result.get("isError", False),
                                "output": "\n".join(item["text"] for item in result["content"]),
                            }
                        )
                        write_trace(trace, rows)
                        return result

                    await invoke("write_session_file", {"path": "result.json", "content": "{}"})
                    result = await invoke("submit_research_checkpoint", ARGS)
                    assert not result["isError"] and boundary.closed
                    if exit_kind == "error":
                        raise RuntimeError("SDK stopped after checkpoint")
                    await asyncio.Event().wait()
            finally:
                stopped.append(True)

    monkeypatch.setattr(
        web_native_worker, "builtin_registry", lambda: SimpleNamespace(resolve=lambda *a: (Client(), None))
    )
    profile = Profile(
        "science_agent", harness, ModelBinding("fixture", "unused", "http://127.0.0.1:1/v1"), Budget(tool_calls=4)
    )
    await asyncio.wait_for(
        web_native_worker.run(
            {
                "session_id": "fixture",
                "workspace": str(service.workspace),
                "control": str(tmp_path / "control"),
                "profile": profile.public(),
                "runtimes": {},
                "base_url": "http://127.0.0.1:1",
                "token": "unused",
                "text": "Inspect inputs",
                "audit_required": True,
            },
            tools=service.tools(),
            event_sink=events.append,
            checkpoint_handoff=boundary,
        ),
        timeout=5,
    )
    assert stopped == [True] and service.checkpoint
    completed = [e for e in events if e["kind"] == "completed"]
    assert len(completed) == 1 and completed[0]["event_audit"]["passed"]
    assert completed[0]["checkpoint_calls"] == 1 and completed[0]["tool_calls"] == 1
    assert completed[0]["completion_source"] == "research_checkpoint_handoff"
    assert not completed[0]["research_complete"]
    assert not [e for e in events if e["kind"] == "text"]
    assert not list((tmp_path / "control").rglob("connection.json"))


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["chat", "responses", "messages", "responses_bridge"])
async def test_http_gate_stops_native_requests_but_reopens_for_independent_sc(tmp_path, monkeypatch, kind):
    from agent.harness.gateway import EvidenceGateway
    from agent.harness.messages_bridge import MessagesBridge
    from agent.harness.responses_bridge import ResponsesChatBridge
    from agent.harness.responses_gateway import ResponsesGateway, ResponsesPolicy
    from agent.rsi.engine.model_gateway import GatewayPolicy

    monkeypatch.setenv("VENUS_JOB_QUEUE_ROOT", str(tmp_path / "queue"))
    gateways = {
        "chat": EvidenceGateway,
        "responses": ResponsesGateway,
        "messages": MessagesBridge,
        "responses_bridge": ResponsesChatBridge,
    }
    policy = (
        ResponsesPolicy(model="fixture-model", scientific_tools=set())
        if kind == "responses"
        else GatewayPolicy(model="fixture-model", mode="science_agent", scientific_tools=set(), max_calls=3)
    )
    route = {
        "chat": "/chat/completions",
        "responses": "/responses",
        "messages": "/messages",
        "responses_bridge": "/responses",
    }[kind]
    payload = {"model": "fixture-model", "stream": False}
    if kind in {"responses", "responses_bridge"}:
        payload.update(input="Independently review actual evidence", store=False)
    else:
        payload.update(messages=[{"role": "user", "content": "Independently review actual evidence"}])
        if kind == "messages":
            policy.thinking = "disabled"
            payload["max_tokens"] = policy.max_tokens
    seen = []

    async def upstream(request):
        seen.append(json.loads(request.content))
        # A real upstream rejection proves requests are routed again; it must
        # never be converted into a synthetic successful reviewer response.
        return httpx.Response(503, json={"error": {"message": "fixture unavailable"}})

    output = tmp_path / "requests.jsonl"
    async with gateways[kind](
        policy=policy,
        upstream_url="https://fixture.invalid/v1",
        api_key="fixture",
        output=output,
        transport=httpx.MockTransport(upstream),
    ) as gateway:
        async with httpx.AsyncClient(trust_env=False) as client:
            with native_phase(policy) as boundary:
                boundary.save(ARGS, RESULT)
                unauthorized = await client.post(gateway.base_url + route, json=payload)
                assert unauthorized.status_code == 401
                for _ in range(3):
                    stopped = await client.post(
                        gateway.base_url + route, json=payload, headers={"Authorization": "Bearer " + gateway.token}
                    )
                    assert stopped.status_code == 400
                    assert stopped.json()["error"]["code"] == "research_checkpoint_handoff"
                assert gateway.calls == 0 and not seen
            resumed = await client.post(
                gateway.base_url + route, json=payload, headers={"Authorization": "Bearer " + gateway.token}
            )
            assert resumed.status_code == 503
            assert gateway.calls == 1 and len(seen) == 1
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert sum(row["event"] == "research_checkpoint_handoff" for row in rows) == 3
    assert sum(row["event"] == "request" for row in rows) == 1
    assert not any(row["event"] in {"policy_rejected", "budget_rejected"} for row in rows)


@pytest.mark.asyncio
async def test_write_already_waiting_for_workspace_lock_cannot_change_saved_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setenv("VENUS_JOB_QUEUE_ROOT", str(tmp_path / "queue"))
    monkeypatch.setattr("mcp_tools.scheduler.catalog.native_tools", lambda: {})
    service = ResearchWorkspace(tmp_path / "work", tmp_path / "tools", {"steps": [{"id": "inspect"}]})
    (service.workspace / "result.json").write_text("{}")
    (tmp_path / "control").mkdir()
    entered, release = asyncio.Event(), asyncio.Event()
    boundary = ResearchHandoff()
    tools = service.tools()
    checkpoint_tool = next(tool for tool in tools if tool.name == "submit_research_checkpoint")
    original = checkpoint_tool.coroutine

    async def slow_checkpoint(*args, **kwargs):
        entered.set()
        await release.wait()
        return await original(*args, **kwargs)

    checkpoint_tool.coroutine = slow_checkpoint
    events = []

    def observe(event):
        events.append(event)
        if (
            event.get("kind") == "research_tool_status"
            and event.get("status") == "running"
            and event.get("tool") == "write_session_file"
        ):
            release.set()

    class Client:
        async def run(self, context):
            connection = json.loads((context.task.capability / "connection.json").read_text())
            rows = []
            async with httpx.AsyncClient(trust_env=False) as client:

                async def invoke(identity, name, args):
                    rows.append(
                        {
                            "kind": "tool_call_start",
                            "tool_call_id": identity,
                            "tool_name": "mcp__venusfactory__" + name,
                            "args": args,
                        }
                    )
                    response = await client.post(
                        connection["url"],
                        headers={"Authorization": "Bearer " + connection["token"]},
                        json={
                            "jsonrpc": "2.0",
                            "id": identity,
                            "method": "tools/call",
                            "params": {"name": name, "arguments": args},
                        },
                    )
                    result = response.json()["result"]
                    rows.append(
                        {
                            "kind": "tool_result",
                            "tool_call_id": identity,
                            "is_error": result.get("isError", False),
                            "output": "\n".join(part["text"] for part in result["content"]),
                        }
                    )
                    return result

                saved = asyncio.create_task(invoke("save", "submit_research_checkpoint", ARGS))
                await entered.wait()
                write = asyncio.create_task(
                    invoke(
                        "late-write",
                        "write_session_file",
                        {"path": "result.json", "content": "changed after checkpoint"},
                    )
                )
                receipt, denied = await asyncio.gather(saved, write)
                assert not receipt["isError"] and denied["isError"]
                assert "Checkpoint saved" in denied["content"][0]["text"]
                write_trace(context.task.output / "control/events.jsonl", rows)
            return {"status": "completed", "text": "Checkpoint saved"}

    monkeypatch.setattr(
        web_native_worker, "builtin_registry", lambda: SimpleNamespace(resolve=lambda *args: (Client(), None))
    )
    profile = Profile(
        "science_agent",
        "deepseek-harness",
        ModelBinding("fixture", "unused", "http://127.0.0.1:1/v1"),
        Budget(tool_calls=4),
    )
    await asyncio.wait_for(
        web_native_worker.run(
            {
                "session_id": "concurrent-fixture",
                "workspace": str(service.workspace),
                "control": str(tmp_path / "control"),
                "profile": profile.public(),
                "runtimes": {},
                "base_url": "http://127.0.0.1:1",
                "token": "unused",
                "text": "Save inspected data",
                "audit_required": True,
            },
            tools=tools,
            event_sink=observe,
            checkpoint_handoff=boundary,
        ),
        timeout=5,
    )
    assert (service.workspace / "result.json").read_text() == "{}"
    assert next(event for event in events if event["kind"] == "completed")["event_audit"]["passed"]
