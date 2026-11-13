"""A host-observed phase cap produces audited partial work, never a completion."""
import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest
from langchain_core.tools import StructuredTool
from test_research_phase_budget import local_queue, relay_fixture  # noqa: F401

from agent.harness import web_native_worker
from agent.harness.contracts import Budget, ModelBinding, Profile
from agent.harness.research_handoff import native_phase
from agent.harness.research_workspace import ResearchWorkspace

HARNESSES = ("deepseek-harness", "kimi-code", "codex", "claude-code")


async def exercise(tmp_path, monkeypatch, *, harness="kimi-code", exit_kind="error",
                   known_limit=True, failed_tool=False, slow_tool=False, outer_cancel=False):
    monkeypatch.setattr("mcp_tools.scheduler.catalog.native_tools", lambda: {})
    service = ResearchWorkspace(tmp_path / "work", tmp_path / "tools", {"steps": [{"id": "inspect"}]})
    control = tmp_path / "control"
    control.mkdir()
    tools = service.tools()
    events = []
    entered, cleanup_started, cleaned = asyncio.Event(), asyncio.Event(), asyncio.Event()
    adapter_stopped = []
    if slow_tool:
        original = next(tool for tool in tools if tool.name == "write_session_file")

        async def delayed_write(**arguments):
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                cleanup_started.set()
                await asyncio.sleep(0.1)
                cleaned.set()

        replacement = StructuredTool.from_function(coroutine=delayed_write, name=original.name,
                     description=original.description, args_schema=original.get_input_schema(), metadata=original.metadata)
        tools = [replacement if tool is original else tool for tool in tools]

    class Client:
        async def run(self, ctx):
            trace = ctx.task.output / "events.jsonl"
            trace.parent.mkdir(parents=True, exist_ok=True)
            trace.touch()
            connection = json.loads((ctx.task.capability / "connection.json").read_text())
            tool_task = None

            def record(row):
                with trace.open("a") as stream:
                    stream.write(json.dumps(row) + "\n")

            try:
                async with httpx.AsyncClient(trust_env=False, timeout=5) as client:
                    async def invoke():
                        name = "write_session_file"
                        arguments = {"path": "../outside.txt" if failed_tool else "partial.json", "content": "{}"}
                        record({"kind": "tool_call_start", "tool_call_id": "write1",
                                "tool_name": "mcp__venusfactory__" + name, "args": arguments})
                        response = await client.post(connection["url"],
                            headers={"Authorization": "Bearer " + connection["token"]},
                            json={"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                                  "params": {"name": name, "arguments": arguments}})
                        result = response.json()["result"]
                        record({"kind": "tool_result", "tool_call_id": "write1", "is_error": result["isError"],
                                "output": "\n".join(item["text"] for item in result["content"])})

                    if slow_tool:
                        tool_task = asyncio.create_task(invoke())
                        await asyncio.wait_for(entered.wait(), 2)
                    else:
                        await invoke()
                    body = {"model": "fixture-model", "stream": False,
                            "messages": [{"role": "user", "content": "Synthetic implementation step"}]}
                    response = await client.post(ctx.relay.base_url + "/chat/completions", json=body,
                                                 headers={"Authorization": "Bearer " + ctx.relay.token})
                    assert response.status_code == 200
                    if not known_limit:
                        return {"status": "runtime_error", "error_type": "research_phase_budget_exhausted"}
                    response = await client.post(ctx.relay.base_url + "/chat/completions", json=body,
                                                 headers={"Authorization": "Bearer " + ctx.relay.token})
                    assert response.status_code == 400
                    assert response.json()["error"]["code"] == "research_phase_budget_exhausted"
                    if exit_kind == "error":
                        response.raise_for_status()
                    elif exit_kind == "status":
                        return {"status": "runtime_error", "error_type": "ModelBudgetExceeded"}
                    elif exit_kind == "success":
                        return {"status": "completed", "text": "This unsupported completion must not be emitted"}
                    await asyncio.Event().wait()
            finally:
                if tool_task is not None:
                    tool_task.cancel()
                    await asyncio.gather(tool_task, return_exceptions=True)
                adapter_stopped.append(True)

    monkeypatch.setattr(web_native_worker, "builtin_registry",
                        lambda: SimpleNamespace(resolve=lambda *args: (Client(), None)))

    def send(event):
        if event["kind"] == "research_interrupted":
            assert adapter_stopped == [True]
            assert not list(control.rglob("connection.json"))
            if slow_tool:
                assert cleaned.is_set()
        events.append(event)

    async with relay_fixture(tmp_path, "chat", limit=4) as case:
        profile = Profile("science_agent", harness, ModelBinding("fixture", "fixture-model", case.relay.base_url),
                          Budget(model_calls=1, tool_calls=4, wall_seconds=10))
        with native_phase(case.policy, relay=case.relay) as boundary:
            assert boundary.request_budget_exhausted is False
            with pytest.raises(AttributeError):
                boundary.request_budget_exhausted = True
            request = {"session_id": "fixture", "workspace": str(service.workspace), "control": str(control),
                       "profile": profile.public(), "runtimes": {}, "base_url": case.relay.base_url,
                       "token": boundary.relay_token, "text": "Inspect supplied synthetic data", "audit_required": True}
            task = asyncio.create_task(web_native_worker.run(request, tools=tools, event_sink=send,
                                                            checkpoint_handoff=boundary))
            if outer_cancel:
                await asyncio.wait_for(cleanup_started.wait(), 3)
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            elif not known_limit:
                with pytest.raises(RuntimeError, match="research_phase_budget_exhausted"):
                    await asyncio.wait_for(task, 3)
            else:
                await asyncio.wait_for(task, 3)
            assert boundary.request_budget_exhausted is known_limit
            assert case.relay.calls == 1 and len(case.seen) == 1
        assert (await case.send()).status_code == 200
        assert case.relay.calls == 2
    assert adapter_stopped == [True]
    assert service.checkpoint is None
    assert not list(control.rglob("connection.json"))
    assert not [row for row in events if row["kind"] in {"completed", "text"}]
    return SimpleNamespace(events=events, control=control, workspace=service.workspace, cleaned=cleaned)


@pytest.mark.asyncio
@pytest.mark.parametrize("harness", HARNESSES)
@pytest.mark.parametrize("exit_kind", ["error", "status", "wait", "success"])
async def test_confirmed_budget_cutoff_yields_partial_event_after_cleanup(tmp_path, monkeypatch, harness, exit_kind):
    case = await exercise(tmp_path, monkeypatch, harness=harness, exit_kind=exit_kind)
    interruptions = [row for row in case.events if row["kind"] == "research_interrupted"]
    assert len(interruptions) == 1
    event = interruptions[0]
    assert event["reason"] == "native_model_request_limit"
    assert event["research_complete"] is False
    assert event["event_audit"]["passed"] is True
    assert event["event_audit"]["tool_calls"] == 1
    assert event["event_audit"]["tool_errors"] == 0
    assert event["tool_calls"] == 1 and event["checkpoint_calls"] == 0
    assert "text" not in event and "messages" not in event
    assert json.loads((case.workspace / "partial.json").read_text()) == {}
    receipt = json.loads((case.control / "native/request_budget_interruption.json").read_text())
    assert receipt == event


@pytest.mark.asyncio
async def test_partial_event_keeps_real_failed_tool_receipt(tmp_path, monkeypatch):
    case = await exercise(tmp_path, monkeypatch, failed_tool=True)
    event = next(row for row in case.events if row["kind"] == "research_interrupted")
    assert event["event_audit"]["passed"] is True
    assert event["event_audit"]["tool_errors"] == 1
    assert not (tmp_path / "outside.txt").exists()
    assert any(row["kind"] == "tool_result" and row["is_error"] for row in case.events)


@pytest.mark.asyncio
async def test_pending_tool_is_cleaned_and_unpaired_trace_is_not_repaired(tmp_path, monkeypatch):
    case = await exercise(tmp_path, monkeypatch, slow_tool=True, exit_kind="wait")
    event = next(row for row in case.events if row["kind"] == "research_interrupted")
    assert case.cleaned.is_set()
    assert event["event_audit"]["passed"] is False
    assert "unpaired_tool_calls" in event["event_audit"]["problems"]
    assert any(row["kind"] == "research_tool_status" and row["status"] == "cancelled" for row in case.events)
    assert not (case.workspace / "partial.json").exists()


@pytest.mark.asyncio
async def test_outer_cancellation_during_tool_cleanup_does_not_become_sc_interruption(tmp_path, monkeypatch):
    case = await exercise(tmp_path, monkeypatch, slow_tool=True, exit_kind="wait", outer_cancel=True)
    assert case.cleaned.is_set()
    assert not any(row["kind"] == "research_interrupted" for row in case.events)
    assert not (case.control / "native/request_budget_interruption.json").exists()


@pytest.mark.asyncio
async def test_native_claim_of_budget_exhaustion_without_gateway_evidence_still_raises(tmp_path, monkeypatch):
    case = await exercise(tmp_path, monkeypatch, known_limit=False)
    assert not any(row["kind"] == "research_interrupted" for row in case.events)
    assert not (case.control / "native/request_budget_interruption.json").exists()
