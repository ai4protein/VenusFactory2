"""Actual local MCP attempts may reach SC after native failure, never completion."""
import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest
from langchain_core.tools import StructuredTool
from test_research_phase_budget import local_queue  # noqa: F401

from agent.harness import web_native_worker
from agent.harness.contracts import Budget, ModelBinding, Profile
from agent.harness.research_handoff import ResearchHandoff
from agent.harness.research_workspace import ResearchWorkspace


async def exercise(tmp_path, monkeypatch, *, harness="kimi-code", work="training", exit_kind="status",
                   handoff=True, ordinary_mode=None, outer_cancel=False):
    monkeypatch.setattr("mcp_tools.scheduler.catalog.native_tools", lambda: {})
    service = ResearchWorkspace(tmp_path / "work", tmp_path / "tools", {"steps": [{"id": "fit"}]})
    control = tmp_path / "control"
    control.mkdir()
    tools = service.tools()
    entered, cleanup_started, cleaned = asyncio.Event(), asyncio.Event(), asyncio.Event()
    adapter_stopped, events = [], []
    if work in {"slow", "late_denied"}:
        original = next(t for t in tools if t.name == "run_session_python")

        async def invoke_python(**arguments):
            if work == "late_denied":
                from mcp_tools.tools.capability_policy import CapabilityDisabled
                raise CapabilityDisabled("Disabled after discovery")
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                cleanup_started.set()
                await asyncio.sleep(0.1)
                cleaned.set()

        replacement = StructuredTool.from_function(coroutine=invoke_python, name=original.name,
            description=original.description, args_schema=original.get_input_schema(), metadata=original.metadata)
        tools = [replacement if t is original else t for t in tools]
    if work.startswith("queue_"):
        async def calculate(sequence: str) -> dict:
            """Queue fixture that must not execute in the Web worker."""
            raise AssertionError("must use queued invocation")

        queued_tool = StructuredTool.from_function(coroutine=calculate, name="calculate_physchem_from_sequence",
                                                   description="Inspect a sequence")
        tools.append(queued_tool)
        monkeypatch.setattr("mcp_tools.scheduler.catalog.native_tools", lambda: {queued_tool.name: queued_tool})

        async def queue_invoke(name, arguments, **callbacks):
            assert name == queued_tool.name and arguments["sequence"] == "ACDE"
            if work == "queue_admission_value":
                raise ValueError("Queue admission rejected")
            if work == "queue_admission_http":
                from fastapi import HTTPException
                raise HTTPException(status_code=429, detail="Quota reached")
            job = {"id": "host-job", "job_number": "TEST-1", "position": 1, "status": "queued"}
            if work == "queue_waiting":
                callbacks["on_status"](job)
                raise RuntimeError("Failed before execution")
            if work in {"queue_running", "queue_policy_rejection", "queue_startup"}:
                callbacks["on_status"]({**job, "status": "running", "position": 0})
                if work != "queue_running":
                    callbacks["on_terminal"]({**job, "status": "failed", "result": {"is_error": True,
                        "error_type": "CapabilityDisabled" if work == "queue_policy_rejection" else "ModuleNotFoundError"}})
                raise RuntimeError("Failure without a scientific handler value")
            value = {"success": False, "error": "Scientific preflight rejected"}
            callbacks["on_terminal"]({**job, "status": "failed", "result": {"value": value, "is_error": True}})
            return value

        monkeypatch.setattr("mcp_tools.scheduler.client.invoke", queue_invoke)
    if ordinary_mode:
        tools = [t for t in tools if t.name != "submit_research_checkpoint"]
    boundary = ResearchHandoff() if handoff else None

    class Client:
        async def run(self, ctx):
            trace = ctx.task.output / "events.jsonl"
            trace.parent.mkdir(parents=True, exist_ok=True)
            trace.touch()
            connection = json.loads((ctx.task.capability / "connection.json").read_text())
            rows = []
            tool_task = None

            def record(row):
                rows.append(row)
                trace.write_text("".join(json.dumps(item) + "\n" for item in rows))

            try:
                async with httpx.AsyncClient(trust_env=False, timeout=5) as client:
                    async def invoke(name, arguments):
                        identity = str(len(rows))
                        record({"kind": "tool_call_start", "tool_call_id": identity,
                                "tool_name": "mcp__venusfactory__" + name, "args": arguments})
                        response = await client.post(connection["url"],
                            headers={"Authorization": "Bearer " + connection["token"]},
                            json={"jsonrpc": "2.0", "id": identity, "method": "tools/call",
                                  "params": {"name": name, "arguments": arguments}})
                        payload = response.json()
                        result = payload.get("result", {"isError": True, "content": [{"text": "RPC rejected"}]})
                        record({"kind": "tool_result", "tool_call_id": identity, "is_error": result["isError"],
                                "output": "\n".join(item["text"] for item in result["content"])})
                        return result

                    if work.startswith("queue_"):
                        await invoke("describe_tool", {"name": "calculate_physchem_from_sequence"})
                        result = await invoke("call_tool", {"name": "calculate_physchem_from_sequence", "arguments": {"sequence": "ACDE"}})
                        assert result["isError"]
                    elif work in {"training", "bad_arguments"}:
                        arguments = {"name": "preflight", "source_file": "missing.py", "evaluation_file": "missing.json",
                                     "hypothesis": "A real training preflight is not a fitted model"}
                        if work == "bad_arguments":
                            arguments = {"name": "preflight"}
                        if ordinary_mode:
                            await invoke("describe_tool", {"name": "train_research_model"})
                            result = await invoke("call_tool", {"name": "train_research_model", "arguments": arguments})
                        else:
                            result = await invoke("train_research_model", arguments)
                        assert result["isError"]
                    elif work in {"slow", "late_denied"}:
                        tool_task = asyncio.create_task(invoke("run_session_python", {"path": "analysis.py"}))
                        if work == "slow":
                            await asyncio.wait_for(entered.wait(), 2)
                        else:
                            assert (await tool_task)["isError"]
                    elif work == "write":
                        await invoke("write_session_file", {"path": "draft.md", "content": "Not a scientific result"})
                    elif work == "discovery":
                        await invoke("describe_tool", {"name": "train_research_model"})
                    elif work == "unknown":
                        assert (await invoke("unknown_scientific_tool", {}))["isError"]
                    elif work == "forged":
                        record({"kind": "tool_call_start", "tool_call_id": "invented",
                                "tool_name": "mcp__venusfactory__train_research_model", "args": {}})
                        record({"kind": "tool_result", "tool_call_id": "invented", "is_error": False,
                                "output": "training succeeded"})
                    if exit_kind == "exception":
                        raise RuntimeError("Provider payload must not appear: PRIVATE_FIXTURE_TOKEN")
                    if exit_kind == "empty":
                        return {"status": "completed", "final_text": "  "}
                    if exit_kind == "success":
                        return {"status": "completed", "final_text": "The attempted preflight failed."}
                    if exit_kind == "unsafe_error":
                        return {"status": "PRIVATE_FIXTURE_TOKEN /remote/path", "error_type": "secret=PRIVATE_FIXTURE_TOKEN"}
                    return {"status": "runtime_error", "error_type": "research_phase_budget_exhausted"
                            if exit_kind == "budget_string" else "model_response_empty"}
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
            if work == "slow":
                assert cleaned.is_set()
        events.append(event)

    profile = Profile("science_agent", harness, ModelBinding("fixture", "fixture-model", "http://127.0.0.1:1"),
                      Budget(model_calls=3, tool_calls=6, wall_seconds=10))
    request = {"session_id": "fixture", "workspace": str(service.workspace), "control": str(control),
               "profile": profile.public(), "runtimes": {}, "base_url": "http://127.0.0.1:1", "token": "unused",
               "text": "Inspect supplied synthetic data", "audit_required": True,
               "research_focus": [] if ordinary_mode else ["train_research_model"]}
    if ordinary_mode:
        request["mode"] = ordinary_mode
    task = asyncio.create_task(web_native_worker.run(request, tools=tools, event_sink=send,
                                                    checkpoint_handoff=boundary))
    if outer_cancel:
        await asyncio.wait_for(cleanup_started.wait(), 3)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    elif exit_kind != "success" and (not handoff or work in {"none", "write", "discovery", "forged", "unknown", "bad_arguments", "late_denied", "queue_admission_value", "queue_admission_http", "queue_waiting", "queue_running", "queue_policy_rejection", "queue_startup"}):
        with pytest.raises((RuntimeError, ValueError)):
            await asyncio.wait_for(task, 4)
    else:
        await asyncio.wait_for(task, 4)
    assert adapter_stopped == [True]
    assert not list(control.rglob("connection.json"))
    assert service.checkpoint is None
    if boundary:
        assert not boundary.request_budget_exhausted
    return SimpleNamespace(events=events, control=control, cleaned=cleaned)


@pytest.mark.asyncio
@pytest.mark.parametrize("harness", ["deepseek-harness", "kimi-code", "codex", "claude-code"])
@pytest.mark.parametrize("exit_kind", ["exception", "status", "empty"])
async def test_real_training_preflight_failure_reaches_sc_after_cleanup(tmp_path, monkeypatch, harness, exit_kind):
    case = await exercise(tmp_path, monkeypatch, harness=harness, exit_kind=exit_kind)
    event = next(row for row in case.events if row["kind"] == "research_interrupted")
    assert event["reason"] == "native_execution_failed" and event["research_complete"] is False
    assert event["event_audit"]["passed"] is True and event["event_audit"]["tool_errors"] == 1
    assert event["scientific_tool_attempts"][0]["tool"] == "train_research_model"
    assert event["scientific_tool_attempts"][0]["status"] == "failed"
    assert event["native_error_type"] == {"exception": "RuntimeError", "status": "model_response_empty", "empty": "ValueError"}[exit_kind]
    assert event["native_status"] == {"exception": "exception", "status": "runtime_error", "empty": "completed"}[exit_kind]
    assert not any(row["kind"] in {"text", "completed"} for row in case.events)
    assert "PRIVATE_FIXTURE_TOKEN" not in json.dumps(event)
    assert json.loads((case.control / "native/execution_failure_interruption.json").read_text()) == event
    assert not (case.control / "native/request_budget_interruption.json").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("work", ["none", "write", "discovery", "forged", "unknown", "bad_arguments", "late_denied"])
async def test_no_admitted_scientific_attempt_cannot_become_partial_work(tmp_path, monkeypatch, work):
    case = await exercise(tmp_path, monkeypatch, work=work)
    assert not any(row["kind"] == "research_interrupted" for row in case.events)
    assert not (case.control / "native/execution_failure_interruption.json").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["agent", "researcher"])
async def test_ordinary_modes_still_raise_native_failure(tmp_path, monkeypatch, mode):
    case = await exercise(tmp_path, monkeypatch, handoff=False, ordinary_mode=mode)
    assert not any(row["kind"] == "research_interrupted" for row in case.events)


@pytest.mark.asyncio
async def test_cleanup_settles_before_handoff_and_does_not_repair_unpaired_trace(tmp_path, monkeypatch):
    case = await exercise(tmp_path, monkeypatch, work="slow")
    event = next(row for row in case.events if row["kind"] == "research_interrupted")
    assert case.cleaned.is_set()
    assert event["scientific_tool_attempts"][0]["status"] == "cancelled"
    assert event["event_audit"]["passed"] is False
    assert "unpaired_tool_calls" in event["event_audit"]["problems"]


@pytest.mark.asyncio
async def test_outer_cancel_during_cleanup_never_sends_handoff(tmp_path, monkeypatch):
    case = await exercise(tmp_path, monkeypatch, work="slow", outer_cancel=True)
    assert case.cleaned.is_set()
    assert not any(row["kind"] == "research_interrupted" for row in case.events)


@pytest.mark.asyncio
async def test_adapter_budget_string_is_a_native_failure_not_confirmed_budget(tmp_path, monkeypatch):
    case = await exercise(tmp_path, monkeypatch, exit_kind="budget_string")
    event = next(row for row in case.events if row["kind"] == "research_interrupted")
    assert event["reason"] == "native_execution_failed"
    assert event["native_error_type"] == "research_phase_budget_exhausted"


@pytest.mark.asyncio
async def test_provider_payload_is_not_copied_into_failure_receipt(tmp_path, monkeypatch):
    case = await exercise(tmp_path, monkeypatch, exit_kind="unsafe_error")
    event = next(row for row in case.events if row["kind"] == "research_interrupted")
    assert "PRIVATE_FIXTURE_TOKEN" not in json.dumps(event)
    assert event["native_status"] == "unknown" and event["native_error_type"] == "RuntimeError"


@pytest.mark.asyncio
async def test_valid_final_text_does_not_become_partial(tmp_path, monkeypatch):
    case = await exercise(tmp_path, monkeypatch, exit_kind="success")
    assert any(row["kind"] == "completed" for row in case.events)
    assert not any(row["kind"] == "research_interrupted" for row in case.events)


@pytest.mark.asyncio
@pytest.mark.parametrize("work", ["queue_admission_value", "queue_admission_http", "queue_waiting",
                                  "queue_running", "queue_policy_rejection", "queue_startup"])
async def test_unstarted_queue_request_is_not_scientific_attempt(tmp_path, monkeypatch, work):
    case = await exercise(tmp_path, monkeypatch, work=work)
    assert not any(row["kind"] == "research_interrupted" for row in case.events)


@pytest.mark.asyncio
async def test_queue_execution_requires_actual_terminal_value(tmp_path, monkeypatch):
    case = await exercise(tmp_path, monkeypatch, work="queue_terminal")
    event = next(row for row in case.events if row["kind"] == "research_interrupted")
    assert event["scientific_tool_attempts"] == [{"tool": "calculate_physchem_from_sequence", "call_id":
        event["scientific_tool_attempts"][0]["call_id"], "status": "failed", "job_id": "host-job"}]
