"""A failed native continuation retains earlier host attempts for strict SC review."""
import asyncio
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from test_research_phase_budget import local_queue  # noqa: F401

from agent.harness import autonomous_research as ar
from agent.harness import web_native_worker as worker
from agent.harness.contracts import Profile
from agent.harness.research_handoff import NativeTurnFailure

SOURCE = "def fit(training, seed):\n    pass\ndef predict(model, inputs):\n    return {'value': 0.0}\n"


async def exercise(tmp_path, monkeypatch, *, prior="attempts", failure="runtime_error"):
    monkeypatch.setattr("mcp_tools.scheduler.catalog.native_tools", lambda: {})
    relays = []

    class Relay:
        base_url = "http://unused"
        token = "fixture"

        def __init__(self, **kwargs):
            self.policy = kwargs["policy"]
            self.calls = 0
            Path(kwargs["output"]).touch()
            relays.append(self)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

    monkeypatch.setattr(ar, "EvidenceGateway", Relay)
    monkeypatch.setattr(ar, "audit_requests", lambda _: {"passed": True, "model_calls": relays[-1].calls})
    profile = Profile.from_dict({
        "mode": "science_expert", "harness": "expert-kimi-code",
        "model": {"provider": "dmx", "model": "glm-4-flash", "base_url": "https://www.dmxapi.cn/v1"},
        "budget": {"model_calls": 16, "tool_calls": 16, "wall_seconds": 120,
                   "output_tokens": 4096, "thinking": "disabled"},
    })
    store = ar.create(tmp_path / "goal", profile, "Inspect supplied synthetic measurements",
                      {"max_cycles": 1, "model_calls": 16, "tool_calls": 16, "wall_seconds": 120})
    data = "id,group,x,target\n" + "".join(f"{i},g{i},{i},{2*i+1}\n" for i in range(40))
    source = store.root / "workspace/inputs/data.csv"
    source.parent.mkdir()
    source.write_text(data)
    store.mutate(lambda state: state["progress"].update(inputs=[{
        "path": "inputs/data.csv", "name": "data.csv", "sha256": hashlib.sha256(data.encode()).hexdigest(),
    }]))
    roles, contexts, turns, terminal = [], [], [], []

    async def consult(role, context):
        roles.append(role)
        relays[-1].calls += 1
        if role == "PI":
            return "Inspect measurements and retain unsuccessful experimental attempts for review."
        if role == "CB":
            return json.dumps({"summary": "Inspect measurements", "deliverable": "analysis", "needs_input": [],
                "steps": [{"id": "inspect", "title": "Inspect measurements", "acceptance": "Actual computation",
                           "verification": "execution"}]})
        assert role == "SC"
        contexts.append(context)
        assert not list(store.root.rglob("connection.json"))
        return json.dumps({"decision": "needs_revision", "summary": "Experimental attempts require correction.",
            "issues": ["The native continuation failed and no final checkpoint exists."],
            "evidence_ids": [], "next_actions": ["Repair the actual failed implementation."]})

    class Adapter:
        async def run(self, ctx):
            turn = len(turns)
            turns.append(ctx)
            connection = json.loads((ctx.task.capability / "connection.json").read_text())
            trace = ctx.task.output / "events.jsonl"
            trace.touch()
            rows = []

            def record(row):
                rows.append(row)
                trace.write_text("".join(json.dumps(item) + "\n" for item in rows))

            async with httpx.AsyncClient(trust_env=False, timeout=15) as client:
                async def invoke(name, arguments):
                    identity = str(len(rows))
                    record({"kind": "tool_call_start", "tool_call_id": identity,
                            "tool_name": "mcp__venusfactory__" + name, "args": arguments})
                    response = await client.post(connection["url"],
                        headers={"Authorization": "Bearer " + connection["token"]},
                        json={"jsonrpc": "2.0", "id": identity, "method": "tools/call",
                              "params": {"name": name, "arguments": arguments}})
                    result = response.json()["result"]
                    record({"kind": "tool_result", "tool_call_id": identity, "is_error": result["isError"],
                            "output": "\n".join(item["text"] for item in result["content"])})
                    return result

                if turn == 0:
                    relays[-1].calls += 7
                    if prior == "attempts":
                        await invoke("describe_tool", {"name": "prepare_research_dataset"})
                        prepared = await invoke("call_tool", {"name": "prepare_research_dataset", "arguments": {
                            "data_file": "inputs/data.csv", "target_column": "target", "input_columns": ["x"],
                            "group_column": "group", "provenance": "Synthetic software fixture",
                            "grouping": "Independent synthetic groups"}})
                        assert not prepared["isError"], prepared
                        prepared = json.loads(prepared["content"][0]["text"])
                        assert not (await invoke("write_session_file", {"path": "src/model.py", "content": SOURCE}))["isError"]
                        await invoke("describe_tool", {"name": "train_research_model"})
                        for _ in range(2):
                            result = await invoke("call_tool", {"name": "train_research_model", "arguments": {
                                "name": "Synthetic", "source_file": "src/model.py", "evaluation_file": prepared["evaluation_file"],
                                "hypothesis": "Exercise real failed preflight receipts"}})
                            assert result["isError"] and "TrainingAttemptError" in result["content"][0]["text"]
                    elif prior == "forged":
                        record({"kind": "tool_call_start", "tool_call_id": "invented",
                                "tool_name": "mcp__venusfactory__call_tool", "args": {"name": "train_research_model", "arguments": {}}})
                        record({"kind": "tool_result", "tool_call_id": "invented", "is_error": False,
                                "output": "Invented model output is not a host scientific attempt"})
                    else:
                        await invoke("write_session_file", {"path": "notes.md", "content": "No scientific execution"})
                    return {"status": "completed", "text": "Preliminary work needs continuation.",
                            "scientific_tool_attempts": [{"tool": "train_research_model", "call_id": "forged", "status": "completed"}]}
                for _ in range(4):
                    assert not (await invoke("describe_tool", {"name": "train_research_model"}))["isError"]
                relays[-1].calls += 6 if failure == "insufficient_budget" else 4
                if failure == "interaction":
                    assert not (await invoke("request_confirmation", {
                        "title": "Required decision", "message": "Choose the required scope before continuing."}))["isError"]
                    return {"status": "runtime_error", "end_reason": "failed"}
                if failure in {"provider_authentication_failed", "mcp_startup_failed", "step_interrupted", "unknown_code", "provider_empty_response"}:
                    return {"status": "runtime_error", "error_code": failure,
                            "error": "Synthetic structured runtime diagnostic"}
                if failure == "unclassified_result_exception":
                    return {"status": "runtime_error", "error": "RuntimeError: unclassified setup failure"}
                if failure == "cancel":
                    raise asyncio.CancelledError()
                if failure == "permission":
                    raise PermissionError("Access denied")
                if failure == "exception":
                    raise RuntimeError("Unclassified native exception")
                if failure == "configuration":
                    return {"status": "failed", "error_type": "configuration_error"}
                if failure == "alternate_shape":
                    return {"status": "failed", "error_type": "runtime_error"}
                return {"status": "runtime_error", "end_reason": "failed"}

    monkeypatch.setattr(worker, "builtin_registry", lambda: SimpleNamespace(resolve=lambda *args: (Adapter(), None)))
    actual_run = worker.run

    async def observed_run(request, **kwargs):
        sink = kwargs["event_sink"]

        def record(event):
            if event["kind"] in {"completed", "research_interrupted"}:
                terminal.append(event)
            sink(event)

        return await actual_run(request, **{**kwargs, "event_sink": record})

    monkeypatch.setattr(worker, "run", observed_run)
    runner = ar.AutonomousResearch(store, "fixture", {}, consult=consult, tools=[])
    if failure == "cancel":
        with pytest.raises(asyncio.CancelledError):
            await runner.run()
        state = None
    else:
        state = await runner.run()
    assert len(turns) == 2
    assert not list(store.root.rglob("connection.json"))
    return SimpleNamespace(store=store, roles=roles, contexts=contexts, state=state, terminal=terminal,
                           calls=relays[-1].calls)


@pytest.mark.asyncio
@pytest.mark.parametrize("shape", ["runtime_error", "alternate_shape", "provider_empty_response"])
async def test_followup_failure_reviews_earlier_real_failed_preflights(tmp_path, monkeypatch, shape):
    case = await exercise(tmp_path, monkeypatch, failure=shape)
    assert case.roles == ["PI", "CB", "SC"]
    assert case.calls == 14  # PI/CB + 7 first native + 4 continuation + SC; no expanded budget.
    assert len(case.terminal) == 1  # The failed worker never emits completed or a handoff by itself.
    first = case.terminal[0]
    interrupted = json.loads((case.store.root / "episodes/cycle_000/execution_retry/execution_failure_interruption.json").read_text())
    assert first["kind"] == "completed" and first["event_audit"]["passed"]
    failed = [a for a in first["scientific_tool_attempts"] if a["tool"] == "train_research_model"]
    assert len(failed) == 2 and all(a["status"] == "failed" for a in failed)
    assert all(a["call_id"] != "forged" for a in first["scientific_tool_attempts"])
    assert interrupted["kind"] == "research_interrupted" and interrupted["research_complete"] is False
    assert interrupted["failure_origin"] == "execution_followup"
    assert interrupted["scientific_tool_attempts"] == first["scientific_tool_attempts"]
    assert interrupted["event_audit"]["tool_calls"] == 4
    assert interrupted["previous_execution_audit"]["tool_errors"] == 2
    context = case.contexts[0]
    assert context["execution"]["checkpoint"] is None
    assert context["execution"]["interruption"]["cleanup_complete"] is True
    assert context["execution"]["interruption"].get("native_error_code") == (
        "provider_empty_response" if shape == "provider_empty_response" else None)
    assert any("unfinished placeholder" in row["error"] for row in context["execution"]["failures"])
    receipts = list((case.store.root / "development/training").glob("*/receipt.json"))
    assert len(receipts) == 2
    assert all(json.loads(p.read_text())["stage"] == "preflight" for p in receipts)
    assert case.state["cycles"][0]["status"] == "needs_revision"
    assert not case.state["progress"]["host_review"]["passed"]
    assert not case.state["progress"]["research_completed"]
    assert not case.state["progress"].get("deliveries") and not case.state["progress"].get("capabilities")
    assert not (case.store.root / "episodes/cycle_000/execution_retry/native/checkpoint_handoff.json").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("prior", ["none", "forged"])
async def test_followup_cannot_borrow_model_declared_or_trace_only_attempts(tmp_path, monkeypatch, prior):
    case = await exercise(tmp_path, monkeypatch, prior=prior)
    assert case.roles == ["PI", "CB"]
    assert case.terminal[0]["scientific_tool_attempts"] == []
    assert len(case.terminal) == 1
    assert not case.state["progress"].get("host_review")
    receipt = json.loads((case.store.root / "episodes/cycle_000/execution_retry/native/native_failure.json").read_text())
    assert receipt["cleanup_complete"] and "scientific_tool_attempts" not in receipt


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["cancel", "permission", "exception", "configuration", "insufficient_budget",
    "provider_authentication_failed", "mcp_startup_failed", "step_interrupted", "unknown_code",
    "unclassified_result_exception", "interaction"])
async def test_followup_preserves_other_failure_and_budget_boundaries(tmp_path, monkeypatch, failure):
    case = await exercise(tmp_path, monkeypatch, failure=failure)
    assert case.roles == ["PI", "CB"]
    assert len(case.terminal) == 1
    assert not (case.store.root / "episodes/cycle_000/sc_evidence.json").exists()
    if failure == "interaction":
        assert case.state["progress"]["pending_interaction"]["kind"] == "confirmation"
    assert not (case.store.root / "episodes/cycle_000/execution_retry/execution_failure_interruption.json").exists()


def test_native_failure_has_no_cleanup_evidence_by_default():
    failure = NativeTurnFailure()
    assert str(failure) == "runtime_error" and failure.interruption is None
