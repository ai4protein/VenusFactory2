"""Real isolated MCP plus independently checked, simulated session transcripts.

No tests in this module call an external model or establish autonomous authorship.
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastmcp import Client
from fastmcp.client.transports import StdioTransport

from agent.harness import Budget, ModelBinding, Profile
from agent.harness import native_expert_team as team
from agent.harness.host import RuntimeContext
from agent.rsi.langgraph_tools import mcp_command
from agent.rsi.paper_step import PaperStepLifecycle, digest
from agent.rsi.paper_step_sessions import accept_development, accept_reuse, phase_task, prepare_phase
from agent.rsi.paper_step_tool_service import TOOLS, PaperStepTools, source_hash
from agent.rsi.reproduction_tool_contracts import validate_reproduction_result
from agent.rsi.storage import atomic_json, sha256_file

CODE = "def reproduce(inputs):\n    return {'sum': sum(inputs)}\n"


@pytest.fixture
def lifecycle(tmp_path):
    ref = tmp_path / "host_reference.txt"
    ref.write_text("independent sum fixture")
    d = {
        "schema": 1,
        "paper": {"title": "Numerical fixture", "doi": "test:fixture"},
        "scope": {"covered_steps": ["summation"], "excluded_steps": ["biological results"]},
        "interface": {"id": "sum.v1", "description": "list of numbers -> sum object"},
        "references": {str(ref): sha256_file(ref)},
        "cases": {
            "development": [{"id": "dev", "inputs": [1, 2], "expected": {"sum": 3}}],
            "acceptance": [{"id": "accept", "inputs": [3, 5], "expected": {"sum": 8}}],
            "reuse": [{"id": "reuse", "inputs": [7, 11], "expected": {"sum": 18}}],
        },
        "budget": {"candidates": 2, "operation_seconds": 30},
        "tolerance": {"atol": 1e-12, "rtol": 0},
    }
    return PaperStepLifecycle(tmp_path / "host_private", d)


def invoke_chain(task, *, phase, capability_id=None, code=CODE):
    service = PaperStepTools(task.workspace, task.capability)
    calls = [("inspect_paper_step_task", {}, service.inspect)]
    if phase == "develop":
        calls += [
            (
                "evaluate_paper_step",
                {"source_code": code, "rationale": "Numerical fixture"},
                lambda: service.evaluate(code, "Numerical fixture"),
            ),
            ("submit_paper_step", {"source_sha256": source_hash(code)}, lambda: service.submit(source_hash(code))),
        ]
    else:
        calls += [
            ("discover_paper_steps", {}, service.discover),
            ("execute_registered_paper_step", {"capability_id": capability_id}, lambda: service.reuse(capability_id)),
        ]
    events = []
    for i, (name, args, call) in enumerate(calls):
        cid = "call-" + str(i)
        events.append(
            {"kind": "tool_call_start", "tool_call_id": cid, "tool_name": "mcp__venusfactory__" + name, "args": args}
        )
        value = call()
        validate_reproduction_result(name, value, workspace=task.workspace, capability=task.capability)
        events.append({"kind": "tool_result", "tool_call_id": cid, "is_error": False, "output": json.dumps(value)})
    return events


def simulated_session(folder, *, phase, capability_id=None, session_id="simulated-develop", profile=None, code=CODE):
    task = phase_task(folder)
    before = task.binding()
    task.output.mkdir()
    events = invoke_chain(task, phase=phase, capability_id=capability_id, code=code)
    profile = profile or Profile(
        "science_agent",
        "kimi-code",
        ModelBinding("test_fixture", "simulated", "https://example.org/v1"),
        Budget(tool_calls=8),
    )
    lock = {"profile": profile.public(), "task": before}
    lock["sha256"] = digest(lock)
    atomic_json(task.output / "profile.lock.json", lock)
    atomic_json(
        task.output / "run.json",
        {"status": "completed", "session_id": session_id, "profile_sha256": lock["sha256"], "audit": {"passed": True}},
    )
    (task.output / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    (task.output / "model_requests.jsonl").write_text(
        json.dumps({"event": "request", "call": 1})
        + "\n"
        + json.dumps({"event": "response", "call": 1, "status": 200, "completed": True})
        + "\n"
    )
    return task


@pytest.mark.asyncio
async def test_real_isolated_mcp_entry_and_nested_code_worker(lifecycle, tmp_path):
    folder = prepare_phase(lifecycle, tmp_path / "dev", "develop")
    task = phase_task(folder)
    transport = StdioTransport(
        command="bwrap",
        args=mcp_command(task.workspace, task.capability, Path(sys.prefix)),
        keep_alive=False,
        cwd=str(task.workspace),
    )
    async with Client(transport) as client:
        assert {t.name for t in await client.list_tools()} == TOOLS
        for name, args in [
            ("inspect_paper_step_task", {}),
            ("evaluate_paper_step", {"source_code": CODE, "rationale": "test isolated execution"}),
            ("submit_paper_step", {"source_sha256": source_hash(CODE)}),
        ]:
            answer = await client.call_tool(name, args)
            assert not answer.is_error
            value = json.loads("\n".join(x.text for x in answer.content if hasattr(x, "text")))
            validate_reproduction_result(name, value, workspace=task.workspace, capability=task.capability)
    assert (task.workspace / "step_candidate.json").exists()
    assert "host_reference.txt" not in (task.capability / "step_task.json").read_text()
    assert "acceptance" not in json.dumps(
        json.loads((task.capability / "step_task.json").read_text())["public_task"].get("cases", {})
    )


def test_development_failure_repair_and_restart_budget(lifecycle, tmp_path):
    task = phase_task(prepare_phase(lifecycle, tmp_path / "dev", "develop"))
    service = PaperStepTools(task.workspace, task.capability)
    failure = service.evaluate("def reproduce(inputs): pass", "placeholder must fail")
    assert failure["status"] == "development_failed" and failure["attempts_remaining"] == 1
    validate_reproduction_result("evaluate_paper_step", failure, workspace=task.workspace, capability=task.capability)
    service = PaperStepTools(task.workspace, task.capability)
    assert service.evaluate(CODE, "repair")["development_passed"]
    with pytest.raises(ValueError, match="budget"):
        service.evaluate(CODE, "third attempt")
    assert service.submit(source_hash(CODE))["registered"] is False


def test_wrong_outputs_and_changed_receipts_are_rejected(lifecycle, tmp_path):
    task = phase_task(prepare_phase(lifecycle, tmp_path / "dev", "develop"))
    service = PaperStepTools(task.workspace, task.capability)
    wrong = "def reproduce(inputs): return {'sum':0}"
    value = service.evaluate(wrong, "incorrect computation")
    assert value["development_passed"] is False
    with pytest.raises(ValueError, match="must pass"):
        service.submit(source_hash(wrong))
    with pytest.raises(ValueError, match="differs"):
        validate_reproduction_result(
            "evaluate_paper_step",
            {**value, "development_passed": True},
            workspace=task.workspace,
            capability=task.capability,
        )


def test_host_acceptance_fresh_session_and_external_witness_reuse(lifecycle, tmp_path):
    develop = prepare_phase(lifecycle, tmp_path / "dev", "develop")
    simulated_session(develop, phase="develop")
    accepted = accept_development(lifecycle, develop)
    assert accepted["assessment"]["passed"]
    assert accepted["source_provenance"]["provider"] == "test_fixture"
    assert not lifecycle.capability(accepted["capability_id"])["agent_authorship_verified"]
    with pytest.raises(ValueError, match="already reserved"):
        accept_development(lifecycle, develop)
    reuse = prepare_phase(
        lifecycle, tmp_path / "reuse", "reuse", capability_id=accepted["capability_id"], case_id="reuse"
    )
    simulated_session(reuse, phase="reuse", capability_id=accepted["capability_id"], session_id="simulated-reuse")
    with pytest.raises(ValueError, match="fresh session"):
        accept_reuse(lifecycle, reuse, development_session="simulated-reuse")
    result = accept_reuse(lifecycle, reuse, development_session="simulated-develop")
    assert result["verification"]["passed"]
    assert lifecycle.verify_reuse(accepted["capability_id"], "reuse")["passed"]


@pytest.mark.parametrize("tamper", ["source_args", "missing_requests", "submission_args"])
def test_source_or_gateway_trace_mismatch_cannot_register(lifecycle, tmp_path, tamper):
    folder = prepare_phase(lifecycle, tmp_path / "dev", "develop")
    task = simulated_session(folder, phase="develop")
    if tamper == "missing_requests":
        (task.output / "model_requests.jsonl").unlink()
    else:
        events = [json.loads(x) for x in (task.output / "events.jsonl").read_text().splitlines()]
        target = "evaluate_paper_step" if tamper == "source_args" else "submit_paper_step"
        for event in events:
            if event.get("kind") == "tool_call_start" and event["tool_name"].endswith(target):
                event["args"] = (
                    {"source_code": "def reproduce(x): return 0"}
                    if tamper == "source_args"
                    else {"source_sha256": "f" * 64}
                )
        (task.output / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    with pytest.raises(ValueError):
        accept_development(lifecycle, folder)
    assert not (lifecycle.root / "registered").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("harness", ["expert-kimi-code", "expert-deepseek-harness"])
async def test_four_role_contract_runs_real_paper_tools_with_simulated_llm(lifecycle, tmp_path, monkeypatch, harness):
    folder = prepare_phase(lifecycle, tmp_path / "dev", "develop")
    task = phase_task(folder)
    task.output.mkdir()
    profile = Profile(
        "science_expert",
        harness,
        ModelBinding("test_fixture", "simulated", "https://example.org/v1"),
        Budget(wall_seconds=30, model_calls=12, tool_calls=8),
    )
    ctx = RuntimeContext(profile, task, SimpleNamespace(calls=0), {})
    roles = []

    async def role(context, name, evidence, folder):
        roles.append(name)
        context.relay.calls += 1
        return (
            json.dumps({"decision": "ready_for_host_validation", "summary": "fixture", "issues": []})
            if name == "SC"
            else "Use the declared paper computation tools"
        )

    class Native:
        async def run(self, context):
            roles.append("MLS")
            context.relay.calls += 2
            events = invoke_chain(context.task, phase="develop")
            (context.task.output / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
            return {
                "status": "completed",
                "session_id": "simulated-native",
                "tool_calls": 3,
                "text": "Fixture submitted",
            }

    monkeypatch.setattr(team, "role_completion", role)
    monkeypatch.setattr(team, "native_adapter", lambda native, runtimes: Native())
    result = await team.NativeExpertTeamAdapter({}).run(ctx)
    assert result["status"] == "completed" and result["four_role_path_completed"]
    assert roles == ["PI", "CB", "MLS", "SC"]
    assert result["tool_calls"] == 3
