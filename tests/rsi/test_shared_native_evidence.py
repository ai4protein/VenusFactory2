"""Maximum admitted batch through real MCP fits and the native event recorder.

The event transport is synthetic and the source program is a fixed fixture.
This is an offline integration test, never autonomous scientific evidence.
"""

import json

import pytest
from test_runtime import FakeClient
from test_shared_model_program import CODE, client_for, make_task, prepare

from agent.harness.events import audit_trace
from agent.kimi_client import KimiEvent
from agent.rsi.protein_model import digest
from agent.rsi.runtime import run_task
from agent.rsi.shared_goal_history import project
from agent.rsi.shared_model_contracts import resolve_shared_result
from agent.rsi.shared_model_display import LEGACY_POLICY, MAX_DISPLAY_CHARACTERS
from agent.rsi.shared_model_display import POLICY as DISPLAY_POLICY
from agent.rsi.shared_model_publication import _operation_receipts
from agent.rsi.storage import atomic_json
from agent.rsi.tool_contracts import validate_generated_result


@pytest.mark.asyncio
@pytest.mark.parametrize("presentation", [None, LEGACY_POLICY, DISPLAY_POLICY])
async def test_32_target_feedback_remains_complete_through_native_recording_and_audit(tmp_path, presentation):
    tasks = []
    for index in range(32):
        task, _, _ = make_task(tmp_path / "inputs" / str(index), protein=f"Synthetic plant protein {index}")
        task.update(id=f"synthetic_{index}", plugin_id="generated_context.v2")
        task["scope"]["interface_version"] = "generated_context.v2"
        tasks.append(task)
    root = tmp_path / "goal"
    folder = root / "episodes/cycle_000"
    work, cap, versions, version = prepare(
        folder, tasks, training_baseline_feedback="group_baselines.v2", tool_presentation=presentation
    )
    allowed = frozenset({"inspect_model_code_task", "evaluate_model_code", "submit_model_code"})
    originals = {}

    class ScientificStream(FakeClient):
        def json(self):
            return {"data": {"tools": [{"name": "mcp__venusfactory__" + name} for name in sorted(allowed)]}}

        async def stream_session_events(self, sid):
            yield KimiEvent(kind="subscribed")
            assert self.submitted
            async with client_for(work, cap) as client:
                operations = [
                    ("inspect_model_code_task", {}),
                    ("evaluate_model_code", {"source_code": CODE, "rationale": "32-target fixture"}),
                    ("submit_model_code", None),
                ]
                trial = None
                for index, (name, args) in enumerate(operations):
                    if args is None:
                        args = {"source_sha256": trial["source_sha256"], "rationale": "independent fixture fits"}
                    yield KimiEvent(
                        kind="tool_call_start",
                        tool_name="mcp__venusfactory__" + name,
                        tool_call_id=str(index),
                        tool_args=args,
                    )
                    returned = await client.call_tool(name, args)
                    result = returned.data
                    validate_generated_result(name, result, workspace=work, capability=cap)
                    if name == "evaluate_model_code":
                        trial = result
                    originals[name] = returned.content[0].text
                    assert json.loads(originals[name]) == result
                    if presentation is not None:
                        assert len(originals[name]) <= MAX_DISPLAY_CHARACTERS
                    yield KimiEvent(
                        kind="tool_result", tool_call_id=str(index), tool_output=originals[name], is_error=False
                    )
            yield KimiEvent(kind="turn_ended", text="completed")

    result = await run_task(
        ScientificStream([]),
        prompt="Synthetic integration check",
        system_prompt="No external model",
        model="m",
        output_dir=folder / "execution",
        allowed_mcp_tools=allowed,
        timeout_s=300,
        max_tool_calls=3,
    )
    assert result["status"] == "completed" and result["tool_calls"] == 3
    trace = folder / "execution/events.jsonl"
    events = [json.loads(line) for line in trace.read_text().splitlines()]
    names = {
        row["tool_call_id"]: row["tool_name"].removeprefix("mcp__venusfactory__")
        for row in events
        if row["kind"] == "tool_call_start"
    }
    if presentation is None:
        assert len(originals["inspect_model_code_task"]) > 30000
        assert len(originals["evaluate_model_code"]) > 30000
    else:
        for name in ["inspect_model_code_task", "evaluate_model_code"]:
            raw = json.loads(originals[name])
            assert raw["schema"] == presentation
            full = resolve_shared_result(name, raw, workspace=work, capability=cap)
            assert len(originals[name]) < len(json.dumps(full, indent=2))
    for row in events:
        if row["kind"] == "tool_result":
            name = names[row["tool_call_id"]]
            assert row["output"] == originals[name]
            validate_generated_result(name, json.loads(row["output"]), workspace=work, capability=cap)
    assert audit_trace(trace, allowed, 3)["passed"]
    receipts = _operation_receipts(folder, cap)
    assert receipts["submit_model_code"][0]["candidate_count"] == 32
    assert len(receipts["evaluate_model_code"][0]["results"]) == 32
    versions.verify(version)
    cycle = {
        "index": 0,
        "phase": "develop",
        "status": "completed",
        "execution_audit_sha256": digest({"passed": True, "synthetic_executor": True}),
    }
    atomic_json(folder / "lifecycle_execution_audit.json", {"passed": True, "synthetic_executor": True})
    atomic_json(root / "cycle_000.json", cycle)
    history = project(root, {"cycles": [cycle]}, [tasks[0], tasks[-1]])
    assert len(history["attempts"]) == 1 and history["attempts"][0]["source_code"] == CODE
    assert [row["task_id"] for row in history["attempts"][0]["results"]] == [tasks[0]["id"], tasks[-1]["id"]]
    if presentation is not None:
        raw = json.loads(originals["inspect_model_code_task"])
        changed = json.loads(originals["inspect_model_code_task"])
        changed["content"]["targets"].pop()
        with pytest.raises(ValueError, match="differs from its complete record"):
            validate_generated_result("inspect_model_code_task", changed, workspace=work, capability=cap)
        artifact = work / raw["artifact"]
        saved = artifact.read_bytes()
        full = json.loads(saved)
        full["targets"].pop()
        atomic_json(artifact, full)
        try:
            with pytest.raises(ValueError, match="record changed"):
                validate_generated_result("inspect_model_code_task", raw, workspace=work, capability=cap)
        finally:
            artifact.write_bytes(saved)
        changed = {**raw, "result_sha256": "../outside"}
        with pytest.raises(ValueError, match="history hash"):
            validate_generated_result("inspect_model_code_task", changed, workspace=work, capability=cap)
    # A clipped scientific result is still inadmissible, even if its files exist.
    original = trace.read_bytes()
    for row in events:
        if row["kind"] == "tool_result" and names[row["tool_call_id"]] == "evaluate_model_code":
            row["output"] = row["output"][: len(row["output"]) // 2]
    trace.write_text("".join(json.dumps(row) + "\n" for row in events))
    try:
        with pytest.raises(ValueError):
            _operation_receipts(folder, cap)
        with pytest.raises(ValueError):
            project(root, {"cycles": [cycle]}, tasks)
    finally:
        trace.write_bytes(original)
