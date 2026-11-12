"""SC artifact previews preserve the exact bytes used by host hash binding."""
import hashlib
import json
from pathlib import Path

import pytest
from test_analysis_report_review import run_capture

from agent.harness import Profile
from agent.harness import autonomous_research as ar
from agent.harness.analysis_report_review import POLICY as ANALYSIS_POLICY
from agent.harness.research_cli_inputs import prepare_inputs, stage_inputs


@pytest.mark.asyncio
@pytest.mark.parametrize("newline", ["\n", "\r\n"])
async def test_report_preview_preserves_hash_bound_line_endings(tmp_path, monkeypatch, newline):
    class Relay:
        calls = 0
        base_url = "http://unused"
        token = "fixture"

        def __init__(self, **kwargs):
            Path(kwargs["output"]).touch()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

    monkeypatch.setattr(ar, "EvidenceGateway", Relay)
    monkeypatch.setattr(ar, "audit_requests", lambda path: {"passed": True, "model_calls": 4})
    profile = Profile.from_dict({
        "mode": "science_expert", "harness": "expert-deepseek-harness",
        "model": {"provider": "dmx", "model": "glm-4-flash", "base_url": "https://www.dmxapi.cn/v1"},
        "budget": {"model_calls": 12, "tool_calls": 12, "wall_seconds": 120,
                   "output_tokens": 4096, "thinking": "disabled"},
    })
    store = ar.create(tmp_path / "goal", profile, "Compute and report the length of the supplied peptide AAAA",
                      {"max_cycles": 1, "model_calls": 12, "tool_calls": 12, "wall_seconds": 120})
    report = ("# Result" + newline + "Length: 4" + newline).encode("utf-8")
    original = tmp_path / "original.fasta"
    original.write_text(">peptide\nAAAA\n")
    with prepare_inputs([original]) as prepared:
        stage_inputs(store, prepared)
    seen = []
    captured_task = []

    async def consult(role, context):
        if role == "PI":
            return "Calculate peptide lengths."
        if role == "CB":
            return json.dumps({"summary": "Compute lengths", "steps": [
                {"id": "calculate", "title": "Compute lengths", "acceptance": "Save calculated values",
                 "verification": "execution"}], "needs_input": []})
        preview = context["artifact_previews"][0]
        assert preview["content"].encode("utf-8") == report
        assert hashlib.sha256(preview["content"].encode("utf-8")).hexdigest() == preview["sha256"]
        assert context["scientific_acceptance"]["analysis_review_policy"] == ANALYSIS_POLICY
        assert context["analysis_review_state"]["stage"] == "final"
        seen.append(preview)
        return json.dumps({"decision": "ready_for_host_validation", "summary": "Computed report reviewed",
            "issues": [], "claims": [{"id": "length", "task_id": captured_task[0],
                "result_pointer": "/result/data/length", "report_path": "report.md",
                "report_quote": "Length: 4", "display_value": "4"}],
            "step_assessments": [{"step_id": "calculate", "verdict": "supported",
                "reason": "The recorded length matches the hash-bound report.", "claim_ids": ["length"]}],
            "input_assessments": [{"task_id": captured_task[0], "kind": "registered_input",
                "addresses_objective": True, "reason": "The supplied peptide sequence matches exactly.",
                "source_task_ids": []}]})

    async def execute(request, service, events):
        await service.write_research_file("calculate.py", "from pathlib import Path\nPath('report.md').write_bytes(" + repr(report) + ")\n")
        assert (await service.run_research_python("calculate.py", 10))["exit_code"] == 0
        captured_task.append(run_capture(service, {"sequence": "AAAA"}, {"data": {"length": len("AAAA")}}))
        await service.submit_research_checkpoint("Computed lengths", ["calculate"],
            ["report.md", "analysis/" + captured_task[0] + ".json"], [])
        events.extend([{"kind": "tool_call_start"}, {"kind": "completed", "text": "Report written", "event_audit": {"passed": True}}])

    state = await ar.AutonomousResearch(store, "fixture", {}, consult=consult, executor=execute, tools=[]).run()
    assert len(seen) == 1
    assert state["status"] == "completed", state
    assert state["progress"]["scientific_acceptance"]["passed"] is True
    assert state["progress"]["plan"]["steps"][0]["status"] == "completed"
    assert (store.root / "workspace/report.md").read_bytes() == report
