"""SC artifact previews preserve the exact bytes used by host hash binding."""
import hashlib
import json
from pathlib import Path

import pytest

from agent.harness import Profile
from agent.harness import autonomous_research as ar


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
    store = ar.create(tmp_path / "goal", profile, "Compute and report peptide lengths",
                      {"max_cycles": 1, "model_calls": 12, "tool_calls": 12, "wall_seconds": 120})
    report = ("# Result" + newline + "Length: 4" + newline).encode("utf-8")
    seen = []

    async def consult(role, context):
        if role == "PI":
            return "Calculate peptide lengths."
        if role == "CB":
            return json.dumps({"summary": "Compute lengths", "steps": [
                {"id": "calculate", "title": "Compute lengths", "acceptance": "Save calculated values"}], "needs_input": []})
        preview = context["artifact_previews"][0]
        assert preview["content"].encode("utf-8") == report
        assert hashlib.sha256(preview["content"].encode("utf-8")).hexdigest() == preview["sha256"]
        seen.append(preview)
        return json.dumps({"decision": "ready_for_host_validation", "summary": "Computed report reviewed", "issues": []})

    async def execute(request, service, events):
        await service.write_research_file("calculate.py", "from pathlib import Path\nPath('report.md').write_bytes(" + repr(report) + ")\n")
        assert (await service.run_research_python("calculate.py", 10))["exit_code"] == 0
        await service.submit_research_checkpoint("Computed lengths", ["calculate"], ["report.md"], [])
        events.extend([{"kind": "tool_call_start"}, {"kind": "completed", "text": "Report written", "event_audit": {"passed": True}}])

    state = await ar.AutonomousResearch(store, "fixture", {}, consult=consult, executor=execute, tools=[]).run()
    assert len(seen) == 1
    assert state["status"] == "completed", state
