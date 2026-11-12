"""A bounded native interruption reaches SC without becoming successful research."""
import json
from pathlib import Path

import pytest

from agent.harness import Profile
from agent.harness import autonomous_research as ar
from agent.harness.analysis_interim_review import POLICY as INTERIM_POLICY


def setup(tmp_path, monkeypatch):
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
    return ar.create(tmp_path / "goal", profile, "Compute lengths for the supplied peptides",
                     {"max_cycles": 1, "model_calls": 12, "tool_calls": 12, "wall_seconds": 120})


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", ["native_model_request_limit", "native_execution_failed"])
@pytest.mark.parametrize("audit_passed", [True, False])
@pytest.mark.parametrize("critic_ready", [True, False])
async def test_native_failure_reviews_partial_evidence_but_cannot_accept(tmp_path, monkeypatch, reason, audit_passed, critic_ready):
    store = setup(tmp_path, monkeypatch)
    roles = []

    async def consult(role, context):
        roles.append(role)
        if role == "PI":
            return "Compute lengths from the supplied sequences."
        if role == "CB":
            return json.dumps({"summary": "Compute lengths", "deliverable": "analysis", "needs_input": [],
                "steps": [{"id": "calculate", "title": "Compute lengths", "acceptance": "Save computed lengths"}]})
        assert context["execution"]["interruption"]["reason"] == reason
        assert context["execution"]["interruption"]["event_audit"]["passed"] is audit_passed
        assert context["execution"]["receipts"]
        assert context["scientific_acceptance"]["analysis_review_policy"] == INTERIM_POLICY
        assert context["analysis_review_state"]["report_paths"] == ["report.md"]
        assert context["analysis_review_state"]["reasons"] == ["execution_interrupted"]
        premature_ready = critic_ready and roles.count("SC") == 1
        if critic_ready and not premature_ready:
            assert "needs_revision" in context["format_repair"]["validation_error"]
        else:
            assert "format_repair" not in context
        return json.dumps({"decision": "ready_for_host_validation" if premature_ready else "needs_revision",
            "summary": "Partial computations inspected; the interrupted execution requires revision.",
            "issues": ["Execution was interrupted despite saved work."], "evidence_ids": [],
            "next_actions": ["Resolve the interruption before submitting a final analysis checkpoint."]})

    async def execute(request, service, events):
        await service.write_research_file("calculate.py",
            'import json\nfrom pathlib import Path\n'
            'json.dump({"lengths":[4,2]},open("lengths.json","w"))\n'
            'Path("report.md").write_text("# Saved partial report\\nLengths: 4, 2\\n")\n')
        result = await service.run_research_python("calculate.py", 10)
        assert result["exit_code"] == 0
        # Even a previously valid partial checkpoint cannot turn a later interruption into success.
        await service.submit_research_checkpoint("Computed lengths", ["calculate"], ["lengths.json", "report.md"], [])
        events.extend([{"kind": "tool_call_start"}, {
            "kind": "research_interrupted", "reason": reason,
            "research_complete": False, "event_audit": {"passed": audit_passed},
            "native_termination": "stopped",
            "scientific_tool_attempts": [{"tool": "run_research_python", "call_id": "calculation", "status": "completed"}],
        }])

    state = await ar.AutonomousResearch(store, "fixture", {}, consult=consult, executor=execute, tools=[]).run()
    assert roles == ["PI", "CB", "SC"] + (["SC"] if critic_ready else [])
    assert state["status"] != "completed"
    assert state["cycles"][0]["status"] == "needs_revision"
    progress = state["progress"]
    assert not progress["research_completed"]
    assert "reviewed_episode" not in progress
    assert not progress["scientific_acceptance"]["passed"]
    assert not progress["host_review"]["passed"]
    assert progress["plan"]["steps"][0]["status"] == "pending"
    assert (store.root / "workspace" / "lengths.json").exists()
    assert (store.root / "workspace" / "report.md").exists()
    saved_review = json.loads((store.root / "episodes/cycle_000/review.json").read_text())
    assert saved_review["decision"] == "needs_revision"
    assert "Execution was interrupted" in " ".join(progress["scientific_acceptance"]["issues"])
    assert not progress.get("capabilities") and not progress.get("deliveries")


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal", [
    {"kind": "research_interrupted", "reason": "untrusted_reason", "research_complete": False},
    {"kind": "research_interrupted", "reason": "native_model_request_limit", "research_complete": True},
    {"kind": "completed", "text": "Done", "event_audit": {"passed": False}},
    {"kind": "research_interrupted", "reason": "native_execution_failed", "research_complete": False},
    {"kind": "research_interrupted", "reason": "native_execution_failed", "research_complete": False,
     "scientific_tool_attempts": [{"tool": "run_research_python", "status": "rejected"}]},
])
async def test_unrecognized_endings_do_not_gain_partial_review_path(tmp_path, monkeypatch, terminal):
    store = setup(tmp_path, monkeypatch)
    roles = []

    async def consult(role, context):
        roles.append(role)
        if role == "PI":
            return "Compute peptide lengths."
        return json.dumps({"summary": "Compute lengths", "steps": [
            {"id": "calculate", "title": "Compute lengths", "acceptance": "Save computed lengths"}], "needs_input": []})

    async def execute(request, service, events):
        events.append(terminal)

    state = await ar.AutonomousResearch(store, "fixture", {}, consult=consult, executor=execute, tools=[]).run()
    assert "SC" not in roles
    assert state["status"] != "completed"
    assert "reviewed_episode" not in state["progress"]
