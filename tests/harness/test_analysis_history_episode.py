"""Two real Scientist episodes reuse capture and cannot replace reports with code.

The LLM and request transport are fixed fixtures. GoalStore, original-input
registration, host capture, private history, checkpoint, and SC gates are real.
"""
import json
from pathlib import Path

import pytest
import test_analysis_report_review as cases
from test_autonomous_research import profile

from agent.harness import analysis_interim_review as interim
from agent.harness import autonomous_research as module
from agent.harness import scientific_acceptance as acceptance
from agent.harness.autonomous_research import AutonomousResearch, create
from agent.harness.research_cli_inputs import prepare_inputs, stage_inputs


@pytest.fixture
def offline_roles(monkeypatch):
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

    monkeypatch.setattr(module, "EvidenceGateway", Relay)
    monkeypatch.setattr(module, "audit_requests", lambda _path: {"passed": True, "model_calls": 4})


def partial_review(context):
    return {"decision": "needs_revision", "summary": "The calculation needs a final report.",
            "issues": ["No verified final analysis report has been submitted."],
            "evidence_ids": context["analysis_review_state"]["verified_task_ids"][:1],
            "next_actions": ["Cite the existing calculation in a complete report."]}


def final_review(context, task=None):
    review = {"decision": "ready_for_host_validation", "summary": "The reported evidence supports the claimed analysis.",
              "issues": [], "claims": [], "input_assessments": [],
              "step_assessments": [{"step_id": "analyze", "verdict": "supported",
                                   "reason": "The actual report addresses the objective.",
                                   "claim_ids": ["measurement"] if task else []}]}
    if task:
        review["claims"] = [{"id": "measurement", "task_id": task, "result_pointer": "/result/data/score",
                             "report_path": "report.md", "report_quote": "Recorded score: 1.233",
                             "display_value": "1.233"}]
        review["input_assessments"] = [{"task_id": task, "kind": "registered_input",
                                         "addresses_objective": True,
                                         "reason": "The original registered sequence matches.",
                                         "source_task_ids": []}]
    return review


@pytest.mark.asyncio
@pytest.mark.parametrize("second_artifact", ["report", "source"])
async def test_next_episode_reuses_calculation_but_requires_actual_report(tmp_path, offline_roles, second_artifact):
    store = create(tmp_path / "goal", profile(), "Analyze the supplied protein sequence and report the measured score.",
                   {"max_cycles": 2, "model_calls": 24, "tool_calls": 24, "wall_seconds": 240})
    input_file = tmp_path / "original.fasta"
    input_file.write_text(">original\nMKWVTFISLLFLFSSAYS\n")
    with prepare_inputs([input_file]) as prepared:
        stage_inputs(store, prepared)
    contexts, captures, current_receipts = [], [], []
    first_receipt_bytes = None
    calls = 0

    async def consult(role, context):
        if role in {"PI", "CB"} and calls == 1:
            feedback = context["progress"]["sc_revision"]
            assert feedback["next_actions"] == ["Cite the existing calculation in a complete report."]
            assert feedback["cited_analysis_task_ids"] == captures
            assert feedback["cited_experiment_ids"] == []
            assert context["progress"]["last_review"]["summary"].startswith("Execution needs revision:")
        if role == "PI":
            return "Preserve the actual supplied sequence and report the measured result."
        if role == "CB":
            return json.dumps({"summary": "Analyze the supplied sequence", "deliverable": "analysis",
                               "steps": [{"id": "analyze", "title": "Analyze the sequence",
                                          "acceptance": "Report the recorded sequence score.",
                                          "verification": "execution"}], "needs_input": []})
        contexts.append(context)
        if context["scientific_acceptance"]["analysis_review_policy"] == interim.POLICY:
            return json.dumps(partial_review(context))
        return json.dumps(final_review(context, captures[0]))

    async def executor(_request, service, events):
        nonlocal calls, first_receipt_bytes
        calls += 1
        if calls == 1:
            task = cases.run_capture(service, {"sequence": "MKWVTFISLLFLFSSAYS"}, {"data": {"score": 1.233}})
            captures.append(task)
            first_receipt_bytes = (service.control / "receipts.json").read_bytes()
            events.append({"kind": "research_interrupted", "reason": "native_execution_failed",
                           "research_complete": False, "native_error_type": "RuntimeError",
                           "scientific_tool_attempts": [{"tool": "analysis", "status": "completed", "job_id": task}],
                           "event_audit": {"passed": True}})
        else:
            assert not service.receipts
            assert json.loads(_request["text"])["prior_analysis"]["records"][0]["task_id"] == captures[0]
            if second_artifact == "report":
                await service.write_research_file("report.md", "Recorded score: 1.233\n")
                artifacts = ["report.md", "analysis/" + captures[0] + ".json"]
            else:
                await service.write_research_file("source.py", "# Recorded score: 1.233\n")
                artifacts = ["source.py", "analysis/" + captures[0] + ".json"]
            await service.submit_research_checkpoint("Save the revised analysis", ["analyze"], artifacts, [])
            events.append({"kind": "completed", "text": "Saved revised artifacts", "event_audit": {"passed": True}})
        current_receipts.append(list(service.receipts))

    state = await AutonomousResearch(store, "fixture", {}, consult=consult, executor=executor, tools=[]).run()
    assert calls == 2 and len(captures) == 1
    assert not any(row["tool"] == "queued_analysis" for row in current_receipts[1])
    assert (store.root / "episodes/cycle_000/tools/receipts.json").read_bytes() == first_receipt_bytes
    assert contexts[0]["scientific_acceptance"]["analysis_review_policy"] == interim.POLICY
    assert contexts[1]["analysis_evidence"]["computations"][0]["host_origin"]["scope"] == "prior_episode"
    assert state["cycles"][0]["progress"]["scientific_acceptance"]["passed"] is False
    assert state["progress"]["research_completed"] is (second_artifact == "report")
    assert state["progress"]["plan"]["steps"][0]["status"] == ("completed" if second_artifact == "report" else "pending")
    assert state["status"] == ("completed" if second_artifact == "report" else "budget_exhausted")


@pytest.mark.asyncio
async def test_noncomputational_analysis_keeps_final_report_review(tmp_path, offline_roles):
    store = create(tmp_path / "goal", profile(), "Summarize the supplied scope in a short research note.",
                   {"max_cycles": 1, "model_calls": 12, "tool_calls": 12, "wall_seconds": 120})

    async def consult(role, context):
        if role == "PI":
            return "Prepare a concise scoped note."
        if role == "CB":
            return json.dumps({"summary": "Write a scope note", "deliverable": "analysis",
                               "steps": [{"id": "analyze", "title": "Write the note",
                                          "acceptance": "Explain the scope and remaining uncertainties.",
                                          "verification": "artifact"}], "needs_input": []})
        assert context["analysis_evidence"]["computations"] == []
        return json.dumps(final_review(context))

    async def executor(_request, service, events):
        await service.write_research_file("report.md", "Scope: inspect the supplied target before proposing experiments.\n")
        await service.submit_research_checkpoint("Prepared a scoped note", ["analyze"], ["report.md"], [])
        events.append({"kind": "completed", "text": "Saved the note", "event_audit": {"passed": True}})

    state = await AutonomousResearch(store, "fixture", {}, consult=consult, executor=executor, tools=[]).run()
    assert state["status"] == "completed"
    assert state["progress"]["scientific_acceptance"]["passed"] is True


def test_pure_python_exit_zero_without_result_binding_does_not_accept_execution():
    # A real report can be eligible, but unsupported computational claims cannot
    # be replaced with an exit-code claim. Sandbox execution remains available.
    import hashlib

    text = "A claimed numeric result.\n"
    digest = hashlib.sha256(text.encode()).hexdigest()
    artifact = {"path": "report.md", "sha256": digest}
    context = {"objective": "Compute a scientific quantity", "plan": {"deliverable": "analysis",
                "steps": [{"id": "analyze", "verification": "execution"}]},
               "execution": {"checkpoint": {"completed_steps": ["analyze"], "artifacts": [artifact]},
                             "receipts": [{"tool": "run_research_python", "result": {"exit_code": 0}}],
                             "step_checks": [{"id": "analyze", "requirements": ["execution"], "passed": True}]},
               "artifact_previews": [{**artifact, "content": text}],
               "analysis_evidence": {"schema": "research-analysis-provenance.v1", "computations": [],
                                     "materials": {"records": []}, "checkpoint_artifacts": {
                                         "records": [{**artifact, "status": "unchanged"}]}}}
    packet = acceptance.evidence_context(context, [], None)
    result = acceptance.assess(final_review(packet), packet)
    assert result["passed"] is False
    assert any("actual result claim" in issue for issue in result["issues"])
