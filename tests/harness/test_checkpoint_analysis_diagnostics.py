"""Checkpoint repair uses verified analysis records without granting acceptance."""
import json

import pytest

from agent.chat_agent_utils import _tool_output_indicates_failure
from agent.harness.research_analysis_evidence import file_digest
from agent.harness.research_execution_feedback import feedback_state
from agent.harness.research_handoff import ResearchHandoff
from agent.harness.research_plan_evidence import checkpoint_steps
from agent.harness.research_workspace import ResearchWorkspace
from mcp_tools.scheduler.analysis_evidence import AnalysisCapture
from mcp_tools.scheduler.receipt_binding import value_digest


@pytest.fixture
def research(tmp_path):
    service = ResearchWorkspace(tmp_path / "workspace", tmp_path / "private", {
        "deliverable": "analysis",
        "steps": [
            {"id": "compute", "verification": "execution", "acceptance": "Calculate the supplied sequence properties"},
            {"id": "report", "verification": "artifact", "acceptance": "Write a report"},
        ],
    })
    workspace = service.workspace
    (workspace / "input.fasta").write_text(">fixture\nACDE\n")
    arguments = {"fasta_file": str(workspace / "input.fasta"), "output_file": str(workspace / "values.json")}
    capture = AnalysisCapture({
        "name": "calculate_physchem_from_fasta", "route": "native", "session_id": "fixture",
        "workspace": str(workspace), "arguments": arguments,
    }, workspace, "sequence_analysis")
    (workspace / "values.json").write_text('{"length":4}')
    value = {"status": "success", "length": 4}
    evidence = capture.finish({"value": value, "is_error": False})
    assert evidence is not None
    job = {
        "id": "a" * 32, "tool": "calculate_physchem_from_fasta", "category": "sequence_analysis",
        "status": "completed", "parent_job": "b" * 32, "session_id": "fixture",
        "arguments_sha256": value_digest(arguments), "job_number": "VF-test",
        "result": {"value": value, "is_error": False, "analysis_evidence": evidence},
    }
    artifact = service.record_queued_analysis(job=job, name=job["tool"], arguments=arguments,
        value=value, owner="fixture-owner", parent_job=job["parent_job"], session_id="fixture")
    (workspace / "report.md").write_text("The sequence contains four residues.\n")
    return service, artifact


def request(completed=None, artifacts=None):
    return {
        "summary": "Computed the supplied sequence properties and saved the report.",
        "completed_steps": ["compute"] if completed is None else completed,
        "artifacts": ["report.md"] if artifacts is None else artifacts,
        "needs_input": [],
    }


@pytest.mark.asyncio
async def test_missing_analysis_citation_preserves_draft_and_permits_same_turn_repair(research):
    service, artifact = research
    original = request()
    result = await service.submit_research_checkpoint(**original)
    failed, _ = _tool_output_indicates_failure(result)
    assert failed and result["checkpointed"] is False and result["partial_saved"] is True
    error = result["error"]
    assert error["code"] == "analysis_evidence_not_cited"
    assert error["step_ids"] == ["compute"]
    assert error["available_evidence"] == [
        {"path": artifact["path"], "sha256": artifact["sha256"], "tool": artifact["tool"]}]
    assert "exact path added to artifacts" in error["repair"]
    assert "Do not rerun a valid calculation" in error["repair"]
    assert service.checkpoint["partial"] is True
    assert service.checkpoint["completed_steps"] == []
    assert [item["path"] for item in service.checkpoint["artifacts"]] == ["report.md"]
    assert original["completed_steps"] == ["compute"]
    draft = json.loads((service.control / "checkpoint-draft.json").read_text())
    assert draft["submitted"] == original
    assert draft["checkpoint"]["completed_steps"] == ["compute"]
    assert json.loads((service.control / "checkpoint.json").read_text()) == service.checkpoint

    handoff = ResearchHandoff()
    with pytest.raises(ValueError, match="successful research checkpoint"):
        handoff.save(original, result)
    assert not handoff.closed

    corrected = request(artifacts=["report.md", artifact["path"]])
    saved = await service.submit_research_checkpoint(**corrected)
    assert saved["checkpointed"] is True
    assert not _tool_output_indicates_failure(saved)[0]
    assert "partial" not in service.checkpoint
    handoff.save(corrected, saved)
    assert handoff.closed
    assert json.loads((service.control / "checkpoint-draft.json").read_text()) == draft
    checks = checkpoint_steps(service.plan, service.checkpoint, workspace=service.workspace,
        receipts=service.receipts, experiments=[], progress={})
    assert checks[0]["passed"] is True
    assert checks[0]["analysis_job_ids"] == ["a" * 32]


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["artifact", "data", "model", "revision", "review", "delivery"])
async def test_non_execution_checkpoint_preserves_existing_training_and_review_flow(research, kind):
    service, _ = research
    service.plan["steps"][0]["verification"] = kind
    result = await service.submit_research_checkpoint(**request())
    assert result["checkpointed"] is True
    assert "error" not in result
    assert service.checkpoint["completed_steps"] == ["compute"]
    assert not (service.control / "checkpoint-draft.json").exists()


@pytest.mark.asyncio
async def test_explicit_partial_work_does_not_require_analysis_citation(research):
    service, _ = research
    result = await service.submit_research_checkpoint(**request(completed=[]))
    assert result["checkpointed"] is True
    assert service.checkpoint["completed_steps"] == []
    assert not (service.control / "checkpoint-draft.json").exists()


@pytest.mark.asyncio
async def test_mixed_checkpoint_keeps_other_claims_without_promoting_execution(research):
    service, _ = research
    result = await service.submit_research_checkpoint(**request(completed=["compute", "report"]))
    assert result["checkpointed"] is False
    assert service.checkpoint["completed_steps"] == ["report"]
    assert result["error"]["step_ids"] == ["compute"]


@pytest.mark.asyncio
@pytest.mark.parametrize("altered", ["input", "output", "analysis", "symlink"])
async def test_changed_evidence_is_not_offered_as_a_repair(research, altered):
    service, artifact = research
    path = service.workspace / {
        "input": "input.fasta", "output": "values.json",
        "analysis": artifact["path"], "symlink": artifact["path"],
    }[altered]
    if altered == "symlink":
        other = service.workspace / "copy.json"
        other.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(other)
    else:
        path.write_text("changed")
    result = await service.submit_research_checkpoint(**request())
    # No verified record exists to recommend; downstream evidence checks remain.
    assert result["checkpointed"] is True
    assert "error" not in result
    checks = checkpoint_steps(service.plan, service.checkpoint, workspace=service.workspace,
        receipts=service.receipts, experiments=[], progress={})
    assert checks[0]["passed"] is False


@pytest.mark.asyncio
async def test_workspace_json_and_file_read_do_not_create_execution_evidence(research):
    service, artifact = research
    service.receipts.clear()
    await service.read_research_file("input.fasta")
    result = await service.submit_research_checkpoint(
        **request(artifacts=["report.md", "input.fasta", artifact["path"]]))
    assert result["checkpointed"] is True
    checks = checkpoint_steps(service.plan, service.checkpoint, workspace=service.workspace,
        receipts=service.receipts, experiments=[], progress={})
    assert checks[0]["passed"] is False
    assert checks[0]["analysis_job_ids"] == []


@pytest.mark.asyncio
@pytest.mark.parametrize("exit_code", [0, 1, True])
async def test_successful_script_is_valid_alternative_but_failed_or_boolean_exit_is_not(research, exit_code):
    service, _ = research
    source = service.workspace / "analysis.py"
    source.write_text("print(4)\n")
    service.record("run_research_python", {"path": "analysis.py"}, {
        "exit_code": exit_code, "submitted_source_sha256": file_digest(service.workspace, "analysis.py")})
    result = await service.submit_research_checkpoint(**request(artifacts=["report.md", "analysis.py"]))
    assert result["checkpointed"] is (type(exit_code) is int and exit_code == 0)


@pytest.mark.asyncio
async def test_measured_model_remains_valid_execution_evidence(research):
    service, _ = research
    service.plan["deliverable"] = "model"
    model = service.workspace / "model.json"
    model.write_text('{"value":0.5}')
    record = {
        "id": "c" * 32, "created": 1, "files": {"model.json": file_digest(service.workspace, "model.json")},
        "score_origin": "host_validation", "fresh_process_replay": True,
    }
    folder = service.development.root / "experiments" / record["id"]
    folder.mkdir(parents=True)
    (folder / "receipt.json").write_text(json.dumps(record))
    result = await service.submit_research_checkpoint(**request(artifacts=["report.md", "model.json"]))
    assert result["checkpointed"] is True
    assert "error" not in result


def event_pair(identity, arguments, result):
    return [
        {"kind": "tool_call_start", "tool_call_id": identity,
         "tool_name": "mcp__venusfactory__submit_research_checkpoint", "args": arguments},
        {"kind": "tool_result", "tool_call_id": identity,
         "is_error": _tool_output_indicates_failure(result)[0], "output": json.dumps(result)},
    ]


@pytest.mark.asyncio
async def test_successful_retry_resolves_checkpoint_error_despite_different_arguments(research):
    service, artifact = research
    first = request()
    failed = await service.submit_research_checkpoint(**first)
    events = event_pair("first", first, failed)
    previous = feedback_state(events, service.workspace)
    assert len(previous["pending"]) == 1
    second = request(artifacts=["report.md", artifact["path"]])
    saved = await service.submit_research_checkpoint(**second)
    corrected = event_pair("second", second, saved)
    assert feedback_state(corrected, service.workspace, previous)["pending"] == []
    assert feedback_state(events + corrected, service.workspace)["errors"] == []


@pytest.mark.asyncio
async def test_input_questions_and_unknown_steps_keep_prior_validation(research):
    service, _ = research
    with pytest.raises(ValueError, match="exact step IDs"):
        await service.submit_research_checkpoint(**request(completed=["outside-plan"]))
    assert service.checkpoint is None

@pytest.mark.asyncio
async def test_empty_artifact_list_gets_exact_citation_hint_when_evidence_exists(research):
    service, artifact = research
    result = await service.submit_research_checkpoint(**request(artifacts=[]))
    assert result["checkpointed"] is False and result["partial_saved"] is True
    assert result["error"]["available_evidence"][0]["path"] == artifact["path"]
    assert service.checkpoint["artifacts"] == []
    assert service.checkpoint["completed_steps"] == []


@pytest.mark.asyncio
async def test_empty_artifact_list_without_verified_analysis_keeps_old_error(research):
    service, _ = research
    service.receipts.clear()
    with pytest.raises(ValueError, match="existing artifact filenames"):
        await service.submit_research_checkpoint(**request(artifacts=[]))
    assert service.checkpoint is None


@pytest.mark.asyncio
async def test_explicit_partial_retry_can_close_without_claiming_execution(research):
    service, _ = research
    failed = await service.submit_research_checkpoint(**request())
    assert failed["checkpointed"] is False
    args = request(completed=[])
    saved = await service.submit_research_checkpoint(**args)
    handoff = ResearchHandoff()
    handoff.save(args, saved)
    assert handoff.closed
    assert service.checkpoint["completed_steps"] == []
