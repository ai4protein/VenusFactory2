"""Queued computations require host task identity and unchanged checkpoint bytes."""

import pytest

from agent.harness.research_analysis_evidence import capture, checkpointed, file_digest, workspace_digest
from agent.harness.research_plan_evidence import checkpoint_steps
from agent.harness.research_workspace import ResearchWorkspace
from mcp_tools.scheduler.receipt_binding import value_digest


@pytest.fixture
def result(tmp_path):
    workspace = tmp_path / "workspace"
    service = ResearchWorkspace(workspace, tmp_path / "private", {"steps": []})
    (workspace / "protein.fasta").write_text(">fixture\nMKWVTFISLLFLFSSAYS\n")
    (workspace / "physchem.json").write_text('{"length":18}')
    args = {"fasta_file": str(workspace / "protein.fasta"), "output_file": str(workspace / "physchem.json")}
    value = {"status": "success", "sequence_length": 18, "output_file": str(workspace / "physchem.json")}
    evidence = {
        "version": 1,
        "tool": "calculate_physchem_from_fasta",
        "category": "sequence_analysis",
        "workspace_sha256": workspace_digest(workspace),
        "arguments_sha256": value_digest(args),
        "result_sha256": value_digest(value),
        "input_files": {"protein.fasta": file_digest(workspace, "protein.fasta")},
        "output_files": {"physchem.json": file_digest(workspace, "physchem.json")},
    }
    job = {
        "id": "a" * 32,
        "status": "completed",
        "tool": evidence["tool"],
        "category": evidence["category"],
        "job_number": "VF-fixture",
        "parent_job": "b" * 32,
        "session_id": "fixture-session",
        "arguments_sha256": evidence["arguments_sha256"],
        "result": {"value": value, "is_error": False, "analysis_evidence": evidence},
    }
    params = {
        "job": job,
        "name": evidence["tool"],
        "arguments": args,
        "value": value,
        "owner": "owner-one",
        "parent_job": "b" * 32,
        "session_id": "fixture-session",
    }
    return service, params


def test_success_creates_scoped_result_and_proves_execution_only(result):
    service, params = result
    saved = capture(service, **params)
    assert str(service.workspace) not in (service.workspace / saved["path"]).read_text()
    assert saved["sha256"] == file_digest(service.workspace, saved["path"])
    cp = {"completed_steps": ["compute"], "artifacts": [{"path": saved["path"], "sha256": saved["sha256"]}]}
    for kind in ("execution", "data", "model", "revision"):
        checks = checkpoint_steps(
            {"steps": [{"id": "compute", "verification": kind}]},
            cp,
            workspace=service.workspace,
            receipts=service.receipts,
            experiments=[],
            progress={},
        )
        assert checks[0]["passed"] is (kind == "execution")
    assert checkpointed(service.receipts, service.workspace, {saved["path"]: saved["sha256"]}) == ["a" * 32]


@pytest.mark.parametrize(
    "field,value",
    [
        ("parent_job", "c" * 32),
        ("session_id", "other-session"),
        ("tool", "read_fasta"),
        ("status", "running"),
        ("arguments_sha256", "0" * 64),
        ("category", "web_search"),
        ("id", "../foreign"),
    ],
)
def test_wrong_terminal_task_binding_is_rejected(result, field, value):
    service, params = result
    params["job"][field] = value
    with pytest.raises(ValueError):
        capture(service, **params)
    assert not service.receipts


@pytest.mark.parametrize(
    "fault", ["failed", "forged_result", "input_changed", "output_changed", "other_workspace", "claimed_evidence"]
)
def test_unverified_or_modified_results_never_become_execution_evidence(result, fault):
    service, params = result
    if fault == "failed":
        params["job"]["result"]["is_error"] = True
    elif fault == "forged_result":
        params["value"] = {"status": "success", "length": 999}
    elif fault == "input_changed":
        (service.workspace / "protein.fasta").write_text(">changed\nAAAA\n")
    elif fault == "output_changed":
        (service.workspace / "physchem.json").write_text("{}")
    elif fault == "other_workspace":
        params["job"]["result"]["analysis_evidence"]["workspace_sha256"] = "0" * 64
    else:
        params["job"]["result"].pop("analysis_evidence")
        params["value"]["analysis_evidence"] = {"version": 1, "passed": True}
        assert capture(service, **params) is None
        assert not service.receipts
        return
    with pytest.raises(ValueError):
        capture(service, **params)
    assert not service.receipts


@pytest.mark.parametrize("changed", ["artifact", "input", "output", "receipt", "symlink"])
def test_checkpoint_revalidates_every_bound_file(result, changed):
    service, params = result
    saved = capture(service, **params)
    artifacts = {saved["path"]: saved["sha256"]}
    if changed == "artifact":
        (service.workspace / saved["path"]).write_text('{"passed":true}')
    elif changed == "input":
        (service.workspace / "protein.fasta").write_text("changed")
    elif changed == "output":
        (service.workspace / "physchem.json").write_text("changed")
    elif changed == "receipt":
        service.receipts[0]["result"]["evidence"]["result_sha256"] = "invalid"
    else:
        path = service.workspace / "physchem.json"
        other = service.workspace / "replacement.json"
        other.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(other)
    assert checkpointed(service.receipts, service.workspace, artifacts) == []


def test_unrelated_checkpoint_and_workspace_do_not_reuse_private_receipt(result, tmp_path):
    service, params = result
    saved = capture(service, **params)
    assert checkpointed(service.receipts, service.workspace, {}) == []
    other = tmp_path / "other"
    other.mkdir()
    assert checkpointed(service.receipts, other, {saved["path"]: saved["sha256"]}) == []


@pytest.mark.asyncio
async def test_scheduler_terminal_callback_keeps_result_and_failure_identity(monkeypatch):
    from mcp_tools.scheduler.client import wait

    for status, error in [("completed", False), ("failed", True)]:
        job = {
            "id": "a" * 32,
            "status": status,
            "position": 0,
            "result": {"value": {"status": "error" if error else "success"}, "is_error": error},
        }
        terminal = []
        assert await wait(job, "owner", on_terminal=terminal.append) == job["result"]["value"]
        assert terminal == [job]


@pytest.mark.parametrize("value", [None, "provider returned no result", []])
def test_absent_terminal_result_does_not_create_evidence(result, value):
    service, params = result
    params["job"]["result"] = value
    assert capture(service, **params) is None
    assert service.receipts == []
