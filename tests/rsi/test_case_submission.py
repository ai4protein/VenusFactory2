import importlib.util
import json
from pathlib import Path

import pytest

from agent.rsi import case_job_tools


def runner(monkeypatch):
    scripts = Path(__file__).resolve().parents[2] / "scripts/rsi"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("receipt_runner_test", scripts / "run_harness_reproduction.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.asyncio
async def test_idempotent_receipt_binds_actual_host_job_without_completion_claim(tmp_path, monkeypatch):
    row = {
        "success": True,
        "job_id": "case_fixture",
        "plugin_id": "phl7.hrex_validation.v1",
        "parameters": {"backend": "cpu"},
        "status": "queued",
    }

    async def request(value):
        assert value["action"] == "submit"
        return row.copy()

    monkeypatch.setattr(case_job_tools, "ROOT", tmp_path)
    monkeypatch.setattr(case_job_tools, "request", request)
    first = await case_job_tools.submit_reproduction(row["plugin_id"], '{"backend":"cpu"}', "stable")
    assert first["status"] == "queued"
    original = (tmp_path / "case_submission.json").read_bytes()
    row["status"] = "completed"
    await case_job_tools.submit_reproduction(row["plugin_id"], '{"backend":"cpu"}', "stable")
    assert (tmp_path / "case_submission.json").read_bytes() == original
    host = runner(monkeypatch)
    assert host.submission_receipt_matches(tmp_path, {row["job_id"]: row})
    row["parameters"] = {"backend": "cuda"}
    assert not host.submission_receipt_matches(tmp_path, {row["job_id"]: row})
    with pytest.raises(ValueError, match="another job"):
        await case_job_tools.submit_reproduction(row["plugin_id"], '{"backend":"cuda"}', "other")


@pytest.mark.asyncio
async def test_rejected_request_never_creates_a_receipt(tmp_path, monkeypatch):
    async def request(value):
        raise ValueError("outside campaign")

    monkeypatch.setattr(case_job_tools, "ROOT", tmp_path)
    monkeypatch.setattr(case_job_tools, "request", request)
    with pytest.raises(ValueError, match="outside campaign"):
        await case_job_tools.submit_reproduction("unknown", "{}", "stable")
    assert not (tmp_path / "case_submission.json").exists()


def test_forged_or_symlinked_receipt_fails_host_gate(tmp_path, monkeypatch):
    host = runner(monkeypatch)
    row = {"job_id": "case_real", "plugin_id": "plugin", "parameters": {}}
    path = tmp_path / "case_submission.json"
    jobs = {"case_real": row}
    for value in [
        None,
        {},
        {"schema": True, **row, "submission_only": True},
        {"schema": 1, **row, "job_id": "forged", "submission_only": True},
    ]:
        path.write_text(json.dumps(value))
        assert not host.submission_receipt_matches(tmp_path, jobs)
    path.unlink()
    elsewhere = tmp_path / "elsewhere.json"
    elsewhere.write_text(json.dumps({"schema": 1, **row, "submission_only": True}))
    path.symlink_to(elsewhere)
    assert not host.submission_receipt_matches(tmp_path, jobs)


@pytest.mark.asyncio
async def test_mcp_submission_schema_exposes_the_same_key_bounds():
    tools = await case_job_tools.mcp.list_tools()
    tool = next(t for t in tools if t.name == "submit_reproduction")
    schema = tool.parameters["properties"]["idempotency_key"]
    assert schema["minLength"] == 1 and schema["maxLength"] == 256
    assert schema["pattern"] == "^[a-zA-Z0-9_-]+$"
