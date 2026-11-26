"""Tool result delivery across real SQLite running-to-terminal transitions."""
import json

import pytest

from backend.web_v2.job_queue import JobQueue
from mcp_tools.scheduler import client
from mcp_tools.scheduler.jobs import ToolJobs, save_json


@pytest.fixture
def queued_tool(tmp_path):
    queue = JobQueue(tmp_path / "queue")
    identity, _ = queue.enqueue(
        "owner-a", "mcp:cpu_analysis", "cpu", {"fixture": True}, "result-race-fixture",
        metadata={"tool": "fixture_analysis", "category": "cpu_analysis"},
    )
    claimed = queue.claim_next("result-race-fixture", "cpu")
    assert claimed and claimed["id"] == identity
    jobs = ToolJobs(queue)
    number = jobs.records.register("mcp", identity, "owner-a", "mcp:cpu_analysis", "fixture_analysis", claimed["created"])
    return queue, jobs, claimed, number["job_number"]


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal,is_error", [("completed", False), ("failed", True)])
async def test_completion_between_status_reads_delivers_persisted_result(queued_tool, monkeypatch, terminal, is_error):
    queue, jobs, claimed, _ = queued_tool
    result = {"value": json.dumps({"status": "error" if is_error else "success", "value": 7}), "is_error": is_error}
    original = queue.public

    def finish_before_public(identity, owner=None):
        # This is the real worker ordering: save its result, then finish the
        # leased SQLite job after ToolJobs.get's first read saw "running".
        assert queue.get(identity, owner)["status"] == "running"
        save_json(queue.files(identity) / "mcp-result.json", result)
        queue.finish(identity, terminal, lease=claimed["lease"])
        return original(identity, owner)

    monkeypatch.setattr(queue, "public", finish_before_public)
    observed = jobs.get(claimed["id"], "owner-a")
    assert observed["status"] == terminal
    assert observed.get("result") == result
    # The normal client should receive the tool value without another submit,
    # poll or speculative rerun (including explicit tool-level failures).
    assert await client.wait(observed, "owner-a") == result["value"]


def test_prewritten_result_is_not_exposed_while_worker_is_running(queued_tool):
    queue, jobs, claimed, _ = queued_tool
    save_json(queue.files(claimed["id"]) / "mcp-result.json", {"value": {"value": 7}, "is_error": False})
    observed = jobs.get(claimed["id"], "owner-a")
    assert observed["status"] == "running" and "result" not in observed


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal", ["failed", "cancelled", "completed"])
async def test_terminal_job_without_result_is_an_explicit_failure(queued_tool, terminal):
    queue, jobs, claimed, _ = queued_tool
    queue.finish(claimed["id"], terminal, lease=claimed["lease"])
    observed = jobs.get(claimed["id"], "owner-a")
    assert observed["status"] == terminal
    assert "value" not in observed.get("result", {})
    with pytest.raises(RuntimeError):
        await client.wait(observed, "owner-a")


@pytest.mark.parametrize("use_job_number", [False, True])
def test_another_owner_cannot_read_a_ready_result(queued_tool, use_job_number):
    queue, jobs, claimed, number = queued_tool
    save_json(queue.files(claimed["id"]) / "mcp-result.json", {"value": {"private": True}, "is_error": False})
    queue.finish(claimed["id"], "completed", lease=claimed["lease"])
    with pytest.raises(FileNotFoundError):
        jobs.get(number if use_job_number else claimed["id"], "owner-b")
