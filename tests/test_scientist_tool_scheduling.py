"""Exercise the real Web Scientist -> MCP queue -> isolated worker boundary."""

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.harness.contracts import Budget, ModelBinding, Profile
from backend.web_v2.job_queue import JobQueue
from backend.web_v2.science_goals import ROOT, ScienceGoalService
from mcp_tools.scheduler.jobs import ToolJobs


@pytest.fixture(params=[None, "browser-owner"], ids=["local-owner", "online-owner"])
def web_goal(tmp_path, monkeypatch, request):
    monkeypatch.setenv("WEBUI_V2_MODE", "online" if request.param else "local")
    monkeypatch.setenv("VENUS_WORKER_AUTOSTART", "0")
    monkeypatch.setenv("VENUS_JOB_QUEUE_ROOT", str(tmp_path / "queue"))
    monkeypatch.setenv("VENUS_RUNTIME_ROOT", str(tmp_path / "runtime"))
    monkeypatch.setenv("VENUS_USAGE_DB", str(tmp_path / "usage.sqlite3"))
    monkeypatch.setenv("VENUS_ACCESS_DB", str(tmp_path / "access.sqlite3"))
    monkeypatch.setenv("VENUS_QUEUE_LEASE_SECONDS", "600")
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")
    # Use the Web service's own path construction AND spawn implementation.
    service = ScienceGoalService(ROOT, state_root=tmp_path / "scientist")
    session = "0a624b72-39a0-41c7-9e1b-536f7d423781"
    folder, metadata = service.paths(session, "c" * 32)
    workspace = folder / "workspace"
    workspace.mkdir(parents=True)
    (workspace / "protein.fasta").write_text(">fixture\nMKWVTFISLLFLFSSAYS\n")
    metadata.write_text(json.dumps({"session_id": session, "online_owner": request.param}))
    profile = Profile(
        "science_agent", "deepseek-harness", ModelBinding("fixture", "fixture", "http://127.0.0.1:1/v1"), Budget()
    )
    monkeypatch.setattr(service, "frozen_profile", lambda path: profile)
    driver = SimpleNamespace(run_command=lambda *args: [sys.executable, "-c", "pass"])
    service.spawn(folder, driver, "run")
    queue = JobQueue()
    identity = json.loads((folder / "queue.json").read_text())["id"]
    row = queue.get(identity)
    claimed = queue.claim_next(row["namespace"], "research")
    assert claimed and claimed["id"] == identity
    assert row["owner"] == (request.param or hashlib.sha256(session.encode()).hexdigest())
    assert row["metadata"]["session_id"] == session
    assert folder.parent.name != session
    yield queue, claimed, workspace
    for record in queue.list(row["owner"]):
        if record["status"] in {"queued", "running"}:
            queue.cancel(record["id"], row["owner"])


def submit_read(queue, parent):
    return ToolJobs(queue).submit(
        "read_fasta",
        {"file_path": "/workspace/protein.fasta"},
        owner=parent["owner"],
        session_id="native-goal-session",
        parent_job=parent["id"],
    )


def test_web_scientist_queues_and_executes_a_real_registered_tool(web_goal):
    from backend.web_v2.worker_pool import execute

    queue, parent, workspace = web_goal
    job = submit_read(queue, parent)
    row = queue.get(job["id"], parent["owner"])
    assert row["status"] == "queued" and row["metadata"]["parent_job"] == parent["id"]
    spec = json.loads(Path(json.loads(row["dispatch"])["command"][-1]).read_text())
    assert spec["workspace"] == str(workspace)
    assert spec["arguments"]["file_path"] == "/workspace/protein.fasta"
    claimed = queue.claim_next(row["namespace"], row["resource"].split(":")[0])
    assert claimed and claimed["id"] == row["id"]
    execute(queue, claimed)
    result = ToolJobs(queue).get(job["id"], parent["owner"])
    log = queue.files(job["id"]) / "worker.log"
    assert result["status"] == "completed", (result, log.read_text()[-2000:] if log.exists() else "")
    assert not result["result"]["is_error"]
    assert "MKWVTFISLLFLFSSAYS" in str(result["result"]["value"])
    assert not list((queue.root / "mcp-inputs").glob("*.json"))


@pytest.mark.parametrize("damage", ["session", "goal", "goal_link", "workspace_link", "stopped", "foreign_owner"])
def test_invalid_or_stopped_scientist_parent_cannot_launch_tools(web_goal, damage):
    queue, parent, workspace = web_goal
    meta = dict(parent["metadata"])
    if damage == "session":
        meta["session_id"] = "another-session"
    elif damage == "goal":
        meta["goal_id"] = "d" * 32
    elif damage in {"goal_link", "workspace_link"}:
        path = workspace.parent if damage == "goal_link" else workspace
        saved = path.with_name(path.name + "-saved")
        path.rename(saved)
        path.symlink_to(saved, target_is_directory=True)
    if damage in {"session", "goal"}:
        queue.finish(parent["id"], "completed", lease=parent["lease"])
        identity, _ = queue.enqueue(parent["owner"], "scientist", "research", {}, "invalid-fixture", metadata=meta)
        parent = queue.claim_next("invalid-fixture", "research")
        assert parent["id"] == identity
    if damage == "stopped":
        queue.cancel(parent["id"], parent["owner"])
    if damage == "foreign_owner":
        parent = {**parent, "owner": "another-owner"}
    with pytest.raises((ValueError, FileNotFoundError)):
        submit_read(queue, parent)
    assert not list((queue.root / "mcp-inputs").glob("*.json"))
    assert not any(row["kind"].startswith("mcp:") for row in queue.list(parent["owner"]))
