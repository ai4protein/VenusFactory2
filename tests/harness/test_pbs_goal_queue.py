"""Real SQLite lease/workspace tests; synthetic PBS receipts, no GPU/provider calls."""
import asyncio
import getpass
import hashlib
import json
import os
import sqlite3
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from agent.harness import Profile
from agent.harness import pbs_goal_queue as subject
from agent.harness.autonomous_research import create
from agent.harness.pbs_goal import ASSIGNMENT, SCHEMA
from agent.rsi.engine.storage import atomic_json
from backend.web_v2.job_queue import JobQueue
from backend.web_v2.task_store import LeaseLost
from mcp_tools.scheduler import jobs


@pytest.fixture
def context(tmp_path, monkeypatch):
    # Fault injection must not restore a finished parent lease into later tests.
    monkeypatch.setattr(os, "environ", dict(os.environ))
    for key in subject.ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("WEBUI_V2_MODE", "local")
    monkeypatch.setenv("VENUS_SCIENTIST_ONLINE", "0")
    monkeypatch.setenv("VENUS_WORKER_AUTOSTART", "0")
    monkeypatch.setenv("VENUS_CAPABILITY_DB", str(tmp_path / "capabilities.sqlite"))
    profile = Profile.from_dict({"mode": "science_expert", "harness": "expert-deepseek-harness",
        "model": {"provider": "dmx", "model": "glm-4-flash", "base_url": "https://www.dmxapi.cn/v1"},
        "budget": {"model_calls": 12, "tool_calls": 12, "wall_seconds": 120, "output_tokens": 4096}})
    store = create(tmp_path / "goals" / ("a" * 32), profile, "Evaluate a synthetic model",
                   {"max_cycles": 2, "model_calls": 24, "tool_calls": 24, "wall_seconds": 300})
    nonce = "b" * 32
    control = tmp_path / "control"
    folder = control / hashlib.sha256(str(store.root).encode()).hexdigest()[:24]
    path = folder / nonce / "submission.json"
    path.parent.mkdir(parents=True, mode=0o700)
    definition = store.read()["definition_sha256"]
    record = {"schema": SCHEMA, "uid": os.getuid(), "owner": getpass.getuser(), "nonce": nonce,
              "goal": str(store.root), "definition_sha256": definition, "record_path": str(path),
              "control_root": str(control), "job_id": "54321.fs", "state": "running", "cancel_requested": False,
              "worker": {"state": "running", "allocation": {"backend": "pbs", "job_id": "54321.fs"}}}
    atomic_json(path, record)
    atomic_json(folder / "current.json", {"attempt_id": nonce})
    atomic_json(store.root / ASSIGNMENT, {"schema": SCHEMA, "nonce": nonce,
                "record": str(path), "definition_sha256": definition})
    root = path.parent / "runtime" / "queue"
    root.mkdir(parents=True)
    monkeypatch.setenv("VENUS_SCIENTIST_PBS_RECORD", str(path))
    monkeypatch.setenv("PBS_JOBID", record["job_id"])
    monkeypatch.setenv("VENUS_JOB_QUEUE_ROOT", str(root))
    return record, store, JobQueue(root)


@pytest.fixture
def tool(monkeypatch):
    class Input(BaseModel):
        fasta_file: str
    adapter = SimpleNamespace(metadata={}, get_input_schema=lambda: Input)
    monkeypatch.setattr(jobs, "native_tools", lambda: {"calculate_physchem_from_fasta": adapter})
    monkeypatch.setattr(jobs, "classify", lambda *a: {"category": "protein_analysis", "resource": "cpu", "timeout_seconds": 60})
    return "calculate_physchem_from_fasta"


def submitted_input(queue, identity):
    row = queue.get(identity)
    command = json.loads(row["dispatch"])["command"]
    return json.loads(Path(command[-1]).read_text())


def test_real_parent_and_tool_submission_anchor_workspace_then_cancel_descendants(context, tool):
    record, store, queue = context
    async def exercise():
        async with subject.queue_run(record) as run:
            parent = queue.get(run["id"], run["owner"])
            assert parent["pid"] == os.getpid() and parent["status"] == "running"
            assert parent["lease"] == run["lease"] and parent["lease_until"] > time.time()
            assert subject.workspace_for_parent(queue, parent) == str(store.root / "workspace")
            result = jobs.ToolJobs(queue).submit(tool, {"fasta_file": str(store.root / "workspace/input.fasta")},
                                                owner=run["owner"], parent_job=run["id"], session_id=store.root.name)
            assert submitted_input(queue, result["id"])["workspace"] == str(store.root / "workspace")
            assert queue.get(result["id"])["metadata"]["parent_job"] == run["id"]
            grandchild, _ = queue.enqueue(run["owner"], "fixture", "cpu", {"kind": "test"}, "test",
                                          metadata={"parent_job": result["id"]})
            other, _ = queue.enqueue("another-owner", "fixture", "cpu", {"kind": "test"}, "test")
            unrelated, _ = queue.enqueue(run["owner"], "fixture", "cpu", {"kind": "test"}, "test")
            store.mutate(lambda value: value.update(status="completed"))
        assert queue.get(run["id"])["status"] == "completed"
        assert queue.get(result["id"])["status"] == "cancelled"
        assert queue.get(grandchild)["status"] == "cancelled"
        assert queue.get(other)["status"] == "queued" and queue.get(unrelated)["status"] == "queued"
        assert all(key not in os.environ for key in subject.ENV_KEYS)
        assert store.read()["usage"]["model_calls"] == 0
    asyncio.run(exercise())


@pytest.mark.parametrize("fault", ["owner", "lease", "nonce", "definition", "goal", "job_id", "current", "workspace_link"])
def test_changed_identity_or_workspace_cannot_submit_tool(context, tool, monkeypatch, tmp_path, fault):
    record, store, queue = context
    async def exercise():
        async with subject.queue_run(record) as run:
            if fault in {"owner", "nonce", "definition", "goal"}:
                with queue.connection(True) as connection:
                    row = connection.execute("SELECT metadata FROM jobs WHERE id=?", (run["id"],)).fetchone()
                    meta = json.loads(row["metadata"])
                    if fault == "owner":
                        connection.execute("UPDATE jobs SET owner=? WHERE id=?", ("another-owner", run["id"]))
                    else:
                        key = {"nonce": "nonce", "definition": "definition_sha256", "goal": "goal_folder"}[fault]
                        meta[key] = "different"
                        connection.execute("UPDATE jobs SET metadata=? WHERE id=?", (json.dumps(meta), run["id"]))
            elif fault == "lease":
                monkeypatch.setenv("VENUS_QUEUE_LEASE_TOKEN", "wrong-lease")
            elif fault == "job_id":
                monkeypatch.setenv("PBS_JOBID", "99999.fs")
            elif fault == "current":
                atomic_json(Path(record["record_path"]).parent.parent / "current.json", {"attempt_id": "c" * 32})
            else:
                workspace = store.root / "workspace"
                workspace.rename(store.root / "original-workspace")
                other = tmp_path / "other-session"
                other.mkdir()
                workspace.symlink_to(other, target_is_directory=True)
            before = list(queue.root.glob("mcp-inputs/*"))
            with pytest.raises((ValueError, FileNotFoundError)):
                jobs.ToolJobs(queue).submit(tool, {"fasta_file": "input.fasta"},
                                           owner=run["owner"], parent_job=run["id"])
            assert list(queue.root.glob("mcp-inputs/*")) == before
    asyncio.run(exercise())


def test_foreign_queue_cannot_reuse_pbs_parent_metadata(context, tool, monkeypatch, tmp_path):
    record, store, queue = context
    async def exercise():
        async with subject.queue_run(record) as run:
            foreign = JobQueue(tmp_path / "foreign-queue")
            parent = queue.get(run["id"])
            identity, _ = foreign.enqueue(run["owner"], "scientist", "research", {"kind": "test"}, "test", metadata=parent["metadata"])
            other = foreign.claim_next("test", "research")
            monkeypatch.setenv("VENUS_QUEUE_JOB_ID", identity)
            monkeypatch.setenv("VENUS_QUEUE_LEASE_TOKEN", other["lease"])
            with pytest.raises(ValueError, match="private queue"):
                jobs.ToolJobs(foreign).submit(tool, {"fasta_file": "input.fasta"}, owner=run["owner"], parent_job=identity)
            foreign.finish(identity, "completed", lease=other["lease"])
    asyncio.run(exercise())


def test_attempt_cannot_acquire_a_second_parent(context):
    record, store, queue = context
    async def exercise():
        async with subject.queue_run(record):
            store.mutate(lambda value: value.update(status="completed"))
        store.mutate(lambda value: value.update(status="active"))
        with pytest.raises(LeaseLost, match="already"):
            async with subject.queue_run(record):
                pytest.fail("A single PBS attempt must not acquire a second run")
    asyncio.run(exercise())


def test_heartbeat_survives_blocked_event_loop(context, monkeypatch):
    record, store, queue = context
    monkeypatch.setattr(JobQueue, "lease_seconds", staticmethod(lambda: 0.3))
    async def exercise():
        async with subject.queue_run(record) as run:
            initial = queue.get(run["id"])["heartbeat"]
            time.sleep(0.55)  # Simulate a synchronous evaluation in the research task.
            with queue.connection() as connection:
                row = connection.execute("SELECT heartbeat,lease_until FROM jobs WHERE id=?", (run["id"],)).fetchone()
            assert row["heartbeat"] > initial and row["lease_until"] > time.time()
            store.mutate(lambda value: value.update(status="completed"))
    asyncio.run(exercise())


@pytest.mark.parametrize("fault", ["lease_replaced", "parent_cancelled", "heartbeat_error"])
def test_heartbeat_failure_interrupts_main_research_even_if_cancel_is_caught(context, monkeypatch, fault):
    record, store, queue = context
    monkeypatch.setattr(JobQueue, "lease_seconds", staticmethod(lambda: 0.3))
    async def exercise():
        interrupted = False
        with pytest.raises(LeaseLost):
            async with subject.queue_run(record) as run:
                if fault == "lease_replaced":
                    with queue.connection(True) as connection:
                        connection.execute("UPDATE jobs SET lease=? WHERE id=?", ("new-owner-lease", run["id"]))
                elif fault == "parent_cancelled":
                    queue.cancel(run["id"], run["owner"])
                else:
                    def broken(*args): raise OSError("synthetic queue failure")
                    monkeypatch.setattr(JobQueue, "heartbeat", broken)
                try:
                    await asyncio.sleep(4)
                except asyncio.CancelledError:
                    interrupted = True
        assert interrupted and all(key not in os.environ for key in subject.ENV_KEYS)
        assert asyncio.current_task().cancelling() == 0
    asyncio.run(exercise())


def test_regular_cancellation_finishes_parent_and_restores_environment(context):
    record, store, queue = context
    async def work():
        async with subject.queue_run(record) as run:
            shared.update(run)
            entered.set()
            await asyncio.sleep(30)
    async def exercise():
        nonlocal entered
        entered = asyncio.Event()
        task = asyncio.create_task(work())
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert queue.get(shared["id"])["status"] == "cancelled"
        assert all(key not in os.environ for key in subject.ENV_KEYS)
    shared, entered = {}, None
    asyncio.run(exercise())


def test_pbs_identity_does_not_grant_web_private_models(context, monkeypatch, tmp_path):
    from backend.web_v2.research_capability_access import owned_sessions
    record, store, queue = context
    database = tmp_path / "sessions.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE sessions(session_id TEXT,owner_key TEXT)")
        connection.execute("INSERT INTO sessions VALUES(?,?)", (store.root.name, "web-visitor-owner"))
    monkeypatch.setenv("VENUSFACTORY_SESSION_DB_PATH", str(database))
    async def exercise():
        async with subject.queue_run(record):
            assert owned_sessions() == []
            store.mutate(lambda value: value.update(status="completed"))
    asyncio.run(exercise())


def test_non_pbs_scientist_still_requires_web_hashed_workspace(context, tool):
    record, store, queue = context
    owner = "regular-owner"
    identity, _ = queue.enqueue(owner, "scientist", "research", {"kind": "test"}, "web",
                                metadata={"session_id": "web-session", "goal_id": store.root.name,
                                          "goal_folder": str(store.root)})
    row = queue.claim_next("web", "research")
    with pytest.raises(ValueError, match="Invalid research workspace"):
        jobs.ToolJobs(queue).submit(tool, {"fasta_file": "input.fasta"}, owner=owner, parent_job=identity)
    queue.finish(identity, "completed", lease=row["lease"])


def test_pbs_tool_cannot_substitute_a_web_session(context, tool):
    record, store, queue = context
    async def exercise():
        async with subject.queue_run(record) as run:
            with pytest.raises(ValueError, match="session differs"):
                jobs.ToolJobs(queue).submit(tool, {"fasta_file": "input.fasta"}, owner=run["owner"],
                                           parent_job=run["id"], session_id="someone-elses-web-session")
    asyncio.run(exercise())


def test_second_cancel_during_heartbeat_cleanup_still_revokes_parent(context, monkeypatch):
    record, store, queue = context
    def delayed_stop(heartbeat):
        heartbeat.stop.wait()
        time.sleep(0.2)
    monkeypatch.setattr(subject._Heartbeat, "watch", delayed_stop)
    async def exercise():
        entered = asyncio.Event()
        shared = {}
        async def work():
            async with subject.queue_run(record) as run:
                shared.update(run)
                entered.set()
                await asyncio.sleep(30)
        task = asyncio.create_task(work())
        await entered.wait()
        task.cancel()
        await asyncio.sleep(0.03)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert queue.get(shared["id"])["status"] == "cancelled"
        assert all(key not in os.environ for key in subject.ENV_KEYS)
    asyncio.run(exercise())
