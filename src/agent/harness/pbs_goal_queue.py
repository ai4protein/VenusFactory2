"""A real, attempt-private queue lease for a standalone PBS Scientist worker.

This identity authorizes its own workspace and MCP descendants only. It is not
an authenticated Web visitor and never grants access to Web private libraries.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import secrets
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from agent.science_goal import GoalStore
from backend.web_v2.job_queue import TERMINAL, JobQueue
from backend.web_v2.task_store import LeaseLost

from .pbs_goal import ASSIGNMENT, SCHEMA, _private, _record

ORIGIN = "pbs_goal.v1"
ENV_KEYS = ("VENUS_QUEUE_JOB_ID", "VENUS_QUEUE_OWNER", "VENUS_QUEUE_LEASE_TOKEN")


def _binding(record_path):
    raw = Path(record_path).absolute()
    if raw != raw.resolve(strict=True):
        raise ValueError("PBS parent receipt cannot use path aliases")
    record = _record(raw)
    goal = Path(record["goal"])
    if not goal.is_absolute() or goal != goal.resolve(strict=True):
        raise ValueError("PBS goal path changed")
    folder = Path(record["control_root"]) / hashlib.sha256(str(goal).encode()).hexdigest()[:24]
    if raw != folder / record["nonce"] / "submission.json":
        raise ValueError("PBS parent receipt is outside its goal ledger")
    current = _private(folder / "current.json")
    marker = _private(goal / ASSIGNMENT)
    if (current.get("attempt_id") != record["nonce"]
            or marker.get("schema") != SCHEMA or marker.get("nonce") != record["nonce"]
            or marker.get("record") != str(raw)
            or marker.get("definition_sha256") != record["definition_sha256"]
            or os.getenv("VENUS_SCIENTIST_PBS_RECORD") != str(raw)
            or not record.get("job_id") or os.getenv("PBS_JOBID") != record["job_id"]
            or record.get("cancel_requested") or record.get("state") in {"finished", "cancelled", "preparation_failed"}
            or record.get("worker", {}).get("state") != "running"
            or record.get("worker", {}).get("allocation", {}).get("backend") != "pbs"
            or record.get("worker", {}).get("allocation", {}).get("job_id") != record["job_id"]):
        raise ValueError("PBS parent is not the current verified allocation")
    if GoalStore(goal).read()["definition_sha256"] != record["definition_sha256"]:
        raise ValueError("PBS goal definition changed")
    return record


def workspace_for_parent(queue, parent):
    """Validate a PBS MCP parent against private receipts, not client assertions."""
    meta = parent.get("metadata", {})
    if meta.get("origin") != ORIGIN or parent.get("kind") != "scientist":
        raise ValueError("Invalid PBS parent origin")
    record = _binding(meta.get("pbs_record", ""))
    expected_root = Path(record["record_path"]).parent / "runtime" / "queue"
    if (queue.root != expected_root or expected_root != expected_root.resolve(strict=True)
            or os.getenv("VENUS_JOB_QUEUE_ROOT") != str(expected_root)
            or parent.get("owner") != "pbs:" + record["nonce"]
            or meta.get("nonce") != record["nonce"]
            or meta.get("definition_sha256") != record["definition_sha256"]
            or meta.get("goal_folder") != record["goal"]
            or meta.get("goal_id") != Path(record["goal"]).name
            or meta.get("session_id") != Path(record["goal"]).name
            or parent.get("status") != "running" or parent.get("cancel_requested")
            or not parent.get("lease") or not parent.get("lease_until")
            or parent["lease_until"] <= time.time()
            or parent.get("id") != os.getenv("VENUS_QUEUE_JOB_ID")
            or parent.get("owner") != os.getenv("VENUS_QUEUE_OWNER")
            or not secrets.compare_digest(str(parent["lease"]), os.getenv("VENUS_QUEUE_LEASE_TOKEN", ""))):
        raise ValueError("PBS parent task or lease does not match its private queue")
    goal = Path(record["goal"])
    workspace = goal / "workspace"
    if not workspace.is_dir() or workspace.is_symlink() or workspace.resolve() != workspace:
        raise ValueError("Invalid PBS research workspace")
    if GoalStore(goal).read()["status"] != "active":
        raise ValueError("PBS research goal is no longer active")
    return str(workspace)


def _cancel_descendants(queue, identity, owner):
    # Never scan another deployment; this queue belongs to one PBS attempt.
    with queue.connection() as connection:
        rows = [dict(row) for row in connection.execute("SELECT id,owner,status,metadata FROM jobs")]
    descendants = {identity}
    pending = {row["id"]: row for row in rows if row["owner"] == owner and row["id"] != identity}
    while pending:
        added = [key for key, row in pending.items()
                 if json.loads(row["metadata"]).get("parent_job") in descendants]
        if not added:
            break
        descendants.update(added)
        for key in added:
            row = pending.pop(key)
            if row["status"] not in TERMINAL:
                queue.cancel(key, owner)


class _Heartbeat:
    def __init__(self, queue, run, record_path, parent, loop):
        self.queue, self.run, self.record_path = queue, run, record_path
        self.parent, self.loop = parent, loop
        self.stop = threading.Event()
        self.failure = None
        self.injected_cancel = False
        self.thread = threading.Thread(target=self.watch, name="pbs-goal-lease", daemon=True)

    def interrupt(self):
        if not self.stop.is_set() and not self.parent.done() and not self.parent.cancelling():
            self.injected_cancel = True
            self.parent.cancel()

    def watch(self):
        interval = min(5.0, self.queue.lease_seconds() / 3)
        while not self.stop.wait(interval):
            try:
                _binding(self.record_path)
                # heartbeat() is fenced and never invokes PID-based orphan recovery.
                self.queue.heartbeat(self.run["id"], self.run["lease"])
                with self.queue.connection() as connection:
                    row = connection.execute("SELECT cancel_requested FROM jobs WHERE id=?", (self.run["id"],)).fetchone()
                if row is None or row["cancel_requested"]:
                    raise LeaseLost("PBS parent task was cancelled")
            except Exception:
                self.failure = LeaseLost("PBS parent lease or allocation is no longer valid")
                self.loop.call_soon_threadsafe(self.interrupt)
                return


@asynccontextmanager
async def queue_run(record):
    """Own a genuine fenced root run while research and its MCP sidecar execute."""
    record = _binding(record["record_path"])
    root = Path(record["record_path"]).parent / "runtime" / "queue"
    if os.getenv("VENUS_JOB_QUEUE_ROOT") != str(root) or root != root.resolve():
        raise ValueError("PBS parent requires its attempt-private queue")
    if any(os.getenv(key) for key in ENV_KEYS):
        raise ValueError("PBS parent cannot inherit another task identity")
    queue = JobQueue(root)
    owner = "pbs:" + record["nonce"]
    namespace = "pbs-parent-" + record["nonce"]
    goal = Path(record["goal"])
    metadata = {"origin": ORIGIN, "pbs_record": record["record_path"], "nonce": record["nonce"],
                "definition_sha256": record["definition_sha256"], "goal_folder": str(goal),
                "goal_id": goal.name, "session_id": goal.name}
    identity, created = queue.enqueue(owner, "scientist", "research", {"kind": "pbs_goal"}, namespace,
                                      metadata=metadata, session_key="pbs:" + record["nonce"],
                                      request_key="pbs-parent:" + record["nonce"], request_hash=record["definition_sha256"])
    if not created:
        raise LeaseLost("This PBS attempt already has a parent task")
    row = queue.claim_next(namespace, "research")
    if row is None or row["id"] != identity:
        queue.cancel(identity, owner)
        raise LeaseLost("PBS parent could not acquire its execution lease")
    run = {"id": identity, "owner": owner, "lease": row["lease"]}
    previous = {key: os.environ.get(key) for key in ENV_KEYS}
    os.environ.update(dict(zip(ENV_KEYS, (identity, owner, row["lease"]))))
    beat = _Heartbeat(queue, run, record["record_path"], asyncio.current_task(), asyncio.get_running_loop())
    failure = None
    try:
        workspace_for_parent(queue, queue.get(identity, owner))
        beat.thread.start()
        yield run
    except BaseException as exc:
        failure = exc
        raise
    finally:
        beat.stop.set()
        cleanup_cancel = None
        if beat.thread.ident is not None:
            joiner = asyncio.create_task(asyncio.to_thread(beat.thread.join, 20))
            while not joiner.done():
                try:
                    await asyncio.shield(joiner)
                except asyncio.CancelledError as exc:
                    cleanup_cancel = exc
            joiner.result()
        cleanup_error = None
        try:
            _cancel_descendants(queue, identity, owner)
            status = GoalStore(goal).read()["status"]
            if failure is not None or beat.failure is not None:
                status = "cancelled" if isinstance(failure, asyncio.CancelledError) and beat.failure is None else "failed"
            elif status not in {"completed", "paused", "waiting_for_input", "interrupted", "cancelled"}:
                status = "interrupted"
            queue.finish(identity, status, "PBS research " + status.replace("_", " "), lease=row["lease"])
        except LeaseLost:
            pass  # A newer lease/terminal record must never be overwritten.
        except Exception as exc:
            cleanup_error = exc
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
        if beat.injected_cancel and asyncio.current_task().cancelling():
            asyncio.current_task().uncancel()
        if beat.failure is not None:
            raise beat.failure from None
        if beat.thread.is_alive():
            raise LeaseLost("PBS parent heartbeat did not stop")
        if cleanup_cancel is not None and failure is None:
            raise cleanup_cancel
        if cleanup_error is not None and failure is None:
            raise cleanup_error
