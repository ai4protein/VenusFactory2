"""Attempt-owned MCP sidecar; no shared listener or credentials are inherited."""
from __future__ import annotations

import asyncio
import json
import os
import socket
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path


def _write_receipt(path, value):
    """Publish a complete owner-only receipt even under the caller's umask 022."""
    path = Path(path)
    fd, name = tempfile.mkstemp(prefix="." + path.name + "-", dir=path.parent)
    pending = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            os.fchmod(stream.fileno(), 0o600)
            json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        pending.replace(path)
    finally:
        pending.unlink(missing_ok=True)


def _verified_root(record):
    from .cluster_allocation import record_environment_matches, worker_job_matches
    from .pbs_goal import _record, assert_execution_allowed

    current = _record(record["record_path"])
    expected = Path(current["record_path"]).parent / "runtime"
    if (current["nonce"] != record.get("nonce")
            or not worker_job_matches(current) or not record_environment_matches(current)
            or current.get("worker", {}).get("state") != "running"
            or current.get("worker", {}).get("allocation", {}).get("job_id") != current["job_id"]
            or current.get("cancel_requested")):
        raise ValueError("MCP service requires the verified running PBS attempt")
    assert_execution_allowed(current["goal"])
    if (Path(os.environ.get("VENUS_RUNTIME_ROOT", "")).absolute() != expected
            or Path(os.environ.get("VENUS_JOB_QUEUE_ROOT", "")).absolute() != expected / "queue"
            or expected.is_symlink() or (expected / "queue").is_symlink()):
        raise ValueError("MCP service must use this PBS attempt's private runtime")
    return expected


def _process_matches(pid, token, *, queue, command):
    from backend.web_v2.job_queue import process_token

    if not token or process_token(pid) != token:
        return False
    try:
        values = Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
        expected = b"VENUS_JOB_QUEUE_ROOT=" + os.fsencode(queue)
        argv = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
        return expected in values and all(os.fsencode(part) in argv for part in command)
    except (OSError, ValueError):
        return False


def _stop_one(pid, token, *, queue, command):
    from backend.web_v2.job_queue import process_token, stop_process_tree

    if process_token(pid) is None:
        return True
    if not _process_matches(pid, token, queue=queue, command=command):
        # A terminated descendant may lose environ before its stat disappears.
        # Wait for that known identity to exit; never signal an unverified PID.
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            current = process_token(pid)
            if current is None:
                return True
            if current != token:
                return False
            time.sleep(.01)
        return False
    stop_process_tree(pid, token)
    deadline = time.monotonic() + 2
    while process_token(pid) is not None and time.monotonic() < deadline:
        time.sleep(.01)
    try:
        os.waitpid(pid, os.WNOHANG)
    except ChildProcessError:
        pass
    return process_token(pid) is None


def _stop(receipt):
    from .pbs_goal import _private

    private = Path(receipt["runtime"])
    queue = private / "queue"
    stopped = _stop_one(
        receipt["pid"], receipt["token"], queue=queue,
        command=[str(Path(receipt["control"]) / "bootstrap.py")],
    )
    pools = 0
    rejected = []
    for path in sorted((queue / "pools").glob("*/supervisor.json")):
        try:
            row = _private(path)
            pid, token = int(row["pid"]), row["token"]
            if _stop_one(pid, token, queue=queue,
                         command=["backend.web_v2.worker_pool", "supervise", path.parent.name]):
                pools += 1
            else:
                rejected.append(path.parent.name)
        except FileNotFoundError:
            continue  # An idle supervisor may remove its completed receipt.
        except (OSError, ValueError, KeyError, TypeError):
            rejected.append(path.parent.name)
    result = {
        "status": "stopped" if stopped and not rejected else "cleanup_incomplete",
        "service_stopped": stopped, "pool_supervisors_checked": pools,
        "rejected_pool_records": rejected, "at": time.time(),
    }
    _write_receipt(private / "mcp-cleanup.json", result)
    if not stopped or rejected:
        raise RuntimeError("PBS MCP cleanup could not verify every owned process")
    return result


def _start(record):
    from mcp_tools.scheduler import service

    from .pbs_goal import _private

    private = _verified_root(record)
    old_port = os.environ.get("MCP_HTTP_PORT")
    receipt = None
    from .pbs_goal_policy import service_environment
    tool_credentials = service_environment(record)
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen(128)
            port = listener.getsockname()[1]
            os.environ["MCP_HTTP_PORT"] = str(port)
            state = service.ensure_service(listener=listener, scheduler_only=True, env_overrides=tool_credentials)
        control = service.control_dir()
        from backend.web_v2.job_queue import process_token
        receipt = {
            "port": port, "pid": state["pid"], "token": process_token(state["pid"]),
            "runtime": str(private), "control": str(control),
            "runtime_id": state["runtime_id"], "source_id": state["source_id"],
            "attempt_id": record["nonce"], "surface": "scheduler_only",
            "previous_port": old_port,
        }
        process = _private(control / "process.json")
        if process["pid"] != receipt["pid"] or process["token"] != receipt["token"]:
            raise ValueError("MCP process receipt differs from its healthy listener")
        _write_receipt(private / "mcp-service.json", {k: v for k, v in receipt.items() if k != "previous_port"})
        return receipt
    except BaseException:
        try:
            if receipt is not None:
                _stop(receipt)
        finally:
            if old_port is None:
                os.environ.pop("MCP_HTTP_PORT", None)
            else:
                os.environ["MCP_HTTP_PORT"] = old_port
        raise


async def _settle(task):
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            continue
    return task.result()


@asynccontextmanager
async def tool_service(record):
    """Keep a private authenticated scheduler alive without blocking heartbeats."""
    starting = asyncio.create_task(asyncio.to_thread(_start, record))
    receipt = None
    try:
        try:
            receipt = await asyncio.shield(starting)
        except asyncio.CancelledError:
            receipt = await _settle(starting)
            raise
        yield {key: value for key, value in receipt.items() if key != "previous_port"}
    finally:
        if receipt is not None:
            stopping = asyncio.create_task(asyncio.to_thread(_stop, receipt))
            try:
                await asyncio.shield(stopping)
            except asyncio.CancelledError:
                await _settle(stopping)
                raise
            finally:
                previous = receipt["previous_port"]
                if previous is None:
                    os.environ.pop("MCP_HTTP_PORT", None)
                else:
                    os.environ["MCP_HTTP_PORT"] = previous
