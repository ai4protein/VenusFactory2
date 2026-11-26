"""Private MCP lifecycle integration: real HTTP and queued CPU tool, no network API."""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from agent.harness import pbs_goal_tools as tools


@pytest.fixture(params=[0o022, 0o077], ids=["umask022", "umask077"])
def sidecar(tmp_path, monkeypatch, request):
    private = tmp_path / "attempt/runtime"
    private.mkdir(parents=True, mode=0o700)
    queue = private / "queue"
    monkeypatch.setenv("VENUS_RUNTIME_ROOT", str(private))
    monkeypatch.setenv("VENUS_JOB_QUEUE_ROOT", str(queue))
    monkeypatch.setenv("TEMP_OUTPUTS_DIR", str(private / "outputs"))
    monkeypatch.setenv("VENUS_USAGE_DB", str(private / "usage.sqlite"))
    monkeypatch.setenv("VENUS_ACCESS_DB", str(private / "access.sqlite"))
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")
    monkeypatch.setenv("WEBUI_V2_MODE", "local")
    monkeypatch.setenv("VENUS_SCIENTIST_ONLINE", "0")
    for kind in ("CPU", "GPU", "API", "RESEARCH"):
        monkeypatch.setenv("VENUS_WORKERS_" + kind, "1")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        monkeypatch.setenv(name, "1")
    for name in ("VENUS_QUEUE_JOB_ID", "VENUS_QUEUE_OWNER", "VENUS_QUEUE_LEASE_TOKEN",
                 "VENUS_CAPABILITY_POLICY_SNAPSHOT", "VENUS_MODEL_RUNTIMES_CONFIG",
                 "MCP_HTTP_PORT", "TAVILY_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(tools, "_verified_root", lambda record: private)
    from agent.harness import pbs_goal_policy
    monkeypatch.setattr(pbs_goal_policy, "service_environment",
                        lambda record: {"TAVILY_API_KEY": "synthetic-sidecar-only-fixture"})
    before = os.umask(request.param)
    yield {"nonce": "a" * 32}, private
    os.umask(before)


def test_verified_scope_rejects_missing_private_record_before_listening(tmp_path, monkeypatch):
    from agent.harness import pbs_goal
    monkeypatch.setattr(pbs_goal, "_record", lambda path: {
        "record_path": str(tmp_path / "submission.json"), "nonce": "a" * 32,
        "job_id": "1.fixture", "worker": {"state": "claimed"}, "goal": str(tmp_path),
    })
    monkeypatch.setenv("PBS_JOBID", "1.fixture")
    with pytest.raises(ValueError, match="verified running PBS"):
        tools._verified_root({"record_path": str(tmp_path / "submission.json"), "nonce": "a" * 32})


def test_real_private_scheduler_http_cpu_tool_and_cleanup(sidecar):
    record, private = sidecar
    seen = {}

    async def scenario():
        import httpx
        from fastmcp import Client
        from fastmcp.client.transports import StreamableHttpTransport

        from mcp_tools.scheduler import client, service

        async with tools.tool_service(record) as receipt:
            seen.update(receipt)
            assert "TAVILY_API_KEY" not in os.environ
            process_env = Path(f"/proc/{receipt['pid']}/environ").read_bytes().split(b"\0")
            assert b"TAVILY_API_KEY=synthetic-sidecar-only-fixture" in process_env
            async with httpx.AsyncClient(trust_env=False) as http:
                assert (await http.get(service.base_url() + "/mcp")).status_code == 404
            unauthenticated = StreamableHttpTransport(service.base_url() + "/scheduler/mcp")
            async with Client(unauthenticated, timeout=15) as denied:
                result = await denied.call_tool("tool_catalog", {}, raise_on_error=False)
                assert result.is_error
            fasta = private / "protein.fasta"
            fasta.write_text(">fixture\nMKWVTFISLLFLFSSAYS\n")
            output = private / "outputs/result.json"
            result = await asyncio.wait_for(
                client.invoke(
                    "calculate_physchem_from_fasta",
                    {"fasta_file": str(fasta), "output_file": str(output)},
                    owner="pbs:fixture", session_id="fixture-session",
                ), timeout=60,
            )
            value = json.loads(result) if isinstance(result, str) else result
            assert value.get("status") == "success", value
            measured = json.loads(output.read_text())
            assert measured["sequence_length"] == 18
            assert measured["molecular_weight"] > 0
        assert "MCP_HTTP_PORT" not in os.environ

    asyncio.run(scenario())
    from backend.web_v2.job_queue import process_token
    assert process_token(seen["pid"]) is None
    cleanup = json.loads((private / "mcp-cleanup.json").read_text())
    assert cleanup["status"] == "stopped" and cleanup["service_stopped"]
    assert cleanup["pool_supervisors_checked"] >= 1
    state = json.loads((private / "mcp-service.json").read_text())
    assert state["surface"] == "scheduler_only"
    for receipt in [private / "mcp-service.json", private / "mcp-cleanup.json",
                    *list((private / "queue/pools").glob("*/supervisor.json"))]:
        assert receipt.stat().st_mode & 0o777 == 0o600
    assert not list(private.glob(".mcp-*.json-*"))
    from backend.web_v2.job_queue import JobQueue
    queue = JobQueue(private / "queue")
    with queue.connection() as connection:
        rows = list(connection.execute("SELECT status,finished FROM jobs"))
    assert len(rows) == 1 and rows[0]["status"] == "completed" and rows[0]["finished"]
    assert "synthetic-sidecar-only-fixture" not in (private / "mcp-service.json").read_text()
    assert "synthetic-sidecar-only-fixture" not in (private / "mcp-cleanup.json").read_text()


def test_cleanup_rejects_foreign_pool_record_without_signalling_it(tmp_path):
    from agent.harness.host import write_json
    from backend.web_v2.job_queue import process_token

    private = tmp_path / "runtime"
    pool = private / "queue/pools/foreign"
    pool.mkdir(parents=True, mode=0o700)
    foreign = subprocess.Popen([sys.executable, "-I", "-c", "import time; time.sleep(30)"],
                               stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, start_new_session=True)
    try:
        write_json(pool / "supervisor.json", {"pid": foreign.pid, "token": process_token(foreign.pid)})
        (pool / "supervisor.json").chmod(0o600)
        receipt = {"runtime": str(private), "pid": 999999999,
                   "token": "not-a-process", "control": str(private / "control")}
        with pytest.raises(RuntimeError, match="cleanup"):
            tools._stop(receipt)
        assert foreign.poll() is None
        assert json.loads((private / "mcp-cleanup.json").read_text())["rejected_pool_records"] == ["foreign"]
    finally:
        foreign.kill()
        foreign.wait(timeout=5)


def test_cancelled_start_waits_then_cleans_its_service(monkeypatch):
    cleaned = []
    def start(record):
        time.sleep(.15)
        return {"previous_port": None, "pid": 1}
    monkeypatch.setattr(tools, "_start", start)
    monkeypatch.setattr(tools, "_stop", lambda value: cleaned.append(value["pid"]))

    async def scenario():
        async def run():
            async with tools.tool_service({}):
                pytest.fail("Cancelled startup must not enter the research body")
        task = asyncio.create_task(run())
        await asyncio.sleep(.02)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    asyncio.run(scenario())
    assert cleaned == [1]


def test_cleanup_accepts_process_disappearing_during_identity_read(monkeypatch, tmp_path):
    from backend.web_v2 import job_queue
    identities = iter(["saved-start-identity", None])
    monkeypatch.setattr(job_queue, "process_token", lambda pid: next(identities))
    monkeypatch.setattr(tools, "_process_matches", lambda *args, **kwargs: False)
    assert tools._stop_one(12345, "saved-start-identity", queue=tmp_path, command=["fixture"])


def test_private_application_never_mounts_public_tools(monkeypatch):
    from mcp_tools.scheduler import mcp_service
    monkeypatch.setattr(mcp_service, "identity", lambda **kwargs: {"runtime_id": "fixture"})

    class PublicForbidden:
        def http_app(self, **kwargs):
            pytest.fail("Private workers must not expose the unauthenticated public tools")

    application = mcp_service.application(PublicForbidden(), scheduler_only=True)
    assert [route.path for route in application.routes] == ["/health", "/scheduler"]


def test_prebound_listener_must_already_be_listening(tmp_path, monkeypatch):
    import socket

    from mcp_tools.scheduler import service
    monkeypatch.setattr(service, "control_dir", lambda: tmp_path)
    monkeypatch.setattr(service.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("must not spawn"))
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        monkeypatch.setenv("MCP_HTTP_PORT", str(listener.getsockname()[1]))
        with pytest.raises(ValueError, match="listening socket"):
            service.ensure_service(listener=listener, scheduler_only=True)


def test_private_and_public_service_identities_are_distinct(tmp_path, monkeypatch):
    from mcp_tools.scheduler import service
    monkeypatch.setattr(service, "control_dir", lambda: tmp_path)
    public, private = service.identity(), service.identity(scheduler_only=True)
    assert public["runtime_id"] != private["runtime_id"]
    assert public["source_id"] == private["source_id"]


def test_fixed_worker_entry_sets_private_umask(tmp_path):
    import shutil

    frozen = tmp_path / "frozen"
    script = frozen / "scripts/rsi/pbs_goal_worker.py"
    script.parent.mkdir(parents=True)
    repository = Path(__file__).resolve().parents[2]
    shutil.copyfile(repository / "scripts/rsi/pbs_goal_worker.py", script)
    access = frozen / "backend/compute"
    access.mkdir(parents=True)
    shutil.copyfile(repository / "backend/__init__.py", access.parent / "__init__.py")
    (access / "__init__.py").touch()
    shutil.copyfile(repository / "backend/compute/access.py", access / "access.py")
    package = frozen / "src/agent/harness"
    package.mkdir(parents=True)
    (package.parent / "__init__.py").touch()
    (package / "__init__.py").touch()
    (package / "pbs_goal_worker.py").write_text(
        "from pathlib import Path\n"
        "def run_record(path):\n"
        "    Path(path).with_name('created-receipt.json').write_text('{}')\n"
        "    return {'status': 'completed', 'cycles': []}\n"
    )
    record = tmp_path / "submission.json"
    record.write_text(json.dumps({"source": {"root": str(frozen)}, "execution": {"extra_python_paths": []}}))
    result = subprocess.run([sys.executable, "-I", str(script), "--record", str(record)],
                            capture_output=True, text=True, timeout=10, umask=0o022)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / "created-receipt.json").stat().st_mode & 0o777 == 0o600


def test_exit_stops_running_child_and_finishes_its_queue_record(sidecar, monkeypatch):
    from types import SimpleNamespace

    from agent.harness import pbs_goal_queue
    from backend.web_v2.job_queue import JobQueue, launch, process_token

    record, private = sidecar
    goal = private.parent / "goal"
    goal.mkdir()
    record.update(record_path=str(private.parent / "submission.json"),
                  goal=str(goal), definition_sha256="b" * 64)
    # Allocation/ownership guards have independent real-receipt tests. Here only
    # their synthetic fixture is substituted; queue leases and processes are real.
    monkeypatch.setattr(pbs_goal_queue, "_binding", lambda path: record)
    monkeypatch.setattr(pbs_goal_queue, "workspace_for_parent", lambda queue, parent: str(goal))
    status = {"status": "active"}
    monkeypatch.setattr(pbs_goal_queue, "GoalStore", lambda path: SimpleNamespace(read=lambda: status))
    queue = JobQueue(private / "queue")
    seen = {}

    async def scenario():
        async with pbs_goal_queue.queue_run(record) as parent, tools.tool_service(record):
            seen["parent"] = parent["id"]
            marker = private / "running-child.pid"
            source = "import os,sys,time; from pathlib import Path; Path(sys.argv[1]).write_text(str(os.getpid())); time.sleep(60)"
            identity, _ = launch([sys.executable, "-c", source, str(marker)],
                         owner=parent["owner"], kind="synthetic_cleanup", resource="cpu",
                         queue=queue, cwd=private, metadata={"parent_job": parent["id"]})
            seen["child"] = identity
            deadline = time.monotonic() + 15
            while not marker.exists() and time.monotonic() < deadline:
                await asyncio.sleep(.05)
            assert marker.exists(), queue.get(identity)["status"]
            seen["pid"] = int(marker.read_text())
            assert queue.get(identity)["status"] == "running"
            assert process_token(seen["pid"])
            status["status"] = "completed"

    asyncio.run(scenario())
    assert process_token(seen["pid"]) is None
    # Read SQL directly: the test must not itself trigger lazy orphan recovery.
    with queue.connection() as connection:
        states = {row["id"]: dict(row) for row in connection.execute("SELECT id,status,finished,exit_code FROM jobs")}
    assert states[seen["parent"]]["status"] == "completed"
    child = states[seen["child"]]
    assert child["status"] in {"cancelled", "interrupted", "failed"}
    if child["status"] == "failed":
        # A worker may persist the signal exit before parent cancellation runs.
        assert child["exit_code"] in {-15, -9}
    assert states[seen["child"]]["finished"]
    assert not any(row["status"] in {"queued", "running"} for row in states.values())
