"""Synthetic Slurm kernel/scheduler fixtures plus real SQLite admission; no cluster jobs."""

import asyncio
import copy
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import test_pbs_goal_queue as pbs_tests
from pydantic import BaseModel

from agent.harness import cluster_allocation as subject
from agent.harness import pbs_goal_queue
from agent.harness.session_sandbox import SandboxUnavailable
from agent.rsi.engine.storage import atomic_json
from backend.compute.contracts import JobRef, Observation, ResourceRequest
from backend.web_v2.job_queue import JobQueue, resource_slot
from mcp_tools.scheduler import jobs

pbs_context = pbs_tests.context  # Reuse the real GoalStore/private-ledger fixture.


@pytest.fixture
def allocation(tmp_path, monkeypatch):
    return make_allocation(tmp_path, monkeypatch)


def make_allocation(tmp_path, monkeypatch, version=2):
    proc, cg = tmp_path / "proc", tmp_path / "cgroup"
    (proc / "self").mkdir(parents=True)
    location = "/slurm/uid_1000/job_4242/step_batch"
    if version == 2:
        (proc / "self/cgroup").write_text("0::" + location + "\n")
        memory = cpuset = cg / location.lstrip("/")
        memory.mkdir(parents=True)
        (memory / "memory.max").write_text("max")
        (memory.parent / "memory.max").write_text(str(16 * 1024**3))
        (cpuset / "cpuset.cpus.effective").write_text("4-5")
    else:
        (proc / "self/cgroup").write_text("1:memory:" + location + "\n2:cpuset:" + location + "\n")
        memory = cg / "memory" / location.lstrip("/")
        cpuset = cg / "cpuset" / location.lstrip("/")
        memory.mkdir(parents=True)
        cpuset.mkdir(parents=True)
        (memory / "memory.limit_in_bytes").write_text(str(2**63 - 4096))
        (memory.parent / "memory.limit_in_bytes").write_text(str(16 * 1024**3))
        (cpuset / "cpuset.cpus").write_text("4-5")
    for folder in {memory, cpuset}:
        (folder / "cgroup.procs").write_text(str(os.getpid()))
    monkeypatch.setattr(subject, "PROC", proc)
    monkeypatch.setattr(subject, "CGROUP", cg)
    monkeypatch.setattr(os, "sched_getaffinity", lambda _: {4, 5})
    monkeypatch.setenv("WEBUI_V2_MODE", "local")
    monkeypatch.setenv("VENUS_SCIENTIST_ONLINE", "0")
    for key in ["VENUS_SANDBOX_LOCAL_MEMORY_MB", "VENUS_SANDBOX_TOTAL_MEMORY_MB", "VENUS_SANDBOX_MEMORY_RESERVE_MB"]:
        monkeypatch.delenv(key, raising=False)
    record = {
        "uid": os.getuid(),
        "nonce": "a" * 32,
        "job_id": "4242",
        "scheduler": {"backend": "slurm", "cluster": "testcluster", "submitted_after": 100, "submitted_before": 120},
        "resources": {"queue": "cpu", "ncpus": 2, "ngpus": 0, "memory_gb": 16, "wall_seconds": 120},
    }
    monkeypatch.setenv("SLURM_JOB_ID", "4242")
    monkeypatch.setenv("SLURM_CLUSTER_NAME", "testcluster")
    monkeypatch.setenv("VENUS_PBS_GOAL_ATTEMPT", record["nonce"])
    observation = Observation(
        "running",
        JobRef("slurm", "testcluster", "4242"),
        "RUNNING",
        ResourceRequest("cpu", 2, 0, 16384, 120),
        110,
        evidence_sha256="b" * 64,
    )
    return record, observation, memory, cpuset


def capture(record, observed, monkeypatch):
    from agent.harness import pbs_goal

    def verify(current):
        assert current is record
        return observed

    monkeypatch.setattr(pbs_goal, "worker_observation", verify, raising=False)
    return subject.capture_allocation(record)


@pytest.mark.parametrize("version", [1, 2])
def test_kernel_and_scheduler_proof_produces_cpu_only_receipt(tmp_path, monkeypatch, version):
    record, observed, _, _ = make_allocation(tmp_path, monkeypatch, version)
    receipt = capture(record, observed, monkeypatch).receipt()
    assert receipt["device"] == "cpu" and receipt["ngpus"] == 0
    assert receipt["memory_hard_limit_bytes"] == 16 * 1024**3
    assert receipt["cpu_ids"] == [4, 5] and receipt["cgroup_version"] == version
    assert "gpu_uuid" not in receipt and receipt["scheduler_evidence_sha256"] == "b" * 64


@pytest.mark.parametrize(
    "fault",
    [
        "env_job",
        "cluster",
        "uid",
        "online",
        "gpu_request",
        "not_running",
        "wrong_resource",
        "wrong_observed_job",
        "no_hash",
        "time",
    ],
)
def test_unverified_scheduler_identity_cannot_authorize_cpu(allocation, monkeypatch, fault):
    record, observed, _, _ = allocation
    if fault in {"env_job", "cluster", "nonce", "online"}:
        key = {
            "env_job": "SLURM_JOB_ID",
            "cluster": "SLURM_CLUSTER_NAME",
            "nonce": "VENUS_PBS_GOAL_ATTEMPT",
            "online": "WEBUI_V2_MODE",
        }[fault]
        monkeypatch.setenv(key, "online" if fault == "online" else "wrong")
    elif fault == "uid":
        record["uid"] += 1
    elif fault == "gpu_request":
        record["resources"]["ngpus"] = 1
    else:
        from dataclasses import replace

        changes = {
            "not_running": {"state": "queued"},
            "wrong_resource": {"resources": ResourceRequest("cpu", 1, 0, 16384, 120)},
            "wrong_observed_job": {"ref": JobRef("slurm", "testcluster", "4243")},
            "no_hash": {"evidence_sha256": None},
            "time": {"submitted_at": 121},
        }
        observed = replace(observed, **changes[fault])
    with pytest.raises((SandboxUnavailable, PermissionError)):
        capture(record, observed, monkeypatch)


@pytest.mark.parametrize(
    "fault",
    ["pid", "missing_memory", "unlimited_memory", "large_memory", "cpuset", "affinity", "foreign_cgroup", "sluid_only"],
)
def test_kernel_enforcement_cannot_be_replaced_by_scheduler_claim(allocation, monkeypatch, fault):
    record, observed, memory, cpuset = allocation
    if fault == "pid":
        (memory / "cgroup.procs").write_text("999999")
    elif fault == "missing_memory":
        (memory.parent / "memory.max").unlink()
    elif fault == "unlimited_memory":
        (memory.parent / "memory.max").write_text("max")
    elif fault == "large_memory":
        (memory.parent / "memory.max").write_text(str(32 * 1024**3))
    elif fault == "cpuset":
        (cpuset / "cpuset.cpus.effective").write_text("0-7")
    elif fault == "affinity":
        monkeypatch.setattr(os, "sched_getaffinity", lambda _: {0, 1})
    else:
        location = "/slurm/job_9999/step_batch" if fault == "foreign_cgroup" else "/slurm/sluid_abc/step_batch"
        (subject.PROC / "self/cgroup").write_text("0::" + location)
    with pytest.raises((SandboxUnavailable, OSError)):
        capture(record, observed, monkeypatch)


@pytest.fixture
def cpu_parent(pbs_context, tmp_path, monkeypatch):
    record, store, queue = pbs_context
    synthetic, observed, _, _ = make_allocation(tmp_path / "kernel", monkeypatch)
    record.update({key: synthetic[key] for key in ("scheduler", "resources", "job_id")})
    monkeypatch.setenv("VENUS_PBS_GOAL_ATTEMPT", record["nonce"])
    receipt = capture(record, observed, monkeypatch).receipt()
    record["worker"]["allocation"] = receipt
    monkeypatch.delenv("VENUS_SCIENTIST_PBS_RECORD", raising=False)
    monkeypatch.delenv("PBS_JOBID", raising=False)
    monkeypatch.setenv("VENUS_SCIENTIST_CLUSTER_RECORD", record["record_path"])
    atomic_json(Path(record["record_path"]), record)
    return record, store, queue


def test_real_cpu_parent_allows_cpu_tools_but_no_gpu_queue_or_nested_slot(cpu_parent, monkeypatch):
    record, store, queue = cpu_parent

    class Input(BaseModel):
        value: str

    tool = SimpleNamespace(
        metadata={"tool_category": "model_prediction", "queue_resource": "gpu"}, get_input_schema=lambda: Input
    )
    monkeypatch.setattr(jobs, "native_tools", lambda: {"synthetic_gpu": tool})
    monkeypatch.setattr(jobs, "assert_tool_call_allowed", lambda *args: None)

    async def exercise():
        async with pbs_goal_queue.queue_run(record) as run:
            assert run["owner"] == "cluster:slurm:" + record["nonce"]
            parent = queue.get(run["id"], run["owner"])
            assert parent["metadata"]["origin"] == "cluster_goal.v1"
            assert "pbs_record" not in parent["metadata"] and "PBS_JOBID" not in os.environ
            assert pbs_goal_queue.workspace_for_parent(queue, parent) == str(store.root / "workspace")
            cpu = queue.submit(run["owner"], "fixture", "cpu")
            queue.cancel(cpu)
            with pytest.raises(SandboxUnavailable, match="GPU tools"):
                queue.submit(run["owner"], "fixture", "gpu")
            with pytest.raises(SandboxUnavailable, match="GPU tools"):
                queue.enqueue(run["owner"], "fixture", "gpu", {"kind": "fixture"}, "fixture")
            with pytest.raises(SandboxUnavailable, match="GPU tools"):
                async with resource_slot("gpu"):
                    pytest.fail("GPU slot must not be granted")
            with pytest.raises(SandboxUnavailable, match="GPU tools"):
                jobs.ToolJobs(queue).submit("synthetic_gpu", {"value": "x"}, owner=run["owner"], parent_job=run["id"])
            assert not list(queue.root.glob("mcp-inputs/*"))
            with queue.connection() as connection:
                assert connection.execute("SELECT COUNT(*) FROM jobs WHERE resource='gpu'").fetchone()[0] == 0
            store.mutate(lambda state: state.update(status="completed"))

    asyncio.run(exercise())


@pytest.mark.parametrize("fault", ["marker", "nonce", "cluster", "receipt_gpu", "receipt_uid", "resource_drift"])
def test_cpu_parent_revalidates_private_allocation_on_each_admission(cpu_parent, monkeypatch, fault):
    record, store, queue = cpu_parent

    async def exercise():
        async with pbs_goal_queue.queue_run(record) as run:
            changed = copy.deepcopy(record)
            if fault == "marker":
                monkeypatch.setenv("VENUS_SCIENTIST_CLUSTER_RECORD", "/missing")
            elif fault == "nonce":
                atomic_json(Path(record["record_path"]).parent.parent / "current.json", {"attempt_id": "c" * 32})
            elif fault == "cluster":
                monkeypatch.setenv("SLURM_CLUSTER_NAME", "foreign")
            elif fault == "receipt_gpu":
                changed["worker"]["allocation"]["ngpus"] = 1
            elif fault == "receipt_uid":
                changed["worker"]["allocation"]["uid"] += 1
            else:
                memory = subject.CGROUP / "slurm/uid_1000/job_4242"
                (memory / "memory.max").write_text(str(32 * 1024**3))
            if fault.startswith("receipt_"):
                atomic_json(Path(record["record_path"]), changed)
            with pytest.raises((ValueError, OSError, SandboxUnavailable)):
                queue.submit(run["owner"], "fixture", "cpu")

    asyncio.run(exercise())


def test_no_cluster_context_does_not_change_global_gpu_capacity(tmp_path, monkeypatch):
    monkeypatch.delenv(subject.RECORD_ENV, raising=False)
    monkeypatch.delenv(subject.LEGACY_RECORD_ENV, raising=False)
    queue = JobQueue(tmp_path / "queue")
    before = queue.capacity("gpu")
    identity = queue.submit("ordinary-local", "fixture", "gpu")
    assert queue.get(identity)["status"] == "queued" and queue.capacity("gpu") == before


@pytest.mark.parametrize("queue_worker", [False, True])
def test_mcp_gpu_call_is_denied_before_queue_worker_bypass(cpu_parent, monkeypatch, queue_worker):
    from mcp_tools.scheduler import catalog, mcp_service
    from mcp_tools.tools import runtime_tool_policy

    monkeypatch.setattr(runtime_tool_policy, "assert_tool_call_allowed", lambda *args: None)
    monkeypatch.setattr(
        catalog,
        "native_tools",
        lambda: {
            "synthetic_gpu": SimpleNamespace(metadata={"tool_category": "model_prediction", "queue_resource": "gpu"})
        },
    )
    monkeypatch.setenv("VENUS_MCP_QUEUE_WORKER", "1" if queue_worker else "0")
    context = SimpleNamespace(message=SimpleNamespace(name="synthetic_gpu", arguments={}))

    async def never(_):
        pytest.fail("A CPU allocation must not enter GPU implementation")

    with pytest.raises(SandboxUnavailable, match="GPU tools"):
        asyncio.run(mcp_service.QueueMiddleware().on_call_tool(context, never))


def test_cpu_sandbox_still_executes_with_verified_cpu_parent(cpu_parent, monkeypatch):
    import shutil

    from agent.harness import session_sandbox

    if not shutil.which("bwrap"):
        pytest.skip("Real bubblewrap is unavailable on this test host")
    record, store, queue = cpu_parent
    monkeypatch.setenv("VENUS_SESSION_SANDBOX_BACKEND", "bwrap")
    workspace = store.root / "workspace"
    (workspace / "check.py").write_text("print('cpu-sandbox-ok')\n")

    async def exercise():
        async with pbs_goal_queue.queue_run(record):
            result = await session_sandbox.execute(workspace, store.root / "sandbox-control", "python", "check.py", 20)
            assert result["exit_code"] == 0 and "cpu-sandbox-ok" in result["stdout"]
            assert result.get("device", "cpu") == "cpu"
            store.mutate(lambda state: state.update(status="completed"))

    asyncio.run(exercise())


@pytest.mark.parametrize(
    "key,value", [("WEBUI_V2_MODE", "ONLINE"), ("WEBUI_V2_MODE", "unknown"), ("VENUS_SCIENTIST_ONLINE", "true")]
)
def test_local_guard_precedes_new_admission_but_does_not_block_cleanup(cpu_parent, monkeypatch, key, value):
    record, store, queue = cpu_parent

    async def exercise():
        async with pbs_goal_queue.queue_run(record) as run:
            monkeypatch.setenv(key, value)
            with pytest.raises(PermissionError):
                queue.submit(run["owner"], "fixture", "cpu")
            with pytest.raises(PermissionError):
                subject.capture_allocation(record)
            store.mutate(lambda state: state.update(status="completed"))
        assert queue.get(run["id"])["status"] == "completed"

    asyncio.run(exercise())


def test_slurm_mcp_private_root_accepts_real_scheduler_identity(cpu_parent, monkeypatch):
    from agent.harness import pbs_goal, pbs_goal_tools

    record, store, queue = cpu_parent
    monkeypatch.setenv("VENUS_RUNTIME_ROOT", str(queue.root.parent))
    checked = []

    def allowed(goal):
        checked.append(goal)
        assert pbs_goal_queue._binding(record["record_path"])["nonce"] == record["nonce"]

    monkeypatch.setattr(pbs_goal, "assert_execution_allowed", allowed)
    assert pbs_goal_tools._verified_root(record) == queue.root.parent
    assert checked == [str(store.root)] and "PBS_JOBID" not in os.environ
    monkeypatch.setenv("SLURM_JOB_ID", "9999")
    with pytest.raises(ValueError):
        pbs_goal_tools._verified_root(record)


def test_authored_cuda_request_cannot_run_in_cpu_slurm_allocation(cpu_parent, monkeypatch):
    from agent.harness import pbs_gpu, session_sandbox

    record, store, queue = cpu_parent
    monkeypatch.delenv("VENUS_RESEARCH_GPU_BACKEND", raising=False)
    assert pbs_gpu.compute_options()["training_devices"] == ["cpu"]
    workspace = store.root / "workspace"
    (workspace / "must_not_run.py").write_text("from pathlib import Path\nPath('executed').write_text('bad')\n")

    async def exercise():
        async with pbs_goal_queue.queue_run(record):
            with pytest.raises(SandboxUnavailable):
                await session_sandbox.execute(
                    workspace, store.root / "sandbox-control", "python", "must_not_run.py", 20, device="cuda"
                )
            assert not (workspace / "executed").exists()
            store.mutate(lambda state: state.update(status="completed"))

    asyncio.run(exercise())


def test_ordinary_queue_does_not_load_agent_or_sandbox_stack(tmp_path):
    import subprocess
    import sys

    environment = dict(os.environ)
    for key in (subject.RECORD_ENV, subject.LEGACY_RECORD_ENV):
        environment.pop(key, None)
    program = """
import asyncio
import os
import sys
from backend.web_v2.job_queue import JobQueue, resource_slot
queue = JobQueue(sys.argv[1])
first = queue.submit('ordinary', 'test', 'cpu')
queue.cancel(first)
second, _ = queue.enqueue('ordinary', 'test', 'cpu', {'kind': 'fixture'}, 'ordinary')
claimed = queue.claim_next('ordinary', 'cpu')
assert claimed['id'] == second
queue.finish(second, 'completed', lease=claimed['lease'])
async def check():
    async with resource_slot('cpu'):
        pass
asyncio.run(check())
assert not any(name == 'agent' or name.startswith('agent.') for name in sys.modules)
"""
    environment["VENUS_JOB_QUEUE_ROOT"] = str(tmp_path / "queue")
    result = subprocess.run(
        [sys.executable, "-c", program, str(tmp_path / "queue")],
        env=environment,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr
