"""Host-verified cluster allocation boundaries; Slurm initially supports CPU only.

Scheduler identity and kernel resource enforcement are separate proofs. Neither
an environment variable nor a requested GPU count authorizes a device.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from .session_sandbox import SandboxUnavailable

PROC = Path("/proc")
CGROUP = Path("/sys/fs/cgroup")
RECORD_ENV = "VENUS_SCIENTIST_CLUSTER_RECORD"
LEGACY_RECORD_ENV = "VENUS_SCIENTIST_PBS_RECORD"


def backend(record):
    scheduler = record.get("scheduler")
    if scheduler is None:
        return "pbs"
    if not isinstance(scheduler, dict) or scheduler.get("backend") not in {"pbs", "slurm"}:
        raise SandboxUnavailable("Unknown cluster scheduler binding")
    return scheduler["backend"]


def worker_job_matches(record, environment=None):
    env = os.environ if environment is None else environment
    kind = backend(record)
    if record.get("uid") != os.getuid() or not record.get("job_id"):
        return False
    if kind == "pbs":
        return env.get("PBS_JOBID") == record["job_id"]
    scheduler = record["scheduler"]
    return (env.get("SLURM_JOB_ID") == record["job_id"]
            and env.get("SLURM_CLUSTER_NAME") == scheduler.get("cluster"))


def record_environment_matches(record):
    current, legacy = os.getenv(RECORD_ENV), os.getenv(LEGACY_RECORD_ENV)
    path = record.get("record_path")
    if backend(record) == "slurm":
        return current == path and not legacy
    return (current or legacy) == path and (not current or not legacy or current == legacy)


def _cpus(raw):
    result = set()
    for part in raw.strip().split(","):
        if not re.fullmatch(r"[0-9]+(?:-[0-9]+)?", part):
            raise SandboxUnavailable("Cluster CPU set is missing or malformed")
        numbers = [int(x) for x in part.split("-")]
        low, high = numbers[0], numbers[-1]
        if not 0 <= low <= high <= 1048575 or high - low > 65536:
            raise SandboxUnavailable("Cluster CPU set exceeds supported bounds")
        result.update(range(low, high + 1))
        if len(result) > 65536:
            raise SandboxUnavailable("Cluster CPU set exceeds supported bounds")
    return result


def _location(relative, job):
    path = PurePosixPath(relative)
    if (not relative.startswith("/") or ".." in path.parts
            or path.parts.count("job_" + job) != 1
            or not any("slurm" in x.lower() for x in path.parts)):
        raise SandboxUnavailable("Slurm job cgroup layout is not verified")
    return relative.lstrip("/")


def _kernel_cpu(record):
    """Verify current PID membership and finite job memory/CPU enforcement.

    Only conventional job_<id> layouts are supported. SLUID-only installations
    require a separately verified mapping and are deliberately rejected here.
    """
    job = record["job_id"]
    controllers = {}
    for row in (PROC / "self/cgroup").read_text().splitlines():
        _, names, location = row.split(":", 2)
        for name in names.split(","):
            if name in controllers:
                raise SandboxUnavailable("Ambiguous cluster cgroup membership")
            controllers[name] = location
    unified = "" in controllers
    if unified:
        memory = cpuset = CGROUP / _location(controllers[""], job)
        memory_name, cpu_name, version = "memory.max", "cpuset.cpus.effective", 2
    else:
        memory = CGROUP / "memory" / _location(controllers.get("memory", ""), job)
        cpuset = CGROUP / "cpuset" / _location(controllers.get("cpuset", ""), job)
        memory_name, cpu_name, version = "memory.limit_in_bytes", "cpuset.cpus", 1
    for path in (memory, cpuset):
        if str(os.getpid()) not in (path / "cgroup.procs").read_text().split():
            raise SandboxUnavailable("Worker does not belong to its Slurm resource cgroups")
    caps = []
    current = memory
    while True:
        raw = (current / memory_name).read_text().strip()
        if raw != "max":
            value = int(raw)
            if 0 < value < 2**60:
                caps.append(value)
        if current.name == "job_" + job:
            break
        current = current.parent
    if not caps:
        raise SandboxUnavailable("Slurm allocation has no finite memory limit")
    cap = min(caps)
    cpus = _cpus((cpuset / cpu_name).read_text())
    affinity = set(os.sched_getaffinity(0))
    resources = record["resources"]
    if (not affinity or not affinity.issubset(cpus) or len(cpus) > resources["ncpus"]
            or cap > resources["memory_gb"] * 1024**3):
        raise SandboxUnavailable("Slurm kernel resource limits exceed the admitted request")
    from .sandbox_resources import Limits
    limits = Limits.load()
    if cap > limits.reserved or cap <= limits.memory + limits.reserve_free:
        raise SandboxUnavailable("Slurm memory limit is incompatible with the sandbox budget")
    # A scheduler-proven CPU allocation grants no GPU. CPU sandboxes mount no
    # NVIDIA nodes; the shared tool/queue guard also rejects GPU workloads.
    return {"memory_hard_limit_bytes": cap, "cpu_ids": sorted(cpus), "cgroup_version": version,
            "memory_cgroup": str(memory), "cpuset_cgroup": str(cpuset)}


@dataclass(frozen=True)
class CPUAllocation:
    job_id: str
    evidence: dict

    def receipt(self):
        return dict(self.evidence)


def capture_allocation(record):
    from backend.compute.access import require_local_execution
    require_local_execution()
    if not worker_job_matches(record):
        raise SandboxUnavailable("Worker identity differs from its cluster submission")
    if backend(record) == "pbs":
        from .pbs_gpu import capture
        from .sandbox_resources import Limits
        allocation = capture(Limits.load())
        if allocation.job_id != record["job_id"]:
            raise SandboxUnavailable("PBS allocation differs from its submission")
        return allocation
    from .pbs_goal import worker_observation
    observation = worker_observation(record)
    from backend.compute.contracts import JobRef, Observation, ResourceRequest
    request = record["resources"]
    expected = ResourceRequest(request["queue"], request["ncpus"], request["ngpus"],
                               request["memory_gb"] * 1024, request["wall_seconds"])
    scheduler = record["scheduler"]
    if (request["ngpus"] != 0 or not isinstance(observation, Observation)
            or observation.state != "running"
            or observation.ref != JobRef("slurm", scheduler["cluster"], record["job_id"])
            or observation.resources != expected
            or not isinstance(observation.evidence_sha256, str)
            or not re.fullmatch(r"[a-f0-9]{64}", observation.evidence_sha256)
            or observation.submitted_at is None
            or not scheduler["submitted_after"] <= observation.submitted_at <= scheduler["submitted_before"]):
        raise SandboxUnavailable("Slurm CPU allocation lacks verified scheduler evidence")
    kernel = _kernel_cpu(record)
    evidence = {"backend": "slurm", "device": "cpu", "job_id": record["job_id"],
                "cluster": scheduler["cluster"], "uid": record["uid"], "nonce": record["nonce"],
                "ncpus": expected.ncpus, "ngpus": 0, "inference_device": "cpu",
                "scheduler_evidence_sha256": observation.evidence_sha256,
                "memory_scope": "whole Slurm allocation", **kernel}
    evidence["kernel_evidence_sha256"] = hashlib.sha256(json.dumps(kernel, sort_keys=True).encode()).hexdigest()
    return CPUAllocation(record["job_id"], evidence)


def assert_resource_admitted(resource):
    """Restrict an attempt-private cluster process without changing global slots."""
    path = os.getenv(RECORD_ENV) or os.getenv(LEGACY_RECORD_ENV)
    if not path:
        return
    from backend.compute.access import require_local_execution
    require_local_execution()
    from .pbs_goal_queue import _binding
    record = _binding(path)
    if backend(record) == "slurm":
        if resource.split(":", 1)[0] == "gpu":
            raise SandboxUnavailable("GPU tools are unavailable in this verified CPU-only Slurm allocation")
        kernel = _kernel_cpu(record)
        receipt = record["worker"]["allocation"]
        if any(kernel[key] != receipt.get(key) for key in kernel):
            raise SandboxUnavailable("Slurm resource enforcement changed since worker admission")
