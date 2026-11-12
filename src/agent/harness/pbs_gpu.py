"""Opt-in GPU execution inside an existing, isolated PBS research allocation.

PBS owns the device and kernel memory limit. The normal sandbox admission,
process cleanup, output limits and syscall policy still apply. No qsub, host
execution fallback, credential forwarding or online GPU code execution occurs.
"""

from __future__ import annotations

import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from .session_sandbox import SandboxUnavailable


@dataclass(frozen=True)
class Allocation:
    job_id: str
    gpu_uuid: str
    device: str
    memory_limit: int

    @property
    def devices(self):
        return (self.device, "/dev/nvidiactl", "/dev/nvidia-uvm")

    def receipt(self):
        return {
            "backend": "pbs",
            "device": "cuda",
            "job_id": self.job_id,
            "gpu_uuid": self.gpu_uuid,
            "device_node": self.device,
            "memory_hard_limit_bytes": self.memory_limit,
            "memory_scope": "whole PBS allocation",
            "inference_device": "cpu",
        }


def capture(limits):
    """Reject ambiguous devices, unbounded memory and allocations outside PBS."""
    if os.getenv("WEBUI_V2_MODE") == "online" or os.getenv("VENUS_SCIENTIST_ONLINE") == "1":
        raise SandboxUnavailable("Authored GPU training is available only in local PBS research jobs")
    if os.getenv("VENUS_RESEARCH_GPU_BACKEND") != "pbs":
        raise SandboxUnavailable("GPU training requires an operator-configured PBS research allocation")
    job = os.getenv("PBS_JOBID", "")
    visible = os.getenv("CUDA_VISIBLE_DEVICES", "")
    if not re.fullmatch(r"[0-9]+(?:\[[0-9]+\])?\.[A-Za-z0-9_.-]+", job):
        raise SandboxUnavailable("A valid PBS job identity is required for GPU training")
    if not re.fullmatch(r"GPU-[a-fA-F0-9]{8}(?:-[a-fA-F0-9]{4}){3}-[a-fA-F0-9]{12}", visible):
        raise SandboxUnavailable("PBS must assign exactly one full GPU UUID")
    controllers = {}
    for line in Path("/proc/self/cgroup").read_text().splitlines():
        _, names, relative = line.split(":", 2)
        for name in names.split(","):
            controllers[name] = relative
    # Initial backend supports the site's audited cgroup v1 PBS hook. An
    # unverified v2 layout must not silently disable memory/device enforcement.
    locations = [controllers.get(name, "") for name in ("memory", "devices")]
    for relative in locations:
        parts = PurePosixPath(relative).parts
        if (
            not relative.startswith("/")
            or ".." in parts
            or not parts
            or parts[-1] != job
            or not any("pbs" in part.lower() for part in parts[:-1])
        ):
            raise SandboxUnavailable("PBS memory and device cgroups could not be verified")
    memory = Path("/sys/fs/cgroup/memory") / locations[0].lstrip("/")
    devices = Path("/sys/fs/cgroup/devices") / locations[1].lstrip("/")
    for folder in (memory, devices):
        if str(os.getpid()) not in (folder / "cgroup.procs").read_text().split():
            raise SandboxUnavailable("Worker is not a member of its PBS allocation")
    cap = int((memory / "memory.limit_in_bytes").read_text())
    if not 2 * 1024**3 <= cap <= min(64 * 1024**3, limits.reserved):
        raise SandboxUnavailable("PBS memory limit must be finite and fit the sandbox total-memory budget")
    if limits.memory + limits.reserve_free >= cap:
        raise SandboxUnavailable("PBS allocation must exceed the sandbox resident-memory limit plus reserve")
    matches = []
    for info in Path("/proc/driver/nvidia/gpus").glob("*/information"):
        fields = dict(line.split(":", 1) for line in info.read_text().splitlines() if ":" in line)
        if fields.get("GPU UUID", "").strip() == visible:
            matches.append(fields.get("Device Minor", "").strip())
    if len(matches) != 1 or not re.fullmatch(r"[0-9]{1,3}", matches[0]):
        raise SandboxUnavailable("Assigned GPU UUID has no unambiguous physical-device mapping")
    allocation = Allocation(job, visible, "/dev/nvidia" + matches[0], cap)
    permissions = (devices / "devices.list").read_text().splitlines()
    for device in allocation.devices:
        info = Path(device).stat()
        if not stat.S_ISCHR(info.st_mode):
            raise SandboxUnavailable("Assigned CUDA device is unavailable")
        major, minor = os.major(info.st_rdev), os.minor(info.st_rdev)
        if device == allocation.device and minor != int(matches[0]):
            raise SandboxUnavailable("Assigned CUDA device identity changed")
        exact = any(
            row.split()[:2] == ["c", f"{major}:{minor}"] and "r" in row.split()[-1] and "w" in row.split()[-1]
            for row in permissions
        )
        broad = any(row.split()[:2] in (["a", "*:*"], ["c", "*:*"], ["c", f"{major}:*"]) for row in permissions)
        if not exact or broad:
            raise SandboxUnavailable("PBS must enforce explicit CUDA-device permissions")
        descriptor = os.open(device, os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW)
        os.close(descriptor)
    return allocation


def process_rss(pid):
    """Resident memory of the sandbox supervisor and its private process tree."""
    pending, seen, total = [pid], set(), 0
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        if len(seen) > 64:
            raise MemoryError("Sandbox process-tree limit reached")
        try:
            status = Path(f"/proc/{current}/status").read_text()
            total += next((int(line.split()[1]) * 1024 for line in status.splitlines() if line.startswith("VmRSS:")), 0)
            pending.extend(int(child) for child in Path(f"/proc/{current}/task/{current}/children").read_text().split())
        except FileNotFoundError:
            continue
    return total


async def run_python(workspace, control, script, timeout=120, *, on_progress=None):
    from backend.web_v2.job_queue import resource_slot

    from .research_workspace import safe_file, workspace_hash
    from .session_sandbox import execute

    safe_file(workspace, script)

    def waiting(job):
        if on_progress:
            on_progress(
                {
                    "status": "queued",
                    "tool": "train_research_model",
                    "script": script,
                    "elapsed_seconds": 0,
                    "stdout_tail": "",
                    "stderr_tail": "",
                    "queue": job,
                }
            )

    async with resource_slot("gpu", kind="research:training", on_wait=waiting):
        result = await execute(workspace, control, "python", script, timeout, on_progress=on_progress, device="cuda")
    result["script_sha256"] = workspace_hash(workspace, script)
    return result
