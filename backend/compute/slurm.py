"""Single-node Slurm adapter, independent of research/HTTP implementation.

Commands follow SchedMD's sbatch/scontrol/sacct/squeue interfaces. Tests use
synthetic command fixtures: a live site's accounting and allocation enforcement
must be verified before this adapter is enabled. Scheduler identity is not a
GPU isolation receipt. Durable attempt state and retries belong to the caller.
"""
from __future__ import annotations

import hashlib
import os
import re
import stat
import subprocess
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

from .access import require_local_execution
from .contracts import (
    BoundedRunner,
    JobRef,
    Observation,
    ResourceRequest,
    SubmissionBinding,
    WorkerLaunch,
    absolute_path,
    name,
)

_SCRIPT = b'#!/bin/sh\n# The controller supplies a frozen Python entrypoint, never shell text.\nexec "$@"\n'
_TERMINAL = frozenset({"COMPLETED", "CANCELLED", "FAILED", "TIMEOUT", "NODE_FAIL",
                       "OUT_OF_MEMORY", "PREEMPTED", "BOOT_FAIL", "DEADLINE", "REVOKED"})
_ACCT_FIELDS = ("JobIDRaw", "Cluster", "UID", "JobName", "State", "Submit", "Partition",
                "ReqCPUS", "ReqNodes", "ReqMem", "TimelimitRaw", "ReqTRES", "ExitCode", "Restarts")
_COMMANDS = frozenset({"sbatch", "squeue", "scontrol", "sacct", "scancel"})


def _integer(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9]{1,18}", value):
        raise ValueError("Incomplete numeric scheduler evidence")
    return int(value)


def _timestamp(value):
    if not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}", value):
        raise ValueError("Incomplete submission timestamp")
    return int(datetime.fromisoformat(value).replace(tzinfo=UTC).timestamp())


def _utc(value):
    return datetime.fromtimestamp(value, UTC).strftime("%Y-%m-%dT%H:%M:%S")


def _memory(value):
    # Requests use --mem, not --mem-per-cpu; a per-CPU suffix cannot be reinterpreted.
    match = re.fullmatch(r"([0-9]+)([KMGT]?)(n?)", value)
    if not match:
        raise ValueError("Unsupported memory evidence")
    amount = int(match[1])
    units = {"": 1, "K": 1 / 1024, "M": 1, "G": 1024, "T": 1024 * 1024}
    mib = amount * units[match[2]]
    if not float(mib).is_integer():
        raise ValueError("Memory evidence is not whole MiB")
    return int(mib)


def _duration(value):
    match = re.fullmatch(r"(?:(\d+)-)?(\d+):([0-5][0-9]):([0-5][0-9])", value)
    if not match:
        raise ValueError("A finite Slurm time limit is required")
    return int(match[1] or 0) * 86400 + int(match[2]) * 3600 + int(match[3]) * 60 + int(match[4])


def _gpu_count(tres):
    values = {}
    for item in tres.split(","):
        key, sep, value = item.partition("=")
        if not sep or key in values:
            raise ValueError("Ambiguous requested resources")
        values[key] = value
    if "gres/gpu" not in values and any(key.startswith("gres/gpu:") for key in values):
        raise ValueError("Missing aggregate GPU count")
    return _integer(values.get("gres/gpu", "0"))


def _state(value, *, held=False):
    if value in _TERMINAL:
        return "terminal"
    if value == "PENDING":
        return "held" if held else "queued"
    if value == "CONFIGURING":
        return "queued"
    if value in {"RUNNING", "COMPLETING", "SUSPENDED"}:
        return "running"
    return "unknown"


class SlurmAdapter:
    """Uses only an injected bounded runner; never starts a shell or retries."""

    def __init__(self, cluster: str, bin_directory: Path, runner: BoundedRunner):
        require_local_execution()
        self.cluster = name(cluster)
        self.bin_directory = absolute_path(bin_directory)
        self.runner = runner

    def _binding(self, binding, ref=None):
        require_local_execution()
        if binding.cluster != self.cluster or binding.uid != os.getuid():
            raise ValueError("Submission does not belong to this local scheduler owner")
        if ref is not None and (ref.backend != "slurm" or ref.cluster != self.cluster):
            raise ValueError("Job reference belongs to another scheduler")

    def _run(self, command, arguments):
        require_local_execution()
        if command not in _COMMANDS:
            raise ValueError("Unknown scheduler command")
        result = self.runner([str(self.bin_directory / command), *arguments], timeout=20,
                             env={"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C", "TZ": "UTC",
                                  "SLURM_TIME_FORMAT": "%Y-%m-%dT%H:%M:%S"},
                             capture_output=True, text=True)
        if (not isinstance(result.stdout, str) or not isinstance(result.stderr, str)
                or len(result.stdout.encode()) > 1024 * 1024 or len(result.stderr.encode()) > 65536):
            raise RuntimeError("Scheduler output exceeded the observation bound")
        return result

    def _scope(self):
        return ["--clusters=" + self.cluster]

    def _matches(self, binding, ref, identity, cluster, uid, job_name, submitted):
        if (cluster != self.cluster or uid != binding.uid or job_name != binding.job_name
                or not binding.submitted_after <= submitted <= binding.submitted_before
                or (ref is not None and identity != ref.job_id)):
            raise ValueError("Scheduler identity does not match the submission binding")
        return JobRef("slurm", cluster, identity)

    def _active(self, text, binding, ref):
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if len(lines) != 1:
            raise ValueError("Expected one active allocation")
        fields = {}
        for match in re.finditer(r"(?:^|\s)([A-Za-z][A-Za-z0-9_/]*)=([^\s]*)", lines[0]):
            key, value = match.groups()
            if key in fields:
                raise ValueError("Duplicate scheduler fields")
            fields[key] = value
        owner = re.fullmatch(r"[^()\s]+\(([0-9]+)\)", fields["UserId"])
        if not owner:
            raise ValueError("Missing operating-system owner")
        submitted = _timestamp(fields["SubmitTime"])
        identity = self._matches(binding, ref, fields["JobId"], fields.get("Cluster", self.cluster),
                                 int(owner[1]), fields.get("JobName", fields.get("Name")), submitted)
        if fields.get("Comment", binding.nonce) != binding.nonce:
            raise ValueError("Submission comment differs from its nonce")
        if fields.get("Requeue") != "0" or fields.get("Restarts") != "0":
            raise ValueError("Automatic requeue or restart is not admitted")
        if any(key in fields for key in ("ArrayJobId", "ArrayTaskId", "HetJobId")):
            raise ValueError("Array and heterogeneous allocations are unsupported")
        if fields["NumNodes"] != "1" or fields["NumTasks"] != "1":
            raise ValueError("Only one node and one task are admitted")
        resources = ResourceRequest(fields["Partition"], _integer(fields["NumCPUs"]),
                                    _gpu_count(fields.get("ReqTRES", fields.get("TRES", ""))),
                                    _memory(fields["MinMemoryNode"]), _duration(fields["TimeLimit"]))
        state = fields["JobState"]
        held = fields.get("Priority") == "0" and fields.get("Reason") == "JobHeldUser"
        return Observation(_state(state, held=held), identity, state, resources, submitted,
                           fields.get("ExitCode"), evidence_sha256=hashlib.sha256(text.encode()).hexdigest())

    def _accounting(self, text, binding, ref):
        rows = [line.split("|") for line in text.splitlines() if line.strip()]
        if len(rows) > 128:
            raise ValueError("Too many accounting candidates")
        found = []
        for row in rows:
            if len(row) != len(_ACCT_FIELDS):
                raise ValueError("Incomplete accounting record")
            fields = dict(zip(_ACCT_FIELDS, row))
            if fields["JobName"] != binding.job_name:
                continue
            submitted = _timestamp(fields["Submit"])
            identity = self._matches(binding, ref, fields["JobIDRaw"], fields["Cluster"],
                                     _integer(fields["UID"]), fields["JobName"], submitted)
            state = fields["State"]
            if re.fullmatch(r"CANCELLED by [0-9]+", state):
                state = "CANCELLED"
            if state not in _TERMINAL or fields["Restarts"] != "0" or fields["ReqNodes"] != "1":
                raise ValueError("Accounting does not establish a terminal allocation")
            if not re.fullmatch(r"[0-9]+:[0-9]+", fields["ExitCode"]):
                raise ValueError("Incomplete scheduler exit evidence")
            resources = ResourceRequest(fields["Partition"], _integer(fields["ReqCPUS"]),
                                        _gpu_count(fields["ReqTRES"]), _memory(fields["ReqMem"]),
                                        _integer(fields["TimelimitRaw"]) * 60)
            found.append(Observation("terminal", identity, state, resources, submitted, fields["ExitCode"],
                                     evidence_sha256=hashlib.sha256(text.encode()).hexdigest()))
        if len(found) != 1:
            raise ValueError("Missing or ambiguous accounting evidence")
        return found[0]

    def submit_held(self, binding: SubmissionBinding, resources: ResourceRequest,
                    launch: WorkerLaunch) -> Observation:
        self._binding(binding)
        if resources.wall_seconds % 60:
            raise ValueError("Slurm resource profiles must use whole minutes without implicit rounding")
        script = Path(__file__).with_name("slurm_worker.sh")
        fd = os.open(script, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, "rb") as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or stream.read(1024) != _SCRIPT:
                raise ValueError("The fixed worker launcher changed")
        args = [*self._scope(), "--hold", "--parsable", "--no-requeue", "--export=NIL",
                "--nodes=1", "--ntasks=1", "--cpus-per-task=" + str(resources.ncpus),
                "--mem=" + str(resources.memory_mib) + "M", "--time=" + str(resources.wall_seconds // 60),
                "--partition=" + resources.partition, "--job-name=" + binding.job_name,
                "--comment=" + binding.nonce, "--chdir=" + str(launch.working_directory),
                "--output=" + str(launch.stdout), "--error=" + str(launch.stdout), "--input=/dev/null"]
        if resources.ngpus:
            args.append("--gres=gpu:" + str(resources.ngpus))
        args += [str(script), str(launch.python), "-B", str(launch.entrypoint), "--record", str(launch.record)]
        ref = None
        try:
            result = self._run("sbatch", args)
            match = re.fullmatch(r"([1-9][0-9]{0,18})(?:;([A-Za-z0-9_.-]+))?", result.stdout.strip())
            if result.returncode or not match or match[2] not in (None, self.cluster):
                return Observation("unknown", reason="submission_receipt_uncertain")
            ref = JobRef("slurm", self.cluster, match[1])
            observation = self.observe(binding, ref)
            if observation.state != "held":
                return Observation("unknown", ref, reason="submission_hold_unverified")
            self.validate_resources(observation, resources)
            return observation
        except PermissionError:
            raise
        except (OSError, RuntimeError, subprocess.TimeoutExpired, ValueError):
            return Observation("unknown", ref, reason="submission_uncertain")

    def observe(self, binding: SubmissionBinding, ref: JobRef | None = None) -> Observation:
        self._binding(binding, ref)
        try:
            if ref is None:
                listing = self._run("squeue", [*self._scope(), "--noheader", "--format=%A",
                                               "--user=" + str(binding.uid), "--name=" + binding.job_name])
                if listing.returncode:
                    return Observation("unknown", reason="queue_query_failed")
                identities = [line.strip() for line in listing.stdout.splitlines() if line.strip()]
                if len(identities) > 1:
                    return Observation("unknown", reason="multiple_submission_candidates")
                if identities:
                    ref = JobRef("slurm", self.cluster, identities[0])
            if ref is not None:
                active = self._run("scontrol", [*self._scope(), "--oneliner", "show", "job", ref.job_id])
                if active.returncode == 0 and active.stdout.strip():
                    return self._active(active.stdout, binding, ref)
            args = [*self._scope(), "--allocations", "--duplicates", "--noheader", "--parsable2", "--noconvert",
                    "--user=" + str(binding.uid), "--name=" + binding.job_name,
                    "--starttime=" + _utc(binding.submitted_after),
                    "--format=" + ",".join(field + "%64" if field in {"JobName", "State", "ReqTRES"} else field
                                           for field in _ACCT_FIELDS)]
            if ref is not None:
                args.append("--jobs=" + ref.job_id)
            accounting = self._run("sacct", args)
            if accounting.returncode:
                return Observation("unknown", ref, reason="accounting_query_failed")
            return self._accounting(accounting.stdout, binding, ref)
        except PermissionError:
            raise
        except (OSError, RuntimeError, subprocess.TimeoutExpired, ValueError, KeyError):
            return Observation("unknown", ref, reason="observation_unverified")

    def validate_resources(self, observation: Observation, resources: ResourceRequest) -> None:
        require_local_execution()
        if observation.state == "unknown" or observation.resources != resources:
            raise ValueError("Observed allocation differs from the admitted resource profile")

    def release(self, binding: SubmissionBinding, ref: JobRef, resources: ResourceRequest) -> Observation:
        self._binding(binding, ref)
        observed = self.observe(binding, ref)
        if observed.state != "held":
            raise ValueError("Only a freshly verified user-held allocation can be released")
        self.validate_resources(observed, resources)
        try:
            result = self._run("scontrol", [*self._scope(), "release", ref.job_id])
            if result.returncode:
                return Observation("unknown", ref, reason="release_uncertain")
        except PermissionError:
            raise
        except (OSError, RuntimeError, subprocess.TimeoutExpired):
            return Observation("unknown", ref, reason="release_uncertain")
        return self.observe(binding, ref)

    def cancel(self, binding: SubmissionBinding, ref: JobRef) -> Observation:
        self._binding(binding, ref)
        observed = self.observe(binding, ref)
        if observed.state == "terminal":
            return observed
        if observed.state == "unknown":
            raise ValueError("Cancellation requires a freshly verified submission identity")
        try:
            result = self._run("scancel", [*self._scope(), ref.job_id])
            if result.returncode:
                return Observation("unknown", ref, reason="cancellation_uncertain")
        except PermissionError:
            raise
        except (OSError, RuntimeError, subprocess.TimeoutExpired):
            return Observation("unknown", ref, reason="cancellation_uncertain")
        return self.observe(binding, ref)

    def worker_identity(self, binding: SubmissionBinding, ref: JobRef,
                        environment: Mapping[str, str]) -> Observation:
        self._binding(binding, ref)
        if (environment.get("SLURM_JOB_ID") != ref.job_id
                or environment.get("SLURM_CLUSTER_NAME") != ref.cluster):
            raise ValueError("Worker environment differs from its assigned job")
        observed = self.observe(binding, ref)
        if observed.state != "running" or observed.scheduler_state != "RUNNING":
            raise ValueError("Worker does not own a verified running allocation")
        return observed
