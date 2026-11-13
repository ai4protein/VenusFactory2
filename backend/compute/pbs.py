"""OpenPBS adapter preserving held-submit and full legacy nonce/owner binding."""
from __future__ import annotations

import hashlib
import json
import os
import pwd
import re
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

NONCE_ENV = "VENUS_PBS_GOAL_ATTEMPT"
_COMMANDS = frozenset({"qsub", "qstat", "qrls", "qdel"})


def _creation_time(value):
    if type(value) is int and value >= 0:
        return value
    if not isinstance(value, str):
        raise ValueError("Incomplete PBS submission timestamp")
    match = re.fullmatch(
        r"(Mon|Tue|Wed|Thu|Fri|Sat|Sun) (Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
        r" {1,2}([0-9]{1,2}) ([0-2][0-9]):([0-5][0-9]):([0-5][0-9]) ([0-9]{4})", value,
    )
    if not match:
        raise ValueError("Unsupported PBS submission timestamp")
    months = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
    stamp = datetime(int(match[7]), months.index(match[2]) + 1, int(match[3]),
                     int(match[4]), int(match[5]), int(match[6]), tzinfo=UTC)
    if ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")[stamp.weekday()] != match[1]:
        raise ValueError("PBS timestamp weekday differs from its date")
    return int(stamp.timestamp())


def _resources(row):
    values = row.get("Resource_List")
    if not isinstance(values, dict) or str(values.get("nodect")) != "1":
        raise ValueError("PBS resource evidence requires one compute node")
    selection = str(values.get("select", ""))
    if not selection.startswith("1:") or "+" in selection:
        raise ValueError("PBS resource evidence requires one resource chunk")
    if row.get("Rerunable") not in (False, "False", "false"):
        raise ValueError("PBS allocations must not rerun automatically")
    counts = []
    for field in ("ncpus", "ngpus"):
        value = values.get(field)
        if isinstance(value, bool) or not re.fullmatch(r"[0-9]+", str(value)):
            raise ValueError("Incomplete PBS resource counts")
        counts.append(int(value))
    memory = re.fullmatch(r"([0-9]+)(b|kb|mb|gb|tb)", str(values.get("mem", "")).lower())
    if not memory:
        raise ValueError("Incomplete PBS memory evidence")
    size = int(memory[1]) * 1024 ** ("b", "kb", "mb", "gb", "tb").index(memory[2])
    if size % (1024 ** 2):
        raise ValueError("PBS memory must be whole MiB")
    wall = re.fullmatch(r"([0-9]+):([0-5][0-9]):([0-5][0-9])", str(values.get("walltime", "")))
    if not wall:
        raise ValueError("Incomplete PBS time limit")
    seconds = int(wall[1]) * 3600 + int(wall[2]) * 60 + int(wall[3])
    return ResourceRequest(row.get("queue"), counts[0], counts[1], size // 1024 ** 2, seconds)


class PBSAdapter:
    """Cluster names identify configured local endpoints, not inferred server aliases.

    Only existing records without scheduler metadata may use legacy_identity.
    Those records retain the original full nonce/name/Unix-owner check without
    inventing a submission time that was never recorded.
    """

    def __init__(self, cluster: str, bin_directory: Path, runner: BoundedRunner, *, legacy_identity=False):
        require_local_execution()
        if type(legacy_identity) is not bool:
            raise ValueError("Legacy compatibility is an explicit controller decision")
        self.cluster = name(cluster)
        self.bin_directory = absolute_path(bin_directory)
        self.runner = runner
        self.legacy_identity = legacy_identity

    def _binding(self, binding, ref=None):
        require_local_execution()
        if binding.cluster != self.cluster or binding.uid != os.getuid():
            raise ValueError("Submission does not belong to this local scheduler owner")
        if ref is not None and (ref.backend != "pbs" or ref.cluster != self.cluster):
            raise ValueError("Job reference belongs to another scheduler")

    def _run(self, command, arguments):
        require_local_execution()
        if command not in _COMMANDS:
            raise ValueError("Unknown scheduler command")
        user = pwd.getpwuid(os.getuid())
        env = {"HOME": user.pw_dir, "USER": user.pw_name, "LOGNAME": user.pw_name,
               "PATH": str(self.bin_directory) + ":/usr/bin:/bin", "LANG": "C", "LC_ALL": "C", "TZ": "UTC"}
        result = self.runner([str(self.bin_directory / command), *arguments], timeout=30,
                             env=env, capture_output=True, text=True)
        if (not isinstance(result.stdout, str) or not isinstance(result.stderr, str)
                or len(result.stdout.encode()) > 16 * 1024 * 1024 or len(result.stderr.encode()) > 65536):
            raise RuntimeError("Scheduler output exceeded the observation bound")
        return result

    def observe(self, binding: SubmissionBinding, ref: JobRef | None = None) -> Observation:
        self._binding(binding, ref)
        try:
            result = self._run("qstat", ["-x", "-f", "-F", "json", *([ref.job_id] if ref else [])])
            if result.returncode:
                return Observation("unknown", ref, reason="status_query_failed")
            rows = json.loads(result.stdout).get("Jobs")
            if not isinstance(rows, dict):
                raise ValueError("Invalid PBS job inventory")
            owner = pwd.getpwuid(binding.uid).pw_name
            matched = []
            for identity, row in rows.items():
                if not isinstance(row, dict):
                    continue
                variables = row.get("Variable_List")
                matches = (isinstance(variables, dict) and variables.get(NONCE_ENV) == binding.nonce
                           and row.get("Job_Name") == binding.job_name
                           and str(row.get("Job_Owner", "")).split("@", 1)[0] == owner)
                if ref is not None and ref.job_id == identity and not matches:
                    raise ValueError("PBS identity differs from its saved submission")
                if not matches:
                    continue
                if ref is not None and ref.job_id != identity:
                    raise ValueError("Another job matches the same submission nonce")
                candidate = JobRef("pbs", self.cluster, identity)
                try:
                    submitted = _creation_time(row.get("ctime"))
                except ValueError:
                    if not self.legacy_identity:
                        raise
                    submitted = None
                if not self.legacy_identity and not binding.submitted_after <= submitted <= binding.submitted_before:
                    raise ValueError("PBS submission timestamp differs from its binding")
                state = row.get("job_state")
                mapping = {"H": "held", "Q": "queued", "W": "queued", "T": "queued",
                           "R": "running", "E": "running", "F": "terminal"}
                normalized = mapping.get(state, "unknown")
                if state == "H" and "u" not in str(row.get("Hold_Types", "")):
                    normalized = "queued"  # Never release an administrator-only hold.
                try:
                    resources = _resources(row)
                except (ValueError, TypeError):
                    # Preserve exact ownership for cancellation, never grant execution.
                    resources = None
                exit_code = row.get("Exit_status")
                if exit_code is not None and type(exit_code) is not int:
                    raise ValueError("Invalid PBS exit status")
                matched.append(Observation(normalized, candidate, state, resources, submitted,
                                           str(exit_code) if exit_code is not None else None,
                                           evidence_sha256=hashlib.sha256(result.stdout.encode()).hexdigest()))
            if len(matched) != 1:
                raise ValueError("Missing or ambiguous PBS submission")
            return matched[0]
        except PermissionError:
            raise
        except (OSError, RuntimeError, subprocess.TimeoutExpired, ValueError, TypeError, KeyError):
            return Observation("unknown", ref, reason="observation_unverified")

    def submit_held(self, binding: SubmissionBinding, resources: ResourceRequest,
                    launch: WorkerLaunch) -> Observation:
        self._binding(binding)
        if self.legacy_identity:
            raise ValueError("Legacy identity mode cannot create new submissions")
        wall = resources.wall_seconds
        args = ["-h", "-r", "n", "-m", "n", "-N", binding.job_name, "-q", resources.partition,
                "-l", f"select=1:ncpus={resources.ncpus}:ngpus={resources.ngpus}:mem={resources.memory_mib}mb",
                "-l", f"walltime={wall // 3600:02}:{wall % 3600 // 60:02}:{wall % 60:02}",
                "-j", "oe", "-o", str(launch.stdout), "-v", NONCE_ENV + "=" + binding.nonce,
                "--", str(launch.python), "-B", str(launch.entrypoint), "--record", str(launch.record)]
        ref = None
        try:
            result = self._run("qsub", args)
            if result.returncode:
                return Observation("unknown", reason="submission_receipt_uncertain")
            ref = JobRef("pbs", self.cluster, result.stdout.strip())
            observed = self.observe(binding, ref)
            if observed.state != "held":
                return Observation("unknown", ref, reason="submission_hold_unverified")
            self.validate_resources(observed, resources)
            return observed
        except PermissionError:
            raise
        except (OSError, RuntimeError, subprocess.TimeoutExpired, ValueError):
            return Observation("unknown", ref, reason="submission_uncertain")

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
            result = self._run("qrls", ["-h", "u", ref.job_id])
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
            result = self._run("qdel", [ref.job_id])
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
        if environment.get("PBS_JOBID") != ref.job_id:
            raise ValueError("Worker environment differs from its assigned job")
        observed = self.observe(binding, ref)
        if observed.state != "running" or observed.scheduler_state != "R" or observed.resources is None:
            raise ValueError("Worker does not own a verified running allocation")
        return observed
