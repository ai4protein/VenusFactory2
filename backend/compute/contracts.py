"""Small scheduler contract; goal ownership, snapshots and leases remain in the controller."""
from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


class CommandResult(Protocol):
    returncode: int
    stdout: str
    stderr: str


class BoundedRunner(Protocol):
    def __call__(self, argv: list[str], *, timeout: float, env: dict[str, str],
                 capture_output: bool = True, text: bool = True) -> CommandResult: ...


def name(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", value):
        raise ValueError("Use one administrator-defined scheduler name")
    return value


def absolute_path(value: Path) -> Path:
    path = Path(value)
    if not path.is_absolute() or any(c in str(path) for c in "\r\n\x00%") or ".." in path.parts:
        raise ValueError("Worker paths must be absolute host-selected paths")
    return path


@dataclass(frozen=True)
class JobRef:
    backend: str
    cluster: str
    job_id: str

    def __post_init__(self):
        if self.backend not in {"pbs", "slurm"}:
            raise ValueError("Unsupported scheduler backend")
        name(self.cluster)
        pattern = r"[1-9][0-9]{0,18}" if self.backend == "slurm" else r"[0-9]+\.[A-Za-z0-9_.-]+"
        if not isinstance(self.job_id, str) or not re.fullmatch(pattern, self.job_id):
            raise ValueError("Use one complete scheduler job identity; arrays are unsupported")


@dataclass(frozen=True)
class SubmissionBinding:
    cluster: str
    nonce: str
    uid: int
    submitted_after: int
    submitted_before: int

    def __post_init__(self):
        name(self.cluster)
        if not isinstance(self.nonce, str) or not re.fullmatch(r"[a-f0-9]{32}", self.nonce):
            raise ValueError("A complete host-generated submission nonce is required")
        if type(self.uid) is not int or self.uid < 0:
            raise ValueError("A numeric operating-system owner is required")
        if (type(self.submitted_after) is not int or type(self.submitted_before) is not int
                or not 0 <= self.submitted_after <= self.submitted_before
                or self.submitted_before - self.submitted_after > 300):
            raise ValueError("Bind submission to a finite, at most five-minute UTC window")

    @property
    def job_name(self) -> str:
        return "vf_" + self.nonce


@dataclass(frozen=True)
class ResourceRequest:
    partition: str
    ncpus: int
    ngpus: int
    memory_mib: int
    wall_seconds: int

    def __post_init__(self):
        name(self.partition)
        for field, low, high in [("ncpus", 1, 8), ("ngpus", 0, 1),
                                 ("memory_mib", 16384, 65536), ("wall_seconds", 30, 86400)]:
            value = getattr(self, field)
            if type(value) is not int or not low <= value <= high:
                raise ValueError("Resource request exceeds supported single-node bounds")


@dataclass(frozen=True)
class WorkerLaunch:
    """Trusted controller inputs, never a public shell/command API.

    The controller must bind entrypoint and record to its source snapshot and
    submission ledger before constructing this value. Python symlink spelling
    is preserved so a virtual environment is not replaced by its base Python.
    """
    python: Path
    entrypoint: Path
    record: Path
    working_directory: Path
    stdout: Path

    def __post_init__(self):
        for field in ("python", "entrypoint", "record", "working_directory", "stdout"):
            absolute_path(getattr(self, field))


@dataclass(frozen=True)
class Observation:
    state: str
    ref: JobRef | None = None
    scheduler_state: str | None = None
    resources: ResourceRequest | None = None
    submitted_at: int | None = None
    exit_code: str | None = None
    reason: str | None = None
    evidence_sha256: str | None = None

    def __post_init__(self):
        if self.state not in {"held", "queued", "running", "terminal", "unknown"}:
            raise ValueError("Invalid normalized scheduler state")


class SchedulerAdapter(Protocol):
    def submit_held(self, binding: SubmissionBinding, resources: ResourceRequest,
                    launch: WorkerLaunch) -> Observation: ...
    def observe(self, binding: SubmissionBinding, ref: JobRef | None = None) -> Observation: ...
    def validate_resources(self, observation: Observation, resources: ResourceRequest) -> None: ...
    def release(self, binding: SubmissionBinding, ref: JobRef,
                resources: ResourceRequest) -> Observation: ...
    def cancel(self, binding: SubmissionBinding, ref: JobRef) -> Observation: ...
    def worker_identity(self, binding: SubmissionBinding, ref: JobRef,
                        environment: Mapping[str, str]) -> Observation: ...
