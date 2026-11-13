"""Durable held-submit/reconcile/release lifecycle for standalone Scientist goals.

PBS identities are never interpreted as local PIDs. An uncertain qsub result
keeps its assignment until the exact owner and nonce can be reconciled.
"""
from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import pwd
import re
import stat
import subprocess
import time
import uuid
from pathlib import Path

from agent.rsi.engine.storage import atomic_json, file_lock
from agent.science_goal import GoalStore
from backend.compute.access import require_local_execution
from backend.compute.contracts import JobRef, ResourceRequest, SubmissionBinding, WorkerLaunch
from backend.compute.factory import adapter_for

from .pbs_goal_config import configuration
from .pbs_goal_transport import bounded_run

SCHEMA = "venusfactory.pbs-goal.v1"
ASSIGNMENT = ".pbs-assignment.json"
NONCE_ENV = "VENUS_PBS_GOAL_ATTEMPT"
TERMINAL = {"finished", "cancelled", "preparation_failed"}
JOB_ID = re.compile(r"[0-9]+\.[A-Za-z0-9_.-]+")


def _backend(record):
    scheduler = record.get("scheduler")
    return "pbs" if scheduler is None else scheduler["backend"]


def _binding(record):
    scheduler = record.get("scheduler")
    if scheduler is None:
        return SubmissionBinding("pbs", record["nonce"], record["uid"], 0, 0)
    return SubmissionBinding(scheduler["cluster"], record["nonce"], record["uid"],
                             scheduler["submitted_after"], scheduler["submitted_before"])


def _reference(record):
    if record.get("job_id") is None:
        return None
    return JobRef(_backend(record), _binding(record).cluster, record["job_id"])


def _resources(resource):
    return ResourceRequest(resource["queue"], resource["ncpus"], resource["ngpus"],
                           resource["memory_gb"] * 1024, resource["wall_seconds"])


def _private(path):
    path = Path(path)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise ValueError("PBS records must be private regular files owned by the current user")
        raw = stream.read(4_000_001)
        if len(raw) > 4_000_000:
            raise ValueError("PBS record is too large")
    return json.loads(raw)


def _record(path):
    record = _private(path)
    if (not isinstance(record, dict) or record.get("schema") != SCHEMA
            or record.get("uid") != os.getuid() or record.get("owner") != pwd.getpwuid(os.getuid()).pw_name
            or not isinstance(record.get("nonce"), str) or re.fullmatch(r"[a-f0-9]{32}", record["nonce"]) is None
            or Path(path).absolute() != Path(record.get("record_path", "")).absolute()):
        raise ValueError("Invalid PBS submission identity")
    return record


@contextlib.contextmanager
def _goal_lease(goal):
    with (Path(goal) / ".runner.lock").open("a") as lease:
        try:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("A runner already owns this goal") from exc
        try:
            yield
        finally:
            fcntl.flock(lease, fcntl.LOCK_UN)


def _save(record):
    record["updated_at"] = time.time()
    atomic_json(Path(record["record_path"]), record)


def _public(record):
    return {"attempt_id": record["nonce"], "job_id": record.get("job_id"), "state": record["state"],
            "backend": _backend(record), "cluster": _binding(record).cluster,
            "scheduler_state": record.get("scheduler_state", record.get("pbs_state")),
            "goal_status": GoalStore(record["goal"]).read()["status"], "pbs_state": record.get("pbs_state"),
            "exit_code": record.get("exit_code"), "cancel_requested": record["cancel_requested"],
            "worker": record.get("worker"), "error_type": record.get("error_type")}


class Controller:
    def __init__(self, settings, runner=None):
        require_local_execution()
        self.settings = configuration(settings)
        self.runner = runner or bounded_run
        self.root = Path(self.settings["control_root"])
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        info = self.root.lstat()
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise ValueError("PBS control root must be a private directory owned by the current user")

    @classmethod
    def from_file(cls, path):
        require_local_execution()
        path = Path(path).absolute()
        return cls(configuration(json.loads(path.read_text()), path.parent))

    def _goal(self, goal):
        raw = Path(goal).absolute()
        path = raw.resolve(strict=True)
        if raw != path or not any(path.is_relative_to(Path(base).resolve()) for base in self.settings["goal_roots"]):
            raise ValueError("Goal must be under an allowed goal root without symlink aliases")
        if path.stat().st_uid != os.getuid() or path.stat().st_mode & 0o022:
            raise ValueError("Goal must be owned by the current user and not writable by others")
        if not (path / "workspace").is_dir():
            raise ValueError("A prepared Scientist goal with a workspace is required")
        GoalStore(path).read()
        return path

    def _folder(self, goal):
        key = hashlib.sha256(str(goal).encode()).hexdigest()[:24]
        folder = self.root / key
        folder.mkdir(mode=0o700, exist_ok=True)
        info = folder.lstat()
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise ValueError("PBS goal ledger must be a private directory owned by the current user")
        return folder

    def _current(self, folder):
        pointer = folder / "current.json"
        if not pointer.exists():
            return None
        nonce = _private(pointer).get("attempt_id")
        if not isinstance(nonce, str) or not re.fullmatch(r"[a-f0-9]{32}", nonce):
            raise ValueError("Invalid PBS current-attempt pointer")
        return _record(folder / nonce / "submission.json")

    def _adapter(self, record):
        settings = record["settings"]
        spec = settings.get("scheduler", {"backend": "pbs", "cluster": "pbs", "bin_directory": settings.get("pbs_bin")})
        scheduler = record.get("scheduler")
        if scheduler is not None and (scheduler.get("backend"), scheduler.get("cluster")) != (spec["backend"], spec["cluster"]):
            raise ValueError("Scheduler endpoint differs from the saved submission")
        if scheduler is None and spec["backend"] != "pbs":
            raise ValueError("Slurm submissions require an explicit scheduler binding")
        return adapter_for(spec["backend"], spec["cluster"], Path(spec["bin_directory"]),
                           self.runner, legacy_pbs=scheduler is None)

    def _invoke(self, command, arguments, record):
        # Kept as a guarded compatibility seam; adapters own command allowlists.
        require_local_execution()
        return self._adapter(record)._run(command, arguments)

    def _observe_worker(self, record):
        adapter = self._adapter(record)
        reference = _reference(record)
        if reference is None:
            raise ValueError("Worker requires a recorded scheduler job identity")
        observed = adapter.worker_identity(_binding(record), reference, os.environ)
        adapter.validate_resources(observed, _resources(record["resources"]))
        return observed

    def _release_assignment(self, record):
        try:
            with _goal_lease(record["goal"]):
                marker = Path(record["goal"]) / ASSIGNMENT
                if not marker.exists() and not marker.is_symlink():
                    return
                assignment = _private(marker)
                if (assignment.get("nonce") != record["nonce"]
                        or assignment.get("record") != record["record_path"]
                        or assignment.get("definition_sha256") != record["definition_sha256"]):
                    return
                store = GoalStore(record["goal"])
                goal_state = store.read()
                if goal_state["status"] == "active" and goal_state["definition_sha256"] == record["definition_sha256"]:
                    def interrupted(state):
                        state["status"] = "interrupted"
                        if state.get("inflight") is not None:
                            state["cycles"].append({**state["inflight"], "status": "interrupted",
                                                    "summary": "PBS execution ended without a checkpoint; reservation retained"})
                            state["inflight"] = None
                        state["events"].append({"event": "pbs_execution_ended", "at": time.time(), "job_id": record["job_id"]})
                    store.mutate(interrupted)
                marker.unlink()
                record.pop("release_pending", None)
        except RuntimeError:
            record["release_pending"] = True

    def _poll(self, record, *, release=False):
        require_local_execution()
        if record["state"] == "preparation_failed":
            return record
        if record["state"] in {"finished", "cancelled"}:
            self._release_assignment(record)
            _save(record)
            return record
        adapter = self._adapter(record)
        binding = _binding(record)
        observed = adapter.observe(binding, _reference(record))
        self._observe(record, observed)
        if observed.state == "unknown":
            return record
        if observed.state == "terminal":
            self._release_assignment(record)
            _save(record)
            return record
        if record["cancel_requested"]:
            if time.time() - record.get("cancel_sent_at", 0) >= 30:
                record["cancel_sent_at"] = time.time()
                _save(record)
                observed = adapter.cancel(binding, observed.ref)
                self._observe(record, observed)
        elif release and not record.get("release_confirmed") and observed.state == "held":
            adapter.validate_resources(observed, _resources(record["resources"]))
            if time.time() - record.get("release_sent_at", 0) >= 30:
                record["release_sent_at"] = time.time()
                _save(record)
                observed = adapter.release(binding, observed.ref, _resources(record["resources"]))
                self._observe(record, observed)
        if observed.state == "terminal":
            self._release_assignment(record)
            _save(record)
        return record

    @staticmethod
    def _observe(record, observed):
        if observed.ref is not None:
            previous = _reference(record)
            binding = _binding(record)
            if ((previous is not None and observed.ref != previous)
                    or observed.ref.backend != _backend(record) or observed.ref.cluster != binding.cluster):
                raise ValueError("Observed scheduler identity changed")
            record["job_id"] = observed.ref.job_id
        if observed.state != "unknown" and observed.ref is None:
            raise ValueError("Verified scheduler state requires a job identity")
        record.update(state="finished" if observed.state == "terminal" else observed.state,
                      scheduler_state=observed.scheduler_state, observed_at=time.time(),
                      error_type=observed.reason, scheduler_evidence_sha256=observed.evidence_sha256)
        if _backend(record) == "pbs":
            record["pbs_state"] = observed.scheduler_state
        if observed.state in {"queued", "running", "terminal"}:
            record["release_confirmed"] = True
        if observed.state == "terminal":
            exit_code = observed.exit_code
            if _backend(record) == "pbs" and exit_code is not None and re.fullmatch(r"-?[0-9]+", str(exit_code)):
                exit_code = int(exit_code)
            record["exit_code"] = exit_code
            if record["cancel_requested"]:
                record["state"] = "cancelled"
        _save(record)

    def submit(self, goal, profile="gpu", resume=False, new_attempt=False):
        require_local_execution()
        goal = self._goal(goal)
        folder = self._folder(goal)
        with file_lock(folder / ".controller.lock"):
            previous = self._current(folder)
            if previous and previous["state"] != "preparation_failed":
                previous = self._poll(previous, release=True)
                if previous["state"] not in TERMINAL:
                    if new_attempt or resume:
                        raise ValueError("A live or uncertain PBS submission already owns this goal")
                    return _public(previous)
                if not new_attempt:
                    return _public(previous)
            resource = self.settings["profiles"].get(profile)
            if resource is None:
                raise ValueError("Unknown administrator resource profile")
            with _goal_lease(goal):
                marker = goal / ASSIGNMENT
                if marker.exists():
                    raise ValueError("An unresolved PBS assignment already owns this goal")
                store = GoalStore(goal)
                state = store.read()
                if resume:
                    if state["status"] not in {"paused", "interrupted", "waiting_for_input"}:
                        raise ValueError("Goal is not resumable")
                    if state.get("inflight") is not None:
                        def reconcile(s):
                            s["cycles"].append({**s["inflight"], "status": "interrupted", "summary": "Previous execution ended; reservation retained"})
                            s["inflight"] = None
                        store.mutate(reconcile)
                    state = store.resume()
                if state["status"] != "active" or state.get("inflight") is not None:
                    raise ValueError("Submit an active goal with no unresolved execution reservation")
                nonce = uuid.uuid4().hex
                attempt = folder / nonce
                attempt.mkdir(mode=0o700)
                record = {"schema": SCHEMA, "uid": os.getuid(), "owner": pwd.getpwuid(os.getuid()).pw_name,
                          "nonce": nonce, "goal": str(goal), "definition_sha256": state["definition_sha256"],
                          "record_path": str(attempt / "submission.json"), "control_root": str(self.root),
                          "settings": self.settings, "resources": resource, "profile": profile,
                          "execution": {key: self.settings[key] for key in ("python", "runtime_root", "credential_toml", "env_file", "environment", "extra_python_paths")},
                          "state": "preparing", "job_id": None, "cancel_requested": False, "created_at": time.time()}
                _save(record)
                atomic_json(folder / "current.json", {"attempt_id": nonce})
                atomic_json(marker, {"schema": SCHEMA, "nonce": nonce, "record": record["record_path"], "definition_sha256": state["definition_sha256"]})
                try:
                    from .pbs_goal_snapshot import freeze_source
                    record["source"] = freeze_source(Path(self.settings["repository"]), attempt / "source")
                    from .pbs_goal_policy import freeze_settings
                    record["tool_settings"] = freeze_settings(self.settings, attempt)
                    entry = Path(record["source"]["root"]) / "scripts/rsi/pbs_goal_worker.py"
                    if not entry.is_file():
                        raise ValueError("Frozen source has no cluster goal worker")
                    launch = WorkerLaunch(Path(self.settings["python"]), entry, Path(record["record_path"]),
                                          Path(record["goal"]) / "workspace", attempt / "scheduler.log")
                    request = _resources(resource)
                except Exception as exc:
                    record.update(state="preparation_failed", error_type=type(exc).__name__)
                    _save(record)
                    marker.unlink()
                    raise
            spec = self.settings.get("scheduler", {"backend": "pbs", "cluster": "pbs"})
            after = int(time.time()) - 5
            record["scheduler"] = {"backend": spec["backend"], "cluster": spec["cluster"],
                                   "submitted_after": after, "submitted_before": after + 300}
            record["state"] = "submitting"
            _save(record)
            try:
                observed = self._adapter(record).submit_held(_binding(record), request, launch)
                self._observe(record, observed)
            except PermissionError:
                raise
            except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
                record["error_type"] = type(exc).__name__
                _save(record)
            return _public(self._poll(record, release=True))

    def status(self, goal):
        require_local_execution()
        goal = self._goal(goal)
        folder = self._folder(goal)
        with file_lock(folder / ".controller.lock"):
            record = self._current(folder)
            if record is None:
                return {"state": "not_submitted", "job_id": None, "goal_status": GoalStore(goal).read()["status"]}
            return _public(self._poll(record, release=True))

    def cancel(self, goal):
        require_local_execution()
        goal = self._goal(goal)
        folder = self._folder(goal)
        with file_lock(folder / ".controller.lock"):
            record = self._current(folder)
            if record is None:
                raise ValueError("No PBS submission exists for this goal")
            if record["state"] not in TERMINAL:
                record["cancel_requested"] = True
                record["cancel_requested_at"] = time.time()
                _save(record)
                GoalStore(goal).pause()
            return _public(self._poll(record))


def worker_observation(record):
    """Re-read private identity and verify the real scheduler before allocation use."""
    require_local_execution()
    current = _record(record["record_path"])
    if any(current.get(key) != record.get(key) for key in ("nonce", "job_id", "definition_sha256")):
        raise ValueError("Worker submission binding changed")
    return Controller(current["settings"])._observe_worker(current)


def worker_claim(record_path):
    require_local_execution()
    record = _record(record_path)
    controller = Controller(record["settings"])
    goal = controller._goal(record["goal"])
    with file_lock(Path(record_path).parent.parent / ".controller.lock"):
        record = _record(record_path)
        current = controller._current(controller._folder(goal))
        if not current or current["nonce"] != record["nonce"] or record["cancel_requested"]:
            raise ValueError("PBS submission is no longer runnable")
        controller._observe_worker(record)
        state = GoalStore(goal).read()
        if state["definition_sha256"] != record["definition_sha256"] or state["status"] != "active" or state.get("inflight") is not None:
            raise ValueError("Goal definition or execution state changed while queued")
        if record.get("worker", {}).get("state") in {"claimed", "running", "finished", "failed"}:
            raise ValueError("This PBS attempt already claimed its worker")
        from .pbs_goal_snapshot import verify_source
        verify_source(record["source"])
        record["worker"] = {"state": "claimed", "started_at": time.time()}
        _save(record)
        return record


def worker_ready(record_path, allocation_receipt):
    """Persist verified kernel allocation evidence before any provider request."""
    from .cluster_allocation import backend, worker_job_matches

    with file_lock(Path(record_path).parent.parent / ".controller.lock"):
        record = _record(record_path)
        if (not worker_job_matches(record, os.environ) or record["cancel_requested"]
                or record.get("worker", {}).get("state") != "claimed"
                or not isinstance(allocation_receipt, dict)
                or allocation_receipt.get("job_id") != record["job_id"]
                or allocation_receipt.get("backend") != backend(record)):
            raise ValueError("Verified allocation does not match the claimed worker")
        record["worker"].update(state="running", allocation=allocation_receipt, ready_at=time.time())
        _save(record)


def worker_finish(record_path, goal_status, error_type=None):
    from .cluster_allocation import worker_job_matches

    with file_lock(Path(record_path).parent.parent / ".controller.lock"):
        record = _record(record_path)
        if not worker_job_matches(record, os.environ):
            raise ValueError("Worker completion does not match the cluster job")
        record["worker"] = {**record.get("worker", {}), "state": "failed" if error_type else "finished",
                            "finished_at": time.time(), "goal_status": goal_status, "error_type": error_type}
        _save(record)


def assert_execution_allowed(goal):
    from .cluster_allocation import capture_allocation, record_environment_matches, worker_job_matches

    marker = Path(goal).resolve() / ASSIGNMENT
    if not marker.exists() and not marker.is_symlink():
        return
    require_local_execution()
    assignment = _private(marker)
    record = _record(assignment["record"])
    if (record["goal"] != str(Path(goal).resolve()) or record["nonce"] != assignment.get("nonce")
            or record["definition_sha256"] != assignment.get("definition_sha256")
            or record["cancel_requested"] or record.get("state") in TERMINAL
            or not record_environment_matches(record)
            or not worker_job_matches(record, os.environ)
            or GoalStore(goal).read()["definition_sha256"] != record["definition_sha256"]):
        raise RuntimeError("This goal is assigned to a cluster; use its status or cancellation command")
    allocation = capture_allocation(record)
    if allocation.job_id != record["job_id"]:
        raise RuntimeError("Cluster allocation differs from the assigned goal")
