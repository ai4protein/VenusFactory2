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

from .pbs_goal_config import configuration, validate_observed_resources
from .pbs_goal_transport import bounded_run

SCHEMA = "venusfactory.pbs-goal.v1"
ASSIGNMENT = ".pbs-assignment.json"
NONCE_ENV = "VENUS_PBS_GOAL_ATTEMPT"
TERMINAL = {"finished", "cancelled", "preparation_failed"}
JOB_ID = re.compile(r"[0-9]+\.[A-Za-z0-9_.-]+")


def _online():
    return os.getenv("WEBUI_V2_MODE") == "online" or os.getenv("VENUS_SCIENTIST_ONLINE") == "1"


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
            "goal_status": GoalStore(record["goal"]).read()["status"], "pbs_state": record.get("pbs_state"),
            "exit_code": record.get("exit_code"), "cancel_requested": record["cancel_requested"],
            "worker": record.get("worker"), "error_type": record.get("error_type")}


class Controller:
    def __init__(self, settings, runner=None):
        self.settings = configuration(settings)
        self.runner = runner or bounded_run
        self.root = Path(self.settings["control_root"])
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        info = self.root.lstat()
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise ValueError("PBS control root must be a private directory owned by the current user")

    @classmethod
    def from_file(cls, path):
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

    def _invoke(self, command, arguments, record):
        user = pwd.getpwuid(os.getuid())
        env = {"HOME": user.pw_dir, "USER": user.pw_name, "LOGNAME": user.pw_name,
               "PATH": record["settings"]["pbs_bin"] + ":/usr/bin:/bin", "LANG": "C.UTF-8"}
        argv = [str(Path(record["settings"]["pbs_bin"]) / command), *arguments]
        return self.runner(argv, timeout=30, env=env, capture_output=True, text=True)

    def _query(self, record):
        args = ["-x", "-f", "-F", "json"]
        if record.get("job_id"):
            args.append(record["job_id"])
        response = self._invoke("qstat", args, record)
        if response.returncode:
            raise RuntimeError("PBS status is unavailable")
        raw = response.stdout
        if len(raw) > 64_000_000:
            raise ValueError("PBS status response exceeds the limit")
        rows = json.loads(raw).get("Jobs")
        if not isinstance(rows, dict):
            raise ValueError("Invalid PBS status response")
        matched = []
        for identity, row in rows.items():
            if not isinstance(row, dict):
                continue
            variables = row.get("Variable_List")
            match = (isinstance(variables, dict) and variables.get(NONCE_ENV) == record["nonce"]
                     and row.get("Job_Name") == "vf_" + record["nonce"]
                     and str(row.get("Job_Owner", "")).split("@", 1)[0] == record["owner"])
            if record.get("job_id") == identity and not match:
                raise ValueError("PBS owner or submission identity differs from its saved receipt")
            if match:
                if not JOB_ID.fullmatch(identity):
                    raise ValueError("Unsupported PBS job identity")
                matched.append((identity, row))
        if len(matched) > 1:
            raise ValueError("Multiple PBS jobs match one submission; operator reconciliation is required")
        return matched[0] if matched else None

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
        if record["state"] == "preparation_failed":
            return record
        if record["state"] in {"finished", "cancelled"}:
            self._release_assignment(record)
            _save(record)
            return record
        try:
            found = self._query(record)
        except (OSError, RuntimeError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
            record.update(state="unknown", error_type=type(exc).__name__)
            _save(record)
            return record
        if found is None:
            record.update(state="unknown", error_type="SubmissionNotObserved")
            _save(record)
            return record
        identity, row = found
        record.update(job_id=identity, pbs_state=row.get("job_state"), observed_at=time.time(), error_type=None)
        phase = row.get("job_state")
        mapping = {"H": "held", "Q": "queued", "W": "queued", "T": "queued", "R": "running", "E": "exiting", "F": "finished"}
        record["state"] = mapping.get(phase, "unknown")
        if phase in {"Q", "W", "T", "R", "E", "F"}:
            record["release_confirmed"] = True
        if phase == "F":
            record["exit_code"] = row.get("Exit_status")
            if record["cancel_requested"]:
                record["state"] = "cancelled"
            self._release_assignment(record)
            _save(record)
            return record
        _save(record)
        if record["cancel_requested"]:
            if time.time() - record.get("cancel_sent_at", 0) >= 30:
                record["cancel_sent_at"] = time.time()
                _save(record)
                try:
                    response = self._invoke("qdel", [identity], record)
                    record["cancel_command_exit"] = response.returncode
                except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
                    record["error_type"] = type(exc).__name__
                _save(record)
        elif (release and not _online() and not record.get("release_confirmed")
              and phase == "H" and "u" in str(row.get("Hold_Types", ""))):
            validate_observed_resources(row, record["resources"])
            if time.time() - record.get("release_sent_at", 0) >= 30:
                record["release_sent_at"] = time.time()
                _save(record)
                try:
                    response = self._invoke("qrls", ["-h", "u", identity], record)
                    record["release_command_exit"] = response.returncode
                except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
                    record["error_type"] = type(exc).__name__
                _save(record)
        return record

    def submit(self, goal, profile="gpu", resume=False, new_attempt=False):
        if _online():
            raise ValueError("Authored PBS GPU research is available only for local execution")
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
                        raise ValueError("Frozen source has no PBS goal worker")
                except Exception as exc:
                    record.update(state="preparation_failed", error_type=type(exc).__name__)
                    _save(record)
                    marker.unlink()
                    raise
            record["state"] = "submitting"
            _save(record)
            wall = resource["wall_seconds"]
            args = ["-h", "-r", "n", "-m", "n", "-N", "vf_" + nonce, "-q", resource["queue"],
                    "-l", f"select=1:ncpus={resource['ncpus']}:ngpus=1:mem={resource['memory_gb']}gb",
                    "-l", f"walltime={wall // 3600:02}:{wall % 3600 // 60:02}:{wall % 60:02}",
                    "-j", "oe", "-o", str(attempt / "pbs.log"), "-v", NONCE_ENV + "=" + nonce,
                    "--", self.settings["python"], str(entry), "--record", record["record_path"]]
            try:
                response = self._invoke("qsub", args, record)
                identity = response.stdout.strip()
                if response.returncode == 0 and JOB_ID.fullmatch(identity):
                    record["job_id"] = identity
                else:
                    record["error_type"] = "AmbiguousSubmissionResponse"
            except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
                record["error_type"] = type(exc).__name__
            _save(record)
            return _public(self._poll(record, release=True))

    def status(self, goal):
        goal = self._goal(goal)
        folder = self._folder(goal)
        with file_lock(folder / ".controller.lock"):
            record = self._current(folder)
            if record is None:
                return {"state": "not_submitted", "job_id": None, "goal_status": GoalStore(goal).read()["status"]}
            return _public(self._poll(record, release=True))

    def cancel(self, goal):
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


def worker_claim(record_path):
    record = _record(record_path)
    controller = Controller(record["settings"])
    goal = controller._goal(record["goal"])
    with file_lock(Path(record_path).parent.parent / ".controller.lock"):
        record = _record(record_path)
        current = controller._current(controller._folder(goal))
        if not current or current["nonce"] != record["nonce"] or record["cancel_requested"]:
            raise ValueError("PBS submission is no longer runnable")
        found = controller._query(record)
        if not found or found[0] != os.getenv("PBS_JOBID") or found[0] != record.get("job_id") or found[1].get("job_state") != "R":
            raise ValueError("Worker does not own the running PBS submission")
        validate_observed_resources(found[1], record["resources"])
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
    with file_lock(Path(record_path).parent.parent / ".controller.lock"):
        record = _record(record_path)
        if (os.getenv("PBS_JOBID") != record.get("job_id") or record["cancel_requested"]
                or record.get("worker", {}).get("state") != "claimed"
                or not isinstance(allocation_receipt, dict)
                or allocation_receipt.get("job_id") != record["job_id"]
                or allocation_receipt.get("backend") != "pbs"):
            raise ValueError("Verified allocation does not match the claimed worker")
        record["worker"].update(state="running", allocation=allocation_receipt, ready_at=time.time())
        _save(record)


def worker_finish(record_path, goal_status, error_type=None):
    with file_lock(Path(record_path).parent.parent / ".controller.lock"):
        record = _record(record_path)
        if os.getenv("PBS_JOBID") != record.get("job_id"):
            raise ValueError("Worker completion does not match the PBS job")
        record["worker"] = {**record.get("worker", {}), "state": "failed" if error_type else "finished",
                            "finished_at": time.time(), "goal_status": goal_status, "error_type": error_type}
        _save(record)


def assert_execution_allowed(goal):
    marker = Path(goal).resolve() / ASSIGNMENT
    if not marker.exists() and not marker.is_symlink():
        return
    assignment = _private(marker)
    record = _record(assignment["record"])
    if (record["goal"] != str(Path(goal).resolve()) or record["nonce"] != assignment.get("nonce")
            or record["definition_sha256"] != assignment.get("definition_sha256")
            or record["cancel_requested"] or record.get("state") in TERMINAL
            or os.getenv("VENUS_SCIENTIST_PBS_RECORD") != record["record_path"]
            or os.getenv("PBS_JOBID") != record.get("job_id")
            or GoalStore(goal).read()["definition_sha256"] != record["definition_sha256"]):
        raise RuntimeError("This goal is assigned to PBS; use its status or cancellation command")
    from .pbs_gpu import capture
    from .sandbox_resources import Limits
    allocation = capture(Limits.load())
    if allocation.job_id != record["job_id"]:
        raise RuntimeError("PBS GPU allocation differs from the assigned goal")
