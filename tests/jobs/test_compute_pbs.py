"""Offline OpenPBS fixtures for the shared scheduling contract."""
import json
import os
import pwd
import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from backend.compute.contracts import JobRef, ResourceRequest, SubmissionBinding, WorkerLaunch
from backend.compute.factory import adapter_for
from backend.compute.pbs import PBSAdapter

NONCE = "b" * 32
BINDING = SubmissionBinding("pbs", NONCE, os.getuid(), 1794528000, 1794528060)
REF = JobRef("pbs", "pbs", "123.fs")
RESOURCES = ResourceRequest("gpu", 2, 1, 16384, 3300)


class Runner:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, argv, **kwargs):
        self.calls.append((argv, kwargs))
        assert self.responses, "Unexpected extra scheduler command"
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return SimpleNamespace(returncode=response[0], stdout=response[1], stderr="")


def row(**changes):
    value = {"Job_Name": BINDING.job_name, "Job_Owner": pwd.getpwuid(os.getuid()).pw_name + "@host",
             "Variable_List": {"VENUS_PBS_GOAL_ATTEMPT": NONCE}, "ctime": 1794528030,
             "job_state": "H", "Hold_Types": "u", "queue": "gpu", "Rerunable": False,
             "Resource_List": {"nodect": 1, "select": "1:ncpus=2:ngpus=1:mem=16gb", "ncpus": 2,
                               "ngpus": 1, "mem": "16gb", "walltime": "00:55:00"}}
    value.update(changes)
    return value


def status(value=None, **extra):
    return json.dumps({"Jobs": {REF.job_id: row() if value is None else value, **extra}})


@pytest.fixture(autouse=True)
def local(monkeypatch):
    monkeypatch.setenv("WEBUI_V2_MODE", "local")
    monkeypatch.setenv("VENUS_SCIENTIST_ONLINE", "0")


@pytest.fixture
def launch(tmp_path):
    return WorkerLaunch(tmp_path / "venv/bin/python", tmp_path / "source/worker.py",
                        tmp_path / "submission.json", tmp_path, tmp_path / "pbs.log")


def adapter(runner, **kwargs):
    return PBSAdapter("pbs", Path("/opt/pbs/bin"), runner, **kwargs)


def test_pbs_held_submit_uses_nonce_only_and_fixed_entrypoint(launch, monkeypatch):
    monkeypatch.setenv("SOME_API_KEY", "not-exported")
    runner = Runner((0, "123.fs\n"), (0, status()))
    observed = adapter(runner).submit_held(BINDING, RESOURCES, launch)
    assert observed.state == "held" and observed.resources == RESOURCES and observed.ref == REF
    command, options = runner.calls[0]
    assert command[:7] == ["/opt/pbs/bin/qsub", "-h", "-r", "n", "-m", "n", "-N"]
    assert "-V" not in command
    assert command[command.index("-v") + 1] == "VENUS_PBS_GOAL_ATTEMPT=" + NONCE
    assert command[-6:] == ["--", str(launch.python), "-B", str(launch.entrypoint), "--record", str(launch.record)]
    assert "SOME_API_KEY" not in options["env"]
    assert options["timeout"] == 30


@pytest.mark.parametrize("changes", [
    {"Job_Name": "vf_" + NONCE[:12]}, {"Job_Owner": "other@host"},
    {"Variable_List": {"VENUS_PBS_GOAL_ATTEMPT": "c" * 32}}, {"ctime": None},
    {"ctime": 1794527999}, {"ctime": 1794528061}, {"ctime": "1794528030"},
    {"Exit_status": "0"},
])
def test_mismatched_pbs_evidence_stays_unknown(changes):
    runner = Runner((0, status(row(**changes))))
    assert adapter(runner).observe(BINDING, REF).state == "unknown"
    assert len(runner.calls) == 1


def test_duplicate_nonce_and_array_job_do_not_get_selected():
    for payload in (status(**{"124.fs": row()}), json.dumps({"Jobs": {"123[1].fs": row()}})):
        runner = Runner((0, payload))
        assert adapter(runner).observe(BINDING).state == "unknown"


def test_legacy_missing_timestamp_retains_full_nonce_owner_validation():
    legacy = row()
    del legacy["ctime"]
    runner = Runner((0, status(legacy)))
    observed = adapter(runner, legacy_identity=True).observe(BINDING, REF)
    assert observed.state == "held" and observed.submitted_at is None
    legacy["Job_Owner"] = "other@host"
    runner = Runner((0, status(legacy)))
    assert adapter(runner, legacy_identity=True).observe(BINDING, REF).state == "unknown"


def test_legacy_mode_cannot_submit_new_jobs(launch):
    runner = Runner()
    with pytest.raises(ValueError, match="Legacy"):
        adapter(runner, legacy_identity=True).submit_held(BINDING, RESOURCES, launch)
    assert not runner.calls


def test_legacy_completed_record_without_resources_is_terminal_not_execution_proof():
    legacy = row(job_state="F", Exit_status=1)
    del legacy["Resource_List"]
    del legacy["ctime"]
    runner = Runner((0, status(legacy)))
    scheduler = adapter(runner, legacy_identity=True)
    observed = scheduler.observe(BINDING, REF)
    assert observed.state == "terminal" and observed.exit_code == "1" and observed.resources is None
    with pytest.raises(ValueError, match="resource"):
        scheduler.validate_resources(observed, RESOURCES)


@pytest.mark.parametrize("response", [(0, "123.fs\n124.fs"), (0, "123[1].fs"), (1, "")])
def test_uncertain_submission_never_retries(launch, response):
    runner = Runner(response)
    assert adapter(runner).submit_held(BINDING, RESOURCES, launch).state == "unknown"
    assert len(runner.calls) == 1


def test_timeout_submission_keeps_uncertain_state(launch):
    runner = Runner(subprocess.TimeoutExpired("fixture", 30))
    assert adapter(runner).submit_held(BINDING, RESOURCES, launch).state == "unknown"
    assert len(runner.calls) == 1


def test_release_rechecks_full_identity_and_resources():
    runner = Runner((0, status()), (0, ""), (0, status(row(job_state="Q", Hold_Types="n"))))
    assert adapter(runner).release(BINDING, REF, RESOURCES).state == "queued"
    assert runner.calls[1][0] == ["/opt/pbs/bin/qrls", "-h", "u", "123.fs"]


@pytest.mark.parametrize("changes", [{"Hold_Types": "s"}, {"queue": "different"}, {"Rerunable": True}])
def test_release_never_changes_unverified_or_administrator_only_hold(changes):
    runner = Runner((0, status(row(**changes))))
    with pytest.raises(ValueError):
        adapter(runner).release(BINDING, REF, RESOURCES)
    assert len(runner.calls) == 1


def test_cancellation_is_bound_and_terminal_cancel_is_idempotent():
    terminal = status(row(job_state="F", Exit_status=271))
    runner = Runner((0, status()), (0, ""), (0, terminal))
    assert adapter(runner).cancel(BINDING, REF).state == "terminal"
    assert runner.calls[1][0] == ["/opt/pbs/bin/qdel", "123.fs"]
    runner = Runner((0, terminal))
    assert adapter(runner).cancel(BINDING, REF).state == "terminal"
    assert len(runner.calls) == 1


def test_worker_needs_real_running_receipt_not_only_environment():
    runner = Runner((0, status()))
    with pytest.raises(ValueError, match="running"):
        adapter(runner).worker_identity(BINDING, REF, {"PBS_JOBID": REF.job_id})
    runner = Runner((0, status(row(job_state="R", Hold_Types="n"))))
    result = adapter(runner).worker_identity(BINDING, REF, {"PBS_JOBID": REF.job_id})
    assert result.resources == RESOURCES and not hasattr(result, "devices")


@pytest.mark.parametrize("operation", ["submit", "observe", "release", "cancel", "worker", "validate"])
def test_online_blocks_all_pbs_adapter_entrypoints(monkeypatch, launch, operation):
    runner = Runner()
    scheduler = adapter(runner)
    monkeypatch.setenv("WEBUI_V2_MODE", "online")
    actions = {"submit": lambda: scheduler.submit_held(BINDING, RESOURCES, launch),
               "observe": lambda: scheduler.observe(BINDING, REF),
               "release": lambda: scheduler.release(BINDING, REF, RESOURCES),
               "cancel": lambda: scheduler.cancel(BINDING, REF),
               "worker": lambda: scheduler.worker_identity(BINDING, REF, {}),
               "validate": lambda: scheduler.validate_resources(None, RESOURCES)}
    with pytest.raises(PermissionError):
        actions[operation]()
    assert not runner.calls


def test_foreign_owner_and_backend_are_rejected_before_command():
    runner = Runner()
    for binding, ref in [(replace(BINDING, uid=os.getuid() + 1), REF),
                         (BINDING, JobRef("slurm", "pbs", "123"))]:
        with pytest.raises(ValueError):
            adapter(runner).observe(binding, ref)
    assert not runner.calls


def test_adapter_factory_is_strict_and_local_only(monkeypatch):
    runner = Runner()
    assert isinstance(adapter_for("pbs", "pbs", Path("/opt/pbs/bin"), runner, legacy_pbs=True), PBSAdapter)
    from backend.compute.slurm import SlurmAdapter
    assert isinstance(adapter_for("slurm", "cluster", Path("/opt/slurm/bin"), runner), SlurmAdapter)
    for backend, legacy in [("slurm", True), ("unknown", False)]:
        with pytest.raises(ValueError):
            adapter_for(backend, "cluster", Path("/bin"), runner, legacy_pbs=legacy)
    monkeypatch.setenv("WEBUI_V2_MODE", "online")
    with pytest.raises(PermissionError):
        adapter_for("pbs", "pbs", Path("/bin"), runner)
    assert not runner.calls


def test_actual_openpbs_ctime_string_is_read_in_utc():
    # Real hl_gpu qstat fixture; TZ=UTC changed the same job from 21:43 to 13:43.
    from datetime import UTC, datetime
    stamp = int(datetime(2026, 11, 13, 13, 43, 40, tzinfo=UTC).timestamp())
    binding = replace(BINDING, submitted_after=stamp - 5, submitted_before=stamp + 295)
    runner = Runner((0, status(row(ctime="Fri Nov 13 13:43:40 2026"))))
    observed = adapter(runner).observe(binding, REF)
    assert observed.state == "held" and observed.submitted_at == stamp
    assert runner.calls[0][1]["env"]["TZ"] == "UTC"
    assert runner.calls[0][1]["env"]["LC_ALL"] == "C"


@pytest.mark.parametrize("stamp", ["Thu Nov 13 00:00:30 2026", "Fri Nov 13 00:00:30 2026 UTC",
                                  "Fri Nov 13 24:00:30 2026", "Fri Feb 31 00:00:30 2026",
                                  "Fri Nov 13 00:01:01 2026", "1794528030", True])
def test_timestamp_ambiguity_or_outside_window_never_matches(stamp):
    runner = Runner((0, status(row(ctime=stamp))))
    assert adapter(runner).observe(BINDING, REF).state == "unknown"


@pytest.mark.parametrize("changes", [{"Resource_List": {}}, {"Rerunable": True},
                                     {"Resource_List": {**row()["Resource_List"], "ngpus": 2}},
                                     {"Resource_List": {**row()["Resource_List"], "nodect": 2}}])
def test_owned_allocation_with_invalid_resources_can_cancel_but_not_run_or_release(changes):
    payload = status(row(**changes))
    runner = Runner((0, payload))
    scheduler = adapter(runner)
    observed = scheduler.observe(BINDING, REF)
    assert observed.ref == REF and observed.state == "held" and observed.resources is None
    with pytest.raises(ValueError, match="resource"):
        scheduler.validate_resources(observed, RESOURCES)
    runner = Runner((0, payload))
    with pytest.raises(ValueError):
        adapter(runner).release(BINDING, REF, RESOURCES)
    assert len(runner.calls) == 1
    runner = Runner((0, status(row(job_state="R", **changes))))
    with pytest.raises(ValueError):
        adapter(runner).worker_identity(BINDING, REF, {"PBS_JOBID": REF.job_id})
    runner = Runner((0, payload), (0, ""), (0, status(row(job_state="F", Exit_status=271))))
    assert adapter(runner).cancel(BINDING, REF).state == "terminal"
    assert runner.calls[1][0] == ["/opt/pbs/bin/qdel", "123.fs"]
