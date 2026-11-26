"""Synthetic Slurm command fixtures, not a claim of live cluster compatibility."""
import os
import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from backend.compute.contracts import JobRef, ResourceRequest, SubmissionBinding, WorkerLaunch
from backend.compute.slurm import SlurmAdapter

NONCE = "a" * 32
AFTER = 1794528000  # 2026-11-13T00:00:00 UTC
BINDING = SubmissionBinding("testcluster", NONCE, os.getuid(), AFTER, AFTER + 60)
RESOURCES = ResourceRequest("gpu", 2, 1, 16384, 3300)
REF = JobRef("slurm", "testcluster", "12345")


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


def active(**changes):
    fields = {"JobId": REF.job_id, "JobName": BINDING.job_name,
              "UserId": f"fixture({os.getuid()})", "JobState": "PENDING",
              "Reason": "JobHeldUser", "Priority": "0", "Comment": NONCE,
              "SubmitTime": "2026-11-13T00:00:30", "Requeue": "0", "Restarts": "0",
              "NumNodes": "1", "NumTasks": "1", "NumCPUs": "2", "MinMemoryNode": "16G",
              "TimeLimit": "00:55:00", "Partition": "gpu", "TRES": "cpu=2,mem=16G,node=1,gres/gpu=1",
              "ExitCode": "0:0"}
    fields.update(changes)
    return " ".join(f"{key}={value}" for key, value in fields.items()) + "\n"


def accounting(**changes):
    fields = {"JobIDRaw": REF.job_id, "Cluster": "testcluster", "UID": str(os.getuid()),
              "JobName": BINDING.job_name, "State": "COMPLETED", "Submit": "2026-11-13T00:00:30",
              "Partition": "gpu", "ReqCPUS": "2", "ReqNodes": "1", "ReqMem": "16Gn",
              "TimelimitRaw": "55", "ReqTRES": "cpu=2,mem=16G,node=1,gres/gpu=1",
              "ExitCode": "0:0", "Restarts": "0"}
    fields.update(changes)
    return "|".join(fields.values()) + "\n"


@pytest.fixture(autouse=True)
def local(monkeypatch):
    monkeypatch.setenv("WEBUI_V2_MODE", "local")
    monkeypatch.setenv("VENUS_SCIENTIST_ONLINE", "0")


@pytest.fixture
def launch(tmp_path):
    return WorkerLaunch(tmp_path / "venv/bin/python", tmp_path / "frozen worker.py",
                        tmp_path / "submission.json", tmp_path, tmp_path / "scheduler.log")


def adapter(runner):
    return SlurmAdapter("testcluster", Path("/opt/slurm/bin"), runner)


def test_held_submission_fixed_argv_and_clean_environment(launch, monkeypatch):
    monkeypatch.setenv("SOME_API_KEY", "fixture-never-export")
    monkeypatch.setenv("SBATCH_WRAP", "malicious inherited option")
    runner = Runner((0, "12345;testcluster\n"), (0, active()))
    result = adapter(runner).submit_held(BINDING, RESOURCES, launch)
    assert result.state == "held" and result.ref == REF and result.resources == RESOURCES
    argv, kwargs = runner.calls[0]
    assert argv[0] == "/opt/slurm/bin/sbatch"
    assert {"--hold", "--parsable", "--no-requeue", "--export=NIL", "--nodes=1", "--ntasks=1",
            "--mem=16384M", "--time=55", "--gres=gpu:1", "--job-name=" + BINDING.job_name,
            "--comment=" + NONCE} <= set(argv)
    assert argv[-5:] == [str(launch.python), "-B", str(launch.entrypoint), "--record", str(launch.record)]
    assert Path(argv[-6]).name == "slurm_worker.sh"
    assert not any(value.startswith(("--wrap", "--get-user-env", "--export=ALL")) for value in argv)
    assert "SOME_API_KEY" not in kwargs["env"] and "SBATCH_WRAP" not in kwargs["env"]
    assert kwargs["timeout"] == 20 and kwargs["env"]["TZ"] == "UTC"
    assert len(runner.calls) == 2


@pytest.mark.parametrize("response", [(1, ""), (0, "12345;other\n"), (0, "12345\n12346\n"),
                                      (0, "12345_7"), (0, "Submitted batch job 12345")])
def test_uncertain_submission_does_not_retry_or_release(launch, response):
    runner = Runner(response)
    result = adapter(runner).submit_held(BINDING, RESOURCES, launch)
    assert result.state == "unknown" and result.ref is None
    assert len(runner.calls) == 1


def test_submission_timeout_is_unknown_without_retry(launch):
    runner = Runner(subprocess.TimeoutExpired("fixture", 20))
    assert adapter(runner).submit_held(BINDING, RESOURCES, launch).state == "unknown"
    assert len(runner.calls) == 1


def test_held_resource_mutation_retains_job_identity_without_release(launch):
    runner = Runner((0, "12345"), (0, active(NumCPUs="4")))
    result = adapter(runner).submit_held(BINDING, RESOURCES, launch)
    assert result.state == "unknown" and result.ref == REF
    assert len(runner.calls) == 2


@pytest.mark.parametrize("changed", [
    {"UserId": f"other({os.getuid() + 1})"}, {"JobName": "vf_" + NONCE[:12]},
    {"SubmitTime": "2026-11-12T23:59:59"}, {"SubmitTime": "2026-11-13T00:01:01"},
    {"Comment": "b" * 32}, {"Cluster": "other"}, {"Requeue": "1"}, {"Restarts": "1"},
    {"NumNodes": "2"}, {"NumTasks": "2"}, {"ArrayJobId": "12000"},
    {"TimeLimit": "UNLIMITED"}, {"JobId": "12345_1"}, {"MinMemoryNode": "8Gc"},
])
def test_active_evidence_mismatch_is_unknown(changed):
    runner = Runner((0, active(**changed)))
    result = adapter(runner).observe(BINDING, REF)
    assert result.state == "unknown" and len(runner.calls) == 1


def test_duplicate_active_fields_rejected():
    runner = Runner((0, active().strip() + " JobId=12345\n"))
    assert adapter(runner).observe(BINDING, REF).state == "unknown"


def test_missing_submit_receipt_can_be_reconciled_without_resubmission():
    runner = Runner((0, "12345\n"), (0, active()))
    result = adapter(runner).observe(BINDING)
    assert result.state == "held" and result.ref == REF
    assert [Path(call[0][0]).name for call in runner.calls] == ["squeue", "scontrol"]
    assert "--user=" + str(os.getuid()) in runner.calls[0][0]
    assert "--name=" + BINDING.job_name in runner.calls[0][0]


def test_multiple_nonce_candidates_are_not_arbitrarily_selected():
    runner = Runner((0, "12345\n12346\n"))
    assert adapter(runner).observe(BINDING).state == "unknown"
    assert len(runner.calls) == 1


def test_accounting_terminal_binding_and_exit_code_are_preserved():
    runner = Runner((1, ""), (0, accounting(State="FAILED", ExitCode="1:0")))
    result = adapter(runner).observe(BINDING, REF)
    assert result.state == "terminal" and result.scheduler_state == "FAILED" and result.exit_code == "1:0"
    assert result.resources == RESOURCES
    command = runner.calls[1][0]
    assert "--allocations" in command and "--duplicates" in command and "--jobs=12345" in command
    assert "--starttime=2026-11-13T00:00:00" in command
    assert not any("Comment" in part for part in command)  # Accounting need not retain comments.


@pytest.mark.parametrize("text", ["", accounting() + accounting(), accounting(Cluster="other"),
                                  accounting(JobName="vf_" + NONCE[:12]), accounting(UID=str(os.getuid() + 1)),
                                  accounting(Submit="Unknown"), accounting(JobIDRaw="12345.batch"),
                                  accounting(Restarts="1"), accounting(ExitCode=""), accounting(State="RUNNING")])
def test_missing_or_conflicting_accounting_never_establishes_terminal(text):
    runner = Runner((1, ""), (0, text))
    assert adapter(runner).observe(BINDING, REF).state == "unknown"
    assert len(runner.calls) == 2


def test_release_revalidates_current_identity_and_resources():
    runner = Runner((0, active()), (0, ""), (0, active(JobState="RUNNING", Reason="None", Priority="500")))
    result = adapter(runner).release(BINDING, REF, RESOURCES)
    assert result.state == "running"
    assert runner.calls[1][0] == ["/opt/slurm/bin/scontrol", "--clusters=testcluster", "release", "12345"]


@pytest.mark.parametrize("changed", [{"NumCPUs": "4"}, {"Reason": "JobHeldAdmin"},
                                     {"UserId": f"other({os.getuid() + 1})"}])
def test_release_does_not_mutate_an_unverified_allocation(changed):
    runner = Runner((0, active(**changed)))
    with pytest.raises(ValueError):
        adapter(runner).release(BINDING, REF, RESOURCES)
    assert len(runner.calls) == 1


def test_cancellation_verifies_identity_and_observes_result():
    runner = Runner((0, active()), (0, ""), (1, ""), (0, accounting(State=f"CANCELLED by {os.getuid()}")))
    result = adapter(runner).cancel(BINDING, REF)
    assert result.state == "terminal" and result.scheduler_state == "CANCELLED"
    assert runner.calls[1][0] == ["/opt/slurm/bin/scancel", "--clusters=testcluster", "12345"]


def test_cancel_refuses_unknown_identity_and_does_not_call_scancel():
    runner = Runner((0, active(Comment="changed")))
    with pytest.raises(ValueError):
        adapter(runner).cancel(BINDING, REF)
    assert len(runner.calls) == 1


def test_terminal_cancel_is_idempotent_without_scancel():
    runner = Runner((1, ""), (0, accounting()))
    assert adapter(runner).cancel(BINDING, REF).state == "terminal"
    assert len(runner.calls) == 2


def test_worker_env_alone_does_not_authorize_execution():
    runner = Runner((0, active()))
    with pytest.raises(ValueError, match="running"):
        adapter(runner).worker_identity(BINDING, REF, {"SLURM_JOB_ID": "12345", "SLURM_CLUSTER_NAME": "testcluster"})
    assert len(runner.calls) == 1


def test_worker_identity_requires_matching_env_and_running_scheduler():
    runner = Runner((0, active(JobState="RUNNING", Reason="None", Priority="500")))
    result = adapter(runner).worker_identity(BINDING, REF, {"SLURM_JOB_ID": "12345", "SLURM_CLUSTER_NAME": "testcluster"})
    assert result.ref == REF and result.state == "running"
    assert not hasattr(result, "devices")  # A scheduler observation does not grant GPU mounts.


@pytest.mark.parametrize("operation", ["submit", "observe", "release", "cancel", "worker", "validate"])
def test_online_blocks_every_entrypoint_before_runner(monkeypatch, launch, operation):
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


@pytest.mark.parametrize("mode", ["online", "unknown", ""])
def test_constructor_rejects_nonlocal_modes(monkeypatch, mode):
    monkeypatch.setenv("WEBUI_V2_MODE", mode)
    with pytest.raises(PermissionError):
        adapter(Runner())


def test_fractional_minute_is_not_silently_increased(launch):
    runner = Runner()
    with pytest.raises(ValueError, match="whole minutes"):
        adapter(runner).submit_held(BINDING, replace(RESOURCES, wall_seconds=61), launch)
    assert not runner.calls


def test_foreign_backend_cluster_and_owner_fail_before_commands():
    runner = Runner()
    scheduler = adapter(runner)
    for binding, ref in [(replace(BINDING, uid=os.getuid() + 1), REF),
                         (BINDING, JobRef("slurm", "other", "12345")),
                         (BINDING, JobRef("pbs", "testcluster", "12345.fs"))]:
        with pytest.raises(ValueError):
            scheduler.observe(binding, ref)
    assert not runner.calls


def test_fixed_worker_script_integrity_is_checked(monkeypatch, tmp_path, launch):
    from backend.compute import slurm
    fake = tmp_path / "slurm.py"
    fake.touch()
    (tmp_path / "slurm_worker.sh").write_text("#!/bin/sh\nmalicious script\n")
    monkeypatch.setattr(slurm, "__file__", str(fake))
    runner = Runner()
    with pytest.raises(ValueError, match="launcher changed"):
        adapter(runner).submit_held(BINDING, RESOURCES, launch)
    assert not runner.calls


@pytest.mark.parametrize("path", [Path("relative.py"), Path("/tmp/bad\npath"), Path("/tmp/out-%j"), Path("/tmp/../other")])
def test_worker_paths_are_not_shell_or_filename_pattern_inputs(path):
    with pytest.raises(ValueError):
        WorkerLaunch(path, Path("/frozen/worker.py"), Path("/private/record.json"), Path("/workspace"), Path("/private/log"))


def test_oversized_scheduler_output_returns_unknown():
    runner = Runner((0, "x" * (1024 * 1024 + 1)))
    assert adapter(runner).observe(BINDING, REF).state == "unknown"
    assert len(runner.calls) == 1


def test_mode_change_between_observation_and_release_rechecks_admission(monkeypatch):
    class ModeChangingRunner(Runner):
        def __call__(self, argv, **kwargs):
            result = super().__call__(argv, **kwargs)
            monkeypatch.setenv("WEBUI_V2_MODE", "online")
            return result

    runner = ModeChangingRunner((0, active()))
    with pytest.raises(PermissionError):
        adapter(runner).release(BINDING, REF, RESOURCES)
    assert len(runner.calls) == 1


def test_cpu_profile_submits_without_gpu_request(launch):
    cpu = ResourceRequest("cpu", 2, 0, 16384, 3300)
    runner = Runner((0, "12345"), (0, active(Partition="cpu", TRES="cpu=2,mem=16G,node=1")))
    result = adapter(runner).submit_held(BINDING, cpu, launch)
    assert result.state == "held" and result.resources == cpu
    assert not any(arg.startswith("--gres=") for arg in runner.calls[0][0])
