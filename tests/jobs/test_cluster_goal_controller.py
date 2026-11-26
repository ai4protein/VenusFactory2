"""Real durable ledger plus synthetic scheduler; no jobs or provider calls."""

from __future__ import annotations

import copy
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pytest
from test_pbs_goal_scheduler import FakePBS
from test_pbs_goal_scheduler import prepared as prepared

from agent.harness import pbs_goal
from agent.harness.pbs_goal import Controller
from test_pbs_goal_cli import cli as cli


class FakeSlurm(FakePBS):
    def __call__(self, argv, *, env, timeout, **kwargs):
        assert timeout > 0 and not any("KEY" in name for name in env)
        command = Path(argv[0]).name

        def option(prefix):
            return next(arg[len(prefix) :] for arg in argv if arg.startswith(prefix))

        with self.transaction() as state:
            state["calls"].append(argv)
            if command == "sbatch":
                assert "--hold" in argv and "--no-requeue" in argv and "--export=NIL" in argv
                assert not any(arg.startswith(("--gres=", "--wrap=")) for arg in argv)
                assert argv[-3].endswith("/scripts/rsi/pbs_goal_worker.py")
                state["counter"] += 1
                identity = str(state["counter"])
                row = {
                    "JobId": identity,
                    "JobName": option("--job-name="),
                    "Cluster": "fixture",
                    "UserId": f"fixture({os.getuid()})",
                    "JobState": "PENDING",
                    "Priority": "0",
                    "Reason": "JobHeldUser",
                    "Comment": option("--comment="),
                    "SubmitTime": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S"),
                    "Requeue": "0",
                    "Restarts": "0",
                    "NumNodes": "1",
                    "NumTasks": "1",
                    "NumCPUs": option("--cpus-per-task="),
                    "MinMemoryNode": option("--mem="),
                    "TimeLimit": f"00:{int(option('--time=')):02}:00",
                    "Partition": option("--partition="),
                    "ReqTRES": "cpu=2,mem=16G,node=1",
                    "ExitCode": "0:0",
                }
                state["jobs"][identity] = row
                if state.pop("lose_sbatch", False):
                    state["query_error"] = True
                    self.path.write_text(json.dumps(state))
                    raise subprocess.TimeoutExpired(argv, timeout)
                return subprocess.CompletedProcess(argv, 0, identity + ";fixture\n", "")
            if state.get("query_error"):
                return subprocess.CompletedProcess(argv, 1, "", "unavailable")
            if command == "squeue":
                rows = [
                    key
                    for key, row in state["jobs"].items()
                    if row["JobName"] == option("--name=") and row["JobState"] not in {"COMPLETED", "CANCELLED"}
                ]
                return subprocess.CompletedProcess(argv, 0, "\n".join(rows), "")
            if command == "scontrol":
                row = state["jobs"].get(argv[-1])
                if not row:
                    return subprocess.CompletedProcess(argv, 1, "", "missing")
                if "release" in argv:
                    row.update(Priority="1", Reason="Resources")
                    return subprocess.CompletedProcess(argv, 0, "", "")
                if row["JobState"] in {"COMPLETED", "CANCELLED"}:
                    return subprocess.CompletedProcess(argv, 1, "", "terminal")
                return subprocess.CompletedProcess(argv, 0, " ".join(f"{key}={val}" for key, val in row.items()), "")
            if command == "scancel":
                state["jobs"][argv[-1]].update(JobState="CANCELLED")
                return subprocess.CompletedProcess(argv, 0, "", "")
            if command == "sacct":
                rows = []
                for row in state["jobs"].values():
                    if row["JobName"] != option("--name="):
                        continue
                    rows.append(
                        "|".join(
                            [
                                row["JobId"],
                                "fixture",
                                str(os.getuid()),
                                row["JobName"],
                                row["JobState"],
                                row["SubmitTime"],
                                row["Partition"],
                                row["NumCPUs"],
                                "1",
                                row["MinMemoryNode"] + "n",
                                "30",
                                row["ReqTRES"],
                                row["ExitCode"],
                                "0",
                            ]
                        )
                    )
                return subprocess.CompletedProcess(argv, 0, "\n".join(rows), "")
            raise AssertionError(command)


@pytest.fixture
def slurm(prepared, tmp_path):
    settings, store, _ = prepared
    settings = copy.deepcopy(settings)
    settings.pop("pbs_bin")
    settings.update(
        schema="venusfactory.cluster-config.v1",
        scheduler={"backend": "slurm", "cluster": "fixture", "bin_directory": str(tmp_path / "slurm-bin")},
    )
    settings["profiles"] = {"cpu": {"queue": "cpu", "ncpus": 2, "ngpus": 0, "memory_gb": 16, "wall_seconds": 1800}}
    return settings, store, FakeSlurm(tmp_path / "fake-slurm")


def record_path(settings):
    records = list(Path(settings["control_root"]).glob("*/*/submission.json"))
    assert len(records) == 1
    return records[0]


def test_slurm_held_submit_reconcile_and_terminal_keep_one_durable_attempt(slurm):
    settings, store, server = slurm
    before = store.read()
    result = Controller(settings, runner=server).submit(store.root, profile="cpu")
    assert result["state"] == "queued"
    assert result["backend"] == "slurm" and result["cluster"] == "fixture"
    assert result["scheduler_state"] == "PENDING" and result["pbs_state"] is None
    assert Controller(settings, runner=server).submit(store.root, profile="cpu")["attempt_id"] == result["attempt_id"]
    assert server.count("sbatch") == 1
    assert server.count("scontrol") >= 3
    record = json.loads(record_path(settings).read_text())
    assert record["scheduler"]["backend"] == "slurm"
    assert record["scheduler"]["submitted_before"] - record["scheduler"]["submitted_after"] == 300
    assert record["resources"]["ngpus"] == 0
    assert store.read()["usage"] == before["usage"]
    server.change(result["job_id"], JobState="COMPLETED", ExitCode="0:0")
    terminal = Controller(settings, runner=server).status(store.root)
    assert terminal["state"] == "finished" and terminal["exit_code"] == "0:0"
    assert store.read()["status"] == "interrupted"
    assert not (store.root / pbs_goal.ASSIGNMENT).exists()
    store.resume()
    Controller(settings, runner=server).status(store.root)
    assert store.read()["status"] == "active"
    assert server.count("sbatch") == 1


def test_ambiguous_slurm_submission_never_resubmits(slurm):
    settings, store, server = slurm
    server.configure(lose_sbatch=True)
    result = Controller(settings, runner=server).submit(store.root, profile="cpu")
    assert result["state"] == "unknown"
    with pytest.raises(ValueError, match="uncertain"):
        Controller(settings, runner=server).submit(store.root, profile="cpu", new_attempt=True)
    assert server.count("sbatch") == 1
    server.configure(query_error=False)
    observed = Controller(settings, runner=server).status(store.root)
    assert observed["state"] == "queued" and server.count("sbatch") == 1


def test_slurm_cancel_uses_scheduler_identity_not_local_pid(slurm):
    settings, store, server = slurm
    result = Controller(settings, runner=server).submit(store.root, profile="cpu")
    cancelled = Controller(settings, runner=server).cancel(store.root)
    assert cancelled["state"] == "cancelled" and cancelled["job_id"] == result["job_id"]
    assert server.count("scancel") == 1 and store.read()["status"] == "paused"
    assert not (store.root / pbs_goal.ASSIGNMENT).exists()


def test_slurm_claim_checks_real_scheduler_and_is_once_only(slurm, monkeypatch):
    settings, store, server = slurm
    result = Controller(settings, runner=server).submit(store.root, profile="cpu")
    path = record_path(settings)
    monkeypatch.setattr(pbs_goal, "bounded_run", server)
    monkeypatch.setenv("SLURM_JOB_ID", result["job_id"])
    monkeypatch.setenv("SLURM_CLUSTER_NAME", "fixture")
    with pytest.raises(ValueError, match="running"):
        pbs_goal.worker_claim(path)
    server.change(result["job_id"], JobState="RUNNING")
    claimed = pbs_goal.worker_claim(path)
    assert claimed["worker"]["state"] == "claimed"
    with pytest.raises(ValueError, match="already claimed"):
        pbs_goal.worker_claim(path)
    assert pbs_goal.worker_observation(claimed).state == "running"
    assert store.read()["usage"]["model_calls"] == 0


@pytest.mark.parametrize("change", [{"ngpus": 1}, {"ngpus": -1}, {"wall_seconds": 61}])
def test_slurm_unsupported_resources_rejected_before_control_dir(slurm, change):
    settings, _, server = slurm
    settings["profiles"]["cpu"].update(change)
    with pytest.raises(ValueError):
        Controller(settings, runner=server)
    assert not Path(settings["control_root"]).exists()
    assert server.read()["calls"] == []


def test_legacy_pbs_without_timestamp_or_scheduler_binding_still_reconciles(prepared):
    settings, store, server = prepared
    result = Controller(settings, runner=server).submit(store.root)
    path = record_path(settings)
    record = json.loads(path.read_text())
    record.pop("scheduler")
    pbs_goal._save(record)
    with server.transaction() as state:
        state["jobs"][result["job_id"]].pop("ctime")
    observed = Controller(settings, runner=server).status(store.root)
    assert observed["state"] == "queued" and server.count("qsub") == 1
    assert Controller(settings, runner=server).cancel(store.root)["state"] == "cancelled"


def test_new_pbs_submission_cannot_drop_timestamp_binding(prepared):
    settings, store, server = prepared
    result = Controller(settings, runner=server).submit(store.root)
    with server.transaction() as state:
        state["jobs"][result["job_id"]].pop("ctime")
    assert Controller(settings, runner=server).status(store.root)["state"] == "unknown"
    with pytest.raises(ValueError):
        Controller(settings, runner=server).submit(store.root, new_attempt=True)
    assert server.count("qsub") == 1


@pytest.mark.parametrize("action", ["submit-cluster", "status-cluster", "cancel-cluster"])
def test_generic_cluster_cli_dispatches_without_local_credential_reads(cli, monkeypatch, action):
    module, _, calls = cli

    def forbidden(*args, **kwargs):
        pytest.fail("Cluster dispatch must not read local model credentials")

    monkeypatch.setattr(module, "resolve_credential", forbidden)
    module.main([action, "--output", "goal", "--cluster-config", "cluster.json", "--resource-profile", "cpu"])
    assert calls[0] == ("config", "cluster.json")
    assert calls[1][0] == action.split("-", 1)[0]
    if action == "submit-cluster":
        assert calls[1][2]["profile"] == "cpu"


def test_existing_cluster_marker_is_rejected_online_before_read(tmp_path, monkeypatch):
    goal = tmp_path / "goal"
    goal.mkdir()
    (goal / pbs_goal.ASSIGNMENT).write_text("not valid JSON; must never read")
    monkeypatch.setenv("WEBUI_V2_MODE", " ONLINE ")
    monkeypatch.setattr(pbs_goal, "_private", lambda *_: pytest.fail("online read cluster assignment"))
    with pytest.raises(PermissionError, match="only in local"):
        pbs_goal.assert_execution_allowed(goal)
    pbs_goal.assert_execution_allowed(tmp_path / "ordinary-non-cluster-goal")


@pytest.mark.parametrize("backend", ["pbs", "slurm"])
def test_known_invalid_launch_clears_assignment_before_scheduler(prepared, slurm, tmp_path, backend):
    settings, store, server = prepared if backend == "pbs" else slurm
    settings["control_root"] = str(tmp_path / "invalid%launch")
    with pytest.raises(ValueError, match="paths"):
        Controller(settings, runner=server).submit(store.root, profile="gpu" if backend == "pbs" else "cpu")
    assert not (store.root / pbs_goal.ASSIGNMENT).exists()
    assert server.read()["calls"] == []
    record = json.loads(record_path(settings).read_text())
    assert record["state"] == "preparation_failed" and record["error_type"] == "ValueError"
    assert store.read()["status"] == "active"


def test_partition_option_like_name_is_rejected_before_ledger(slurm):
    settings, _, server = slurm
    settings["profiles"]["cpu"]["queue"] = "-x"
    with pytest.raises(ValueError, match="scheduler name"):
        Controller(settings, runner=server)
    assert not Path(settings["control_root"]).exists()
    assert server.read()["calls"] == []
