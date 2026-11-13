"""Offline local-only cluster gates; no live scheduler, model or GPU requests."""

from __future__ import annotations

import importlib.util
import json
import os
import pwd
import subprocess
import sys
from pathlib import Path

import pytest

from agent.harness import pbs_goal, pbs_goal_config, pbs_goal_worker
from backend.compute.access import require_local_execution

ROOT = Path(__file__).resolve().parents[1]
BLOCKED = [
    {"WEBUI_V2_MODE": "online"},
    {"WEBUI_V2_MODE": " OnLiNe "},
    {"WEBUI_V2_MODE": "local", "VENUS_SCIENTIST_ONLINE": " 1 "},
    {"WEBUI_V2_MODE": " LOCAL ", "VENUS_SCIENTIST_ONLINE": " TRUE "},
    {"WEBUI_V2_MODE": "staging"},
    {"WEBUI_V2_MODE": ""},
    {"VENUS_SCIENTIST_ONLINE": "unknown"},
]


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch):
    environment = dict(os.environ)
    environment.pop("WEBUI_V2_MODE", None)
    environment.pop("VENUS_SCIENTIST_ONLINE", None)
    monkeypatch.setattr(os, "environ", environment)


def forbidden(*args, **kwargs):
    pytest.fail("Denied cluster entry reached configuration, storage, or scheduler")


@pytest.mark.parametrize("environment", BLOCKED)
def test_online_or_unknown_modes_fail_closed(environment):
    with pytest.raises(PermissionError, match="only in local"):
        require_local_execution(environment)


@pytest.mark.parametrize(
    "environment",
    [
        {},
        {"WEBUI_V2_MODE": "local"},
        {"WEBUI_V2_MODE": " LoCaL "},
        {"VENUS_SCIENTIST_ONLINE": "0"},
        {"VENUS_SCIENTIST_ONLINE": " FaLsE "},
        {"VENUS_SCIENTIST_ONLINE": " off "},
        {"VENUS_SCIENTIST_ONLINE": ""},
    ],
)
def test_standalone_default_and_explicit_local_remain_available(environment):
    assert require_local_execution(environment) is None


@pytest.mark.parametrize("environment", BLOCKED)
@pytest.mark.parametrize("entry", ["constructor", "from_file", "configuration"])
def test_admission_precedes_config_reads_or_directory_creation(monkeypatch, environment, entry):
    os.environ.update(environment)
    monkeypatch.setattr(Path, "read_text", forbidden)
    monkeypatch.setattr(Path, "mkdir", forbidden)
    if entry != "configuration":
        monkeypatch.setattr(pbs_goal, "configuration", forbidden)
    with pytest.raises(PermissionError, match="only in local"):
        if entry == "constructor":
            pbs_goal.Controller({})
        elif entry == "from_file":
            pbs_goal.Controller.from_file("missing-config.json")
        else:
            pbs_goal_config.configuration({})


@pytest.mark.parametrize("method", ["submit", "status", "cancel", "_invoke", "_poll"])
@pytest.mark.parametrize("environment", BLOCKED)
def test_existing_controller_cannot_bypass_after_mode_change(monkeypatch, method, environment):
    controller = object.__new__(pbs_goal.Controller)
    controller._goal = forbidden
    controller.runner = forbidden
    controller._release_assignment = forbidden
    monkeypatch.setattr(pbs_goal, "_save", forbidden)
    os.environ.update(environment)
    with pytest.raises(PermissionError, match="only in local"):
        if method == "_invoke":
            controller._invoke("qstat", [], {})
        elif method == "_poll":
            controller._poll({"state": "finished"}, release=True)
        else:
            getattr(controller, method)("missing-goal")


@pytest.mark.parametrize("entry", ["claim", "run_record"])
@pytest.mark.parametrize("environment", BLOCKED)
def test_worker_admission_precedes_record_credentials_and_finish(monkeypatch, environment, entry):
    os.environ.update(environment)
    monkeypatch.setattr(pbs_goal, "_record", forbidden)
    monkeypatch.setattr(pbs_goal, "worker_finish", forbidden)
    monkeypatch.setattr(pbs_goal_worker, "_selected_credential", forbidden)
    with pytest.raises(PermissionError, match="only in local"):
        if entry == "claim":
            pbs_goal.worker_claim("missing-record.json")
        else:
            pbs_goal_worker.run_record("missing-record.json")


def load_script(name):
    spec = importlib.util.spec_from_file_location("cluster_admission_" + name, ROOT / "scripts/rsi" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("action", ["submit-pbs", "status-pbs", "cancel-pbs"])
def test_cli_rejects_before_controller_configuration(monkeypatch, action):
    cli = load_script("run_autonomous_research")
    os.environ["WEBUI_V2_MODE"] = " ONLINE "
    monkeypatch.setattr(pbs_goal.Controller, "from_file", forbidden)
    monkeypatch.setattr(cli, "GoalStore", forbidden)
    monkeypatch.setattr(cli, "resolve_credential", forbidden)
    with pytest.raises(PermissionError, match="only in local"):
        cli.main([action, "--output", "missing-goal", "--pbs-config", "missing-config.json"])


def test_worker_script_checks_before_record_path_access(monkeypatch):
    entry = load_script("pbs_goal_worker")
    monkeypatch.setattr(sys, "argv", ["pbs_goal_worker.py", "--record", "missing-record.json"])
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(os, "umask", lambda _: 0o077)
    monkeypatch.setattr(Path, "is_symlink", forbidden)
    monkeypatch.setattr(Path, "read_text", forbidden)
    os.environ["VENUS_SCIENTIST_ONLINE"] = " 1 "
    with pytest.raises(PermissionError, match="only in local"):
        entry.main()


@pytest.mark.parametrize("environment", BLOCKED[:3])
def test_fresh_isolated_worker_rejects_online_without_loading_record(tmp_path, environment):
    env = {"PATH": os.defpath, "LANG": "C.UTF-8", **environment}
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            str(ROOT / "scripts/rsi/pbs_goal_worker.py"),
            "--record",
            str(tmp_path / "missing.json"),
        ],
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 1
    assert json.loads(result.stdout) == {"status": "interrupted", "error_type": "PermissionError"}
    assert not list(tmp_path.iterdir())


def test_already_claimed_local_worker_can_finish_despite_mode_change(tmp_path, monkeypatch):
    path = tmp_path / "attempt" / "submission.json"
    path.parent.mkdir()
    record = {
        "schema": pbs_goal.SCHEMA,
        "uid": os.getuid(),
        "owner": pwd.getpwuid(os.getuid()).pw_name,
        "nonce": "a" * 32,
        "record_path": str(path),
        "job_id": "123.fixture",
        "worker": {"state": "running", "started_at": 1},
    }
    pbs_goal._save(record)
    path.chmod(0o600)
    os.environ.update(WEBUI_V2_MODE="ONLINE", VENUS_SCIENTIST_ONLINE="1", PBS_JOBID="123.fixture")
    monkeypatch.setattr(pbs_goal, "Controller", forbidden)
    pbs_goal.worker_finish(path, "interrupted", error_type="SyntheticFailure")
    saved = json.loads(path.read_text())
    assert saved["worker"]["state"] == "failed"
    assert saved["worker"]["goal_status"] == "interrupted"
    assert saved["worker"]["error_type"] == "SyntheticFailure"
    assert saved["worker"]["started_at"] == 1
    assert saved["nonce"] == record["nonce"]
