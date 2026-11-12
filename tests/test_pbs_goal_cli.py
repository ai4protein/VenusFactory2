"""CLI dispatch and the local/PBS execution exclusion boundary."""
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def cli(monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "pbs_goal_cli_under_test", ROOT / "scripts/rsi/run_autonomous_research.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    pbs = ModuleType("agent.harness.pbs_goal")
    calls = []

    class Controller:
        @classmethod
        def from_file(cls, path):
            calls.append(("config", path))
            return cls()

        def submit(self, goal, **kwargs):
            calls.append(("submit", goal, kwargs))
            return {"status": "submitted", "job_id": "123.server"}

        def status(self, goal):
            calls.append(("status", goal))
            return {"status": "queued"}

        def cancel(self, goal):
            calls.append(("cancel", goal))
            return {"status": "cancel_requested"}

    pbs.Controller = Controller
    pbs.assert_execution_allowed = lambda goal: calls.append(("guard", goal))
    monkeypatch.setitem(sys.modules, pbs.__name__, pbs)
    return module, pbs, calls


@pytest.mark.parametrize("action,status", [
    ("submit-pbs", "submitted"), ("status-pbs", "queued"),
    ("cancel-pbs", "cancel_requested"),
])
def test_pbs_actions_dispatch_without_goal_or_credential_loading(cli, monkeypatch, capsys, action, status):
    module, _, calls = cli

    def forbidden(*args, **kwargs):
        pytest.fail("PBS dispatch must not load or mutate the goal or its credential")

    monkeypatch.setattr(module, "GoalStore", forbidden)
    monkeypatch.setattr(module, "resolve_credential", forbidden)
    module.main([action, "--output", "goal", "--pbs-config", "pbs.json"])
    assert calls[0] == ("config", "pbs.json")
    if action == "submit-pbs":
        assert calls[1] == ("submit", "goal", {
            "profile": "gpu", "resume": False, "new_attempt": False,
        })
    else:
        assert calls[1] == (action.removesuffix("-pbs"), "goal")
    assert json.loads(capsys.readouterr().out)["status"] == status


def test_submit_explicit_resume_new_attempt_and_resource_profile(cli, capsys):
    module, _, calls = cli
    module.main([
        "submit-pbs", "--output", "goal", "--pbs-config", "pbs.json",
        "--resource-profile", "gpu-small", "--resume-goal", "--new-attempt",
    ])
    assert calls[1] == ("submit", "goal", {
        "profile": "gpu-small", "resume": True, "new_attempt": True,
    })
    assert json.loads(capsys.readouterr().out)["job_id"] == "123.server"


@pytest.mark.parametrize("action", ["submit-pbs", "status-pbs", "cancel-pbs"])
def test_pbs_config_is_required_before_controller_access(cli, action):
    module, _, calls = cli
    with pytest.raises(SystemExit) as error:
        module.main([action, "--output", "goal"])
    assert error.value.code == 2
    assert calls == []


@pytest.mark.parametrize("flag", ["--new-attempt", "--resume-goal"])
@pytest.mark.parametrize("action", ["run", "resume", "status-pbs", "cancel-pbs"])
def test_submission_flags_cannot_silently_affect_other_actions(cli, flag, action):
    module, _, calls = cli
    with pytest.raises(SystemExit) as error:
        module.main([action, "--output", "goal", "--pbs-config", "pbs.json", flag])
    assert error.value.code == 2
    assert calls == []


@pytest.mark.parametrize("action", ["run", "resume"])
def test_active_pbs_guard_precedes_goal_mutation_and_credential_read(cli, monkeypatch, action):
    module, pbs, calls = cli

    def blocked(goal):
        calls.append(("guard", goal))
        raise RuntimeError("PBS execution owns this goal")

    def forbidden(*args, **kwargs):
        pytest.fail("Blocked local execution reached mutable state or credentials")

    pbs.assert_execution_allowed = blocked
    monkeypatch.setattr(module, "GoalStore", forbidden)
    monkeypatch.setattr(module, "resolve_credential", forbidden)
    with pytest.raises(RuntimeError, match="PBS execution"):
        module.main([action, "--output", "goal"])
    assert calls == [("guard", "goal")]


@pytest.mark.parametrize("action", ["run", "resume"])
def test_local_execution_guard_and_explicit_resume_order(cli, monkeypatch, capsys, action):
    module, _, calls = cli
    profile = SimpleNamespace(model="model")

    class Store:
        def __init__(self, goal):
            calls.append(("store", goal))

        def read(self):
            calls.append(("read",))
            return {"definition": {"profile": {}}}

        def resume(self):
            calls.append(("resume",))

    class Runtime:
        def __init__(self, store, credential, runtimes):
            calls.append(("runtime", credential, runtimes))

        async def run(self):
            calls.append(("run",))
            return {"status": "completed", "cycles": [1]}

    def credential(*args):
        calls.append(("credential",))
        return "fixture-credential"

    monkeypatch.setattr(module, "GoalStore", Store)
    monkeypatch.setattr(module.Profile, "from_dict", lambda value: profile)
    monkeypatch.setattr(module, "resolve_credential", credential)
    monkeypatch.setattr(module, "default_runtimes", lambda root: {"python": "configured"})
    monkeypatch.setattr(module, "AutonomousResearch", Runtime)
    module.main([action, "--output", "goal"])
    names = [item[0] for item in calls]
    expected = ["guard", "store", "read"] + (["resume"] if action == "resume" else [])
    assert names == expected + ["credential", "runtime", "run"]
    assert json.loads(capsys.readouterr().out) == {"status": "completed", "cycles": 1}


def test_real_configuration_and_unsubmitted_status_use_only_temporary_goal(tmp_path, monkeypatch, capsys):
    """Exercise the real parser/controller without any scheduler or credential call."""
    from agent.harness import Profile
    from agent.harness.autonomous_research import create
    from agent.harness.pbs_goal import Controller

    monkeypatch.setenv("WEBUI_V2_MODE", "local")
    monkeypatch.setenv("VENUS_SCIENTIST_ONLINE", "0")
    root = tmp_path / "goals"
    root.mkdir()
    goal = root / "protein-study"
    profile = Profile.from_dict({
        "mode": "science_expert", "harness": "expert-deepseek-harness",
        "model": {"provider": "dmx", "model": "glm-4-flash",
                  "base_url": "https://www.dmxapi.cn/v1"},
        "budget": {"model_calls": 12, "tool_calls": 12,
                   "wall_seconds": 120, "output_tokens": 4096, "thinking": "disabled"},
    })
    store = create(goal, profile, "Fixture protein model goal",
                   {"max_cycles": 2, "model_calls": 24, "tool_calls": 24, "wall_seconds": 120})
    before = store.read()
    settings = json.loads((ROOT / "configs/pbs-goals.example.json").read_text())
    settings.update(repository="repository", control_root="control",
                    python=sys.executable, runtime_root="runtime", pbs_bin="pbs-bin",
                    goal_roots=["goals"], env_file="not-read.env")
    config = tmp_path / "pbs.json"
    config.write_text(json.dumps(settings))

    def forbidden(*args, **kwargs):
        pytest.fail("Unsubmitted status must not invoke PBS or read credentials")

    import subprocess
    monkeypatch.setattr(subprocess, "run", forbidden)
    controller = Controller.from_file(config)
    assert controller.settings["goal_roots"] == [str(root)]
    assert controller.settings["env_file"] == str(tmp_path / "not-read.env")
    assert controller.settings["control_root"] == str(tmp_path / "control")
    spec = importlib.util.spec_from_file_location(
        "pbs_goal_cli_real_test", ROOT / "scripts/rsi/run_autonomous_research.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "resolve_credential", forbidden)
    module.main(["status-pbs", "--output", str(goal), "--pbs-config", str(config)])
    assert json.loads(capsys.readouterr().out) == {
        "state": "not_submitted", "job_id": None, "goal_status": "active",
    }
    assert store.read() == before
    assert not (goal / ".pbs-assignment.json").exists()


def test_real_configuration_rejects_secret_environment_names(tmp_path):
    from agent.harness.pbs_goal import Controller

    settings = json.loads((ROOT / "configs/pbs-goals.example.json").read_text())
    settings.update(repository="repository", control_root="control",
                    python=sys.executable, runtime_root="runtime", pbs_bin="pbs-bin",
                    goal_roots=["goals"])
    settings["environment"]["OPENROUTER_API_KEY"] = "synthetic-unusable-test-value"
    config = tmp_path / "pbs.json"
    config.write_text(json.dumps(settings))
    with pytest.raises(ValueError, match="never credentials"):
        Controller.from_file(config)
    assert not (tmp_path / "control").exists()
