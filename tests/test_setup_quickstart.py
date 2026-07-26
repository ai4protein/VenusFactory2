"""Smoke tests for the one-click installer."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "setup_quickstart.py"
ROOT = SCRIPT.parent.parent


def _load_mod():
    spec = importlib.util.spec_from_file_location("setup_quickstart", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_help_exits_zero():
    mod = _load_mod()
    with pytest.raises(SystemExit) as exc:
        mod.main(["--help"])
    assert exc.value.code == 0


def test_dry_run_yes(capsys):
    mod = _load_mod()
    code = mod.main(
        [
            "--dry-run",
            "-y",
            "--force-install-deps",
            "--skip-frontend",
            "--skip-ckpts",
            "--lang",
            "en",
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "install.py" in out
    assert "Preflight" in out or "preflight" in out.lower() or "安装前" in out or "Scan" in out


def test_dry_run_clean_venv(capsys):
    mod = _load_mod()
    code = mod.main(
        ["--dry-run", "-y", "--clean-venv", "--skip-frontend", "--skip-ckpts", "--lang", "en"]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "Remove" in out or "删除" in out or ".venv" in out


def test_install_py_propagates_exit_code(monkeypatch):
    """Root install.py must not swallow child failures."""
    spec = importlib.util.spec_from_file_location("venus_root_install", ROOT / "install.py")
    assert spec and spec.loader
    root_install = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = root_install
    spec.loader.exec_module(root_install)

    class FakeProc:
        returncode = 7

    monkeypatch.setattr(root_install.subprocess, "run", lambda *a, **k: FakeProc())
    assert root_install.main(["--env", "venv", "--skip-frpc"]) == 7


def test_env_ready_requires_more_than_torch(monkeypatch):
    mod = _load_mod()

    def fake_imports(python, modules):
        if modules == ("torch",):
            return True, "ok"
        return False, "No module named torch_geometric"

    monkeypatch.setattr(mod, "_python_imports_ok", fake_imports)
    ok, detail = mod._env_ready("/usr/bin/python3")
    assert ok is False
    assert "torch_geometric" in detail


def test_reuse_ignores_system_torch(monkeypatch):
    mod = _load_mod()
    survey = mod.EnvSurvey.__new__(mod.EnvSurvey)
    survey.venv_ready = False
    survey.partial_venv = False
    survey.system_ready = True
    survey.system_torch = True
    plan = mod.InstallPlan()
    mod._apply_env_action(plan, "reuse", survey)
    assert plan.install_deps is True


def test_interactive_recommended_no_old_env(monkeypatch):
    mod = _load_mod()
    mod._LANG = "en"
    monkeypatch.delenv("HF_ENDPOINT", raising=False)

    class FakeSurvey:
        has_old_env = False
        venv_exists = False
        venv_ready = False
        venv_torch = False
        partial_venv = False
        system_ready = False
        system_torch = False
        frontend_dist = False
        frontend_node_modules = False
        ckpt_bytes = 0

        def lines(self):
            return [".venv/ not found"]

    monkeypatch.setattr(mod, "EnvSurvey", FakeSurvey)
    monkeypatch.setattr(mod, "_conda_active", lambda: None)
    # mode default, hf mirror no, continue yes
    answers = iter(["", "n", "y"])
    monkeypatch.setattr("builtins.input", lambda _p="": next(answers))
    plan = mod.build_interactive_plan(mod.InstallPlan(dry_run=True))
    assert plan.install_deps is True
    assert plan.preset == "predict-core"
    assert plan.hf_mirror is False


def test_should_not_prompt_with_flags():
    mod = _load_mod()
    args = mod.build_parser().parse_args(["--install-deps", "--skip-ckpts"])
    assert mod._should_prompt(args) is False
