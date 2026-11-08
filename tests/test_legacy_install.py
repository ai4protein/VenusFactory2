"""Legacy public install commands must enforce the same dependency policy."""
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from scripts.bootstrap import legacy_install as installer

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def installation(tmp_path, monkeypatch):
    root = tmp_path / "project with spaces"
    for name in ("configs/install.json", "requirements/bootstrap-constraints.txt",
                 "requirements/service-security.txt", "pyproject.toml"):
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    calls = []
    monkeypatch.setattr(installer, "ROOT", root)
    monkeypatch.setattr(installer, "resolve_profile", lambda value: ("cpu", "test hardware"))
    monkeypatch.setattr(installer.shutil, "which", lambda _: "/tools with spaces/uv")
    def execute(command, **kwargs):
        assert isinstance(command, list)
        assert kwargs == {"cwd": root, "check": True}
        calls.append(command)
    monkeypatch.setattr(installer.subprocess, "run", execute)
    return root, calls


@pytest.mark.parametrize("environment", ["venv", "system"])
def test_all_installations_are_constrained_and_checked(installation, environment):
    root, calls = installation
    assert installer.main(["--type", "cpu"], environment=environment) == 0
    installs = [call for call in calls if call[1:3] == ["pip", "install"]]
    assert len(installs) == 3
    for call in installs:
        assert call.count("--constraint") == 2
        assert str(root / "requirements/bootstrap-constraints.txt") in call
        assert str(root / "requirements/service-security.txt") in call
        assert ("--system" in call) == (environment == "system")
        assert "--prerelease=allow" not in call
    assert "torch==2.10.0" in installs[0] and "torchvision==0.25.0" in installs[0]
    assert "--only-binary" in installs[1]
    assert str(root / "pyproject.toml") in installs[2]
    assert calls[-2][1:3] == ["pip", "check"]
    assert calls[-1][1] == "-c"
    assert "torch_scatter" in calls[-1][2]


@pytest.mark.parametrize("filename", ["bootstrap-constraints.txt", "service-security.txt"])
def test_missing_policy_stops_before_package_manager(installation, filename):
    root, calls = installation
    (root / "requirements" / filename).unlink()
    assert installer.main(["--type", "cpu"]) == 1
    assert calls == []


def test_mismatched_pyg_abi_stops_before_install(installation):
    root, calls = installation
    path = root / "requirements/bootstrap-constraints.txt"
    path.write_text(path.read_text().replace("torch==2.10.0", "torch==2.9.0"))
    assert installer.main(["--type", "cpu"]) == 1
    assert calls == []


def test_install_failure_is_not_reported_as_success(installation, monkeypatch, capsys):
    root, calls = installation
    def fail(command, **kwargs):
        calls.append(command)
        if "torch==2.10.0" in command:
            raise subprocess.CalledProcessError(17, command)
    monkeypatch.setattr(installer.subprocess, "run", fail)
    assert installer.main(["--type", "cpu"]) == 17
    assert not any("torch_geometric" in call for call in calls)
    assert "installed and checked" not in capsys.readouterr().out


def test_failed_final_check_is_not_reported_as_success(installation, monkeypatch, capsys):
    _, calls = installation
    def fail(command, **kwargs):
        calls.append(command)
        if "check" in command:
            raise subprocess.CalledProcessError(9, command)
    monkeypatch.setattr(installer.subprocess, "run", fail)
    assert installer.main(["--type", "cpu"]) == 9
    assert calls[-1][1:3] == ["pip", "check"]
    assert "installed and checked" not in capsys.readouterr().out


def test_missing_uv_uses_the_installing_python(installation, monkeypatch):
    _, calls = installation
    monkeypatch.setattr(installer.shutil, "which", lambda _: None)
    assert installer.ensure_uv("https://pypi.org/simple") == [sys.executable, "-m", "uv"]
    assert calls == [[sys.executable, "-m", "pip", "install", "uv==0.11.6",
                      "--index-url", "https://pypi.org/simple"]]


def test_incomplete_existing_environment_is_preserved(installation):
    root, calls = installation
    folder = root / ".venv"
    folder.mkdir()
    (folder / "keep.txt").write_text("existing files")
    assert installer.main(["--type", "cpu"]) == 1
    assert calls == []
    assert (folder / "keep.txt").read_text() == "existing files"


def test_old_system_python_does_not_start_install(installation, monkeypatch):
    _, calls = installation
    monkeypatch.setattr(installer.sys, "version_info", (3, 10))
    assert installer.main(["--type", "cpu"], environment="system") == 1
    assert calls == []
