"""Real virtualenv execution retains runtime imports without exposing host roots."""
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import venv

import pytest

from agent.harness import session_sandbox as sandbox


@pytest.mark.parametrize("runner", ["session", "kimi"])
@pytest.mark.asyncio
async def test_actual_virtualenv_mounts_stdlib_and_active_packages_only(tmp_path, monkeypatch, runner):
    monkeypatch.setenv("VENUS_SESSION_SANDBOX", "bwrap")
    runtime = tmp_path / "environment"
    venv.EnvBuilder(with_pip=False, symlinks=True).create(runtime)
    version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    packages = runtime / "lib" / version / "site-packages"
    packages.mkdir(parents=True, exist_ok=True)
    (packages / "venus_runtime_probe.py").write_text("VALUE = 73\n")
    (runtime / ".env").write_text("PRIVATE_RUNTIME_CANARY=not_for_guest\n")
    work = tmp_path / "workspace"
    work.mkdir()
    (work / "probe.py").write_text(
        "import encodings,json,statistics,venus_runtime_probe\n"
        "from pathlib import Path\n"
        "assert venus_runtime_probe.VALUE == 73\n"
        "assert statistics.mean([2,4]) == 3\n"
        "assert str(Path(venus_runtime_probe.__file__)).startswith('/opt/venus-runtime/')\n"
        "assert not Path('/opt/venus-runtime/.env').exists()\n"
        "assert not Path('/opt/venus-runtime/pyvenv.cfg').exists()\n"
        "assert not Path('/opt/venus-runtime/conda-meta').exists()\n"
        f"assert not Path({str(runtime)!r}).exists()\n"
        f"assert not Path({str(Path(sys.base_prefix))!r}).exists()\n"
        "print(json.dumps({'value':venus_runtime_probe.VALUE,'stdlib':encodings.__file__}))\n"
    )
    from agent.harness.python_runtime_mounts import python_runtime_mounts
    assert str(packages) in python_runtime_mounts(runtime)
    monkeypatch.setattr(sandbox, "sys", SimpleNamespace(
        prefix=str(runtime), base_prefix=sys.base_prefix,
        executable=str(runtime / "bin/python"), version_info=sys.version_info,
    ))
    if runner == "session":
        result = await sandbox.execute(work, tmp_path / "control", "python", "probe.py", 20)
    else:
        import subprocess
        from agent.rsi.engine.sandbox import bwrap_command
        private, capabilities = tmp_path / "private", tmp_path / "capabilities"
        private.mkdir(); capabilities.mkdir()
        command = bwrap_command(work, private, capabilities, Path("/usr/bin/true"), runtime, 0)
        command = command[:command.index("--") + 1] + ["/opt/venus-runtime/bin/python", "-I", "probe.py"]
        completed = subprocess.run(command, capture_output=True, text=True, timeout=20)
        result = {"exit_code": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr}
    assert result["exit_code"] == 0, result
    assert json.loads(result["stdout"])["value"] == 73


def test_mount_arguments_keep_base_runtime_and_overlay_only_virtualenv_packages(tmp_path, monkeypatch):
    base = tmp_path / "base"
    active = tmp_path / "active"
    version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    (base / "lib" / version).mkdir(parents=True)
    packages = active / "lib" / version / "site-packages"
    packages.mkdir(parents=True)
    monkeypatch.setattr(sandbox, "sys", SimpleNamespace(
        prefix=str(active), base_prefix=str(base),
        executable=sys.executable, version_info=sys.version_info,
    ))
    args = sandbox.command(tmp_path, "python", "analysis.py", 10,
                           scratch=tmp_path / "scratch", memory=1024**3)
    mounts = [(args[i + 1], args[i + 2]) for i, item in enumerate(args) if item == "--ro-bind"]
    base_mount = (str(base / "lib"), "/opt/venus-runtime/lib")
    package_mount = (str(packages), f"/opt/venus-runtime/lib/{version}/site-packages")
    assert base_mount in mounts and package_mount in mounts
    assert mounts.index(base_mount) < mounts.index(package_mount)
    assert not any(source in {str(base), str(active), str(active / "lib")} for source, _ in mounts)
    assert "--unshare-all" in args and "--clearenv" in args
    assert "-I" in args


def test_actual_missing_base_site_packages_does_not_modify_host(tmp_path, monkeypatch):
    import subprocess
    base, active = tmp_path / "base", tmp_path / "active"
    version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    stdlib = base / "lib" / version
    stdlib.mkdir(parents=True)
    (stdlib / "stdlib-marker").write_text("stdlib")
    packages = active / "lib" / version / "site-packages"
    packages.mkdir(parents=True)
    (packages / "package-marker").write_text("package")
    work, scratch = tmp_path / "work", tmp_path / "scratch"
    work.mkdir(); scratch.mkdir()
    monkeypatch.setattr(sandbox, "sys", SimpleNamespace(
        prefix=str(active), base_prefix=str(base),
        executable=sys.executable, version_info=sys.version_info,
    ))
    args = sandbox.command(work, "python", "unused.py", 10, scratch=scratch, memory=1024**3)
    destination = "/opt/venus-runtime/lib/" + version
    command = (
        f"test $(cat {destination}/stdlib-marker) = stdlib && "
        f"test $(cat {destination}/site-packages/package-marker) = package && "
        f"! touch {destination}/changed"
    )
    args = args[:args.index("--") + 1] + ["/usr/bin/sh", "-c", command]
    result = subprocess.run(args, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert "Read-only file system" in result.stderr
    assert not (stdlib / "site-packages").exists()
    assert not (stdlib / "changed").exists()
