"""Cancellation and timeout must terminate an isolated command and its descendants."""

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/rsi"))
from command_lifecycle import run_in_group


def wait_for_file(path):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if path.exists() and path.stat().st_size:
            return
        time.sleep(0.02)
    raise AssertionError("Child did not start")


def active(pid):
    path = Path(f"/proc/{pid}/stat")
    return path.exists() and path.read_text().split()[2] != "Z"


def child_code(path):
    return "import pathlib,os,time;pathlib.Path(" + repr(str(path)) + ").write_text(str(os.getpid()));time.sleep(60)"


def test_timeout_cleans_up_its_process_group(tmp_path):
    pid = tmp_path / "pid"
    with (tmp_path / "log").open("w") as log:
        with pytest.raises(subprocess.TimeoutExpired):
            run_in_group(
                [sys.executable, "-c", child_code(pid)], cwd=tmp_path, env=os.environ.copy(), log=log, timeout=0.3
            )
    assert not active(int(pid.read_text()))


def test_sigterm_reaches_separately_started_child_group(tmp_path):
    pid = tmp_path / "pid"
    scripts = str(Path(__file__).resolve().parents[2] / "scripts/rsi")
    runner = tmp_path / "controller.py"
    runner.write_text(
        "import sys,os\nsys.path.insert(0,"
        + repr(scripts)
        + ")\nfrom command_lifecycle import install_termination_handlers,run_in_group\ninstall_termination_handlers()\nwith open('child.log','w') as log:\n run_in_group([sys.executable,'-c',"
        + repr(child_code(pid))
        + "],cwd='.',env=os.environ.copy(),log=log,timeout=60)\n"
    )
    with (tmp_path / "controller.log").open("w") as log:
        process = subprocess.Popen(
            [sys.executable, str(runner)], cwd=tmp_path, stdout=log, stderr=log, start_new_session=True
        )
        try:
            wait_for_file(pid)
            process.send_signal(signal.SIGTERM)
            assert process.wait(timeout=10) != 0
            assert not active(int(pid.read_text()))
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()


def test_success_preserves_return_code(tmp_path):
    with (tmp_path / "log").open("w") as log:
        assert (
            run_in_group([sys.executable, "-c", "raise SystemExit(7)"], cwd=tmp_path, env=os.environ.copy(), log=log)
            == 7
        )


def test_sigterm_kills_nested_groups_even_when_leaf_ignores_term(tmp_path):
    leaf_pid = tmp_path / "leaf.pid"
    helper = str(Path(__file__).resolve().parents[2] / "scripts/rsi")
    leaf = (
        "import signal,os,pathlib,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);pathlib.Path("
        + repr(str(leaf_pid))
        + ").write_text(str(os.getpid()));time.sleep(60)"
    )
    middle = tmp_path / "middle.py"
    middle.write_text(
        "import sys,os\nsys.path.insert(0,"
        + repr(helper)
        + ")\nfrom command_lifecycle import MPI_TERMINATION_GRACE,install_termination_handlers,run_in_group\ninstall_termination_handlers()\nwith open('leaf.log','w') as log:run_in_group([sys.executable,'-c',"
        + repr(leaf)
        + "],cwd='.',env=os.environ.copy(),log=log,timeout=60,termination_grace=MPI_TERMINATION_GRACE)\n"
    )
    outer = tmp_path / "outer.py"
    outer.write_text(
        "import sys,os\nsys.path.insert(0,"
        + repr(helper)
        + ")\nfrom command_lifecycle import install_termination_handlers,run_in_group\ninstall_termination_handlers()\nwith open('middle.log','w') as log:run_in_group([sys.executable,"
        + repr(str(middle))
        + "],cwd='.',env=os.environ.copy(),log=log,timeout=60)\n"
    )
    with (tmp_path / "outer.log").open("w") as log:
        process = subprocess.Popen(
            [sys.executable, str(outer)], cwd=tmp_path, stdout=log, stderr=log, start_new_session=True
        )
        try:
            wait_for_file(leaf_pid)
            start = time.monotonic()
            process.send_signal(signal.SIGTERM)
            process.wait(timeout=3)
            assert time.monotonic() - start < 3 and not active(int(leaf_pid.read_text()))
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            if leaf_pid.exists() and active(int(leaf_pid.read_text())):
                os.kill(int(leaf_pid.read_text()), signal.SIGKILL)
