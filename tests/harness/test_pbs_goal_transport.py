"""Real subprocess tests; no scheduler, model, credentials or external service."""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from agent.harness import pbs_goal_transport as transport

pytestmark = pytest.mark.skipif(os.name != "posix", reason="PBS transport uses POSIX process groups")


def _environment():
    return {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"}


def _run(code, *, timeout=3, text=True):
    return transport.bounded_run(
        [sys.executable, "-I", "-c", code], timeout=timeout,
        env=_environment(), text=text,
    )


def _running(pid):
    try:
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
    except (FileNotFoundError, ProcessLookupError):
        return False
    return state not in {"Z", "X"}


def test_captures_both_streams_and_preserves_nonzero_exit():
    result = _run("import os; os.write(1, b'out\\r\\n'); os.write(2, b'err\\n'); raise SystemExit(7)")
    assert result.returncode == 7
    assert result.stdout == "out\n"
    assert result.stderr == "err\n"


def test_binary_capture_and_explicit_environment(monkeypatch):
    monkeypatch.setenv("TRANSPORT_TEST_SECRET", "must-not-be-inherited")
    result = _run("import os; assert 'TRANSPORT_TEST_SECRET' not in os.environ; os.write(1, b'\\x00\\xff')", text=False)
    assert result.stdout == b"\x00\xff"
    assert result.stderr == b""


@pytest.mark.parametrize("stream,limit_name", [(1, "MAX_STDOUT_BYTES"), (2, "MAX_STDERR_BYTES")])
def test_large_output_is_bounded_and_error_does_not_expose_body(monkeypatch, stream, limit_name):
    monkeypatch.setattr(transport, limit_name, 4096)
    started = time.monotonic()
    with pytest.raises(RuntimeError, match="PBS command output exceeded") as error:
        _run(f"import os,time; os.write({stream}, b'sensitive-body' * 10000); time.sleep(30)")
    assert "sensitive-body" not in str(error.value)
    assert time.monotonic() - started < 3


def test_full_stdout_and_stderr_pipes_are_drained_concurrently(monkeypatch):
    monkeypatch.setattr(transport, "MAX_STDOUT_BYTES", 1024 * 1024)
    monkeypatch.setattr(transport, "MAX_STDERR_BYTES", 1024 * 1024)
    result = _run("""
import os, threading
def emit(fd, value):
    for _ in range(512):
        os.write(fd, value * 1024)
threads = [threading.Thread(target=emit, args=(1, b'o')),
           threading.Thread(target=emit, args=(2, b'e'))]
for thread in threads: thread.start()
for thread in threads: thread.join()
""")
    assert result.returncode == 0
    assert result.stdout == "o" * (512 * 1024)
    assert result.stderr == "e" * (512 * 1024)


@pytest.mark.parametrize("stream", [1, 2])
def test_slow_partial_output_does_not_reset_deadline_or_escape_in_exception(stream):
    started = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired) as error:
        _run(f"""
import os, time
while True:
    os.write({stream}, b'private-' + b'partial-body')
    time.sleep(.02)
""", timeout=.2)
    assert .15 <= time.monotonic() - started < 3
    assert error.value.output is None and error.value.stderr is None
    assert "private-partial-body" not in str(error.value)


def test_deadline_also_covers_process_with_closed_output_pipes():
    started = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired):
        _run("import os,time; os.close(1); os.close(2); time.sleep(30)", timeout=.2)
    assert time.monotonic() - started < 3


@pytest.mark.parametrize("held_pipe", [1, 2])
def test_exited_parent_with_descendant_pipe_is_cleaned_without_killing_foreign_process(tmp_path, held_pipe):
    child_file = tmp_path / "owned-child.pid"
    foreign = subprocess.Popen(
        [sys.executable, "-I", "-c", "import time; time.sleep(30)"],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        env=_environment(), start_new_session=True,
    )
    child = None
    try:
        code = f"""
import os, time
child = os.fork()
if child == 0:
    os.close({2 if held_pipe == 1 else 1})
    time.sleep(30)
    os._exit(0)
with open({str(child_file)!r}, 'w') as stream:
    stream.write(str(child))
os._exit(0)
"""
        with pytest.raises(subprocess.TimeoutExpired):
            _run(code, timeout=.4)
        child = int(child_file.read_text())
        deadline = time.monotonic() + 2
        while _running(child) and time.monotonic() < deadline:
            time.sleep(.01)
        assert not _running(child), "Owned descendant survived the command deadline"
        assert foreign.poll() is None, "Unrelated process was signalled"
    finally:
        foreign.kill()
        foreign.wait(timeout=3)
        if child is not None:
            try:
                os.waitpid(child, os.WNOHANG)
            except ChildProcessError:
                pass


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan"), True])
def test_invalid_timeout_rejected_before_spawn(timeout):
    with pytest.raises(ValueError, match="timeout"):
        _run("raise AssertionError('should not start')", timeout=timeout)


def test_uncaptured_execution_is_rejected():
    with pytest.raises(ValueError, match="captured output"):
        transport.bounded_run(
            [sys.executable, "-c", "print('must not run')"], timeout=1,
            env=_environment(), capture_output=False,
        )


def test_selector_setup_failure_reaps_its_started_process(monkeypatch):
    real_popen = subprocess.Popen
    children = []

    def start(*args, **kwargs):
        child = real_popen(*args, **kwargs)
        children.append(child)
        return child

    def unavailable():
        raise OSError("selector resource fixture")

    monkeypatch.setattr(transport.subprocess, "Popen", start)
    monkeypatch.setattr(transport.selectors, "DefaultSelector", unavailable)
    with pytest.raises(OSError, match="selector resource"):
        _run("import time; time.sleep(30)")
    assert len(children) == 1 and children[0].returncode is not None
