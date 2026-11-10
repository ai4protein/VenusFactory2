"""Atomic report publication under failures and independent process races."""

import json
import os
import subprocess
import sys

import pytest

from agent.rsi.engine.storage import atomic_json


def test_default_atomic_write_still_replaces_state(tmp_path):
    path = tmp_path / "state.json"
    atomic_json(path, {"generation": 1})
    atomic_json(path, {"generation": 2})
    assert json.loads(path.read_text()) == {"generation": 2}
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("kind", ["file", "directory", "symlink", "dangling_symlink"])
def test_exclusive_write_preserves_existing_destinations(tmp_path, kind):
    path = tmp_path / "report.json"
    target = tmp_path / "target.json"
    if kind == "file":
        path.write_text("original report")
    elif kind == "directory":
        path.mkdir()
    else:
        if kind == "symlink":
            target.write_text("original target")
        path.symlink_to(target)
    before = set(tmp_path.iterdir())
    with pytest.raises(FileExistsError):
        atomic_json(path, {"replacement": True}, exclusive=True)
    assert set(tmp_path.iterdir()) == before
    if kind == "file":
        assert path.read_text() == "original report"
    elif kind == "directory":
        assert path.is_dir() and not list(path.iterdir())
    else:
        assert path.is_symlink() and path.readlink() == target
        if kind == "symlink":
            assert target.read_text() == "original target"
        else:
            assert not target.exists()


@pytest.mark.parametrize("exclusive", [False, True])
def test_serialization_failure_never_publishes_a_partial_report(tmp_path, exclusive):
    path = tmp_path / "report.json"
    with pytest.raises(ValueError):
        atomic_json(path, {"valid_prefix": [1, 2, 3], "invalid": float("nan")}, exclusive=exclusive)
    assert not list(tmp_path.iterdir())


def test_publish_failure_removes_only_its_temporary_file(tmp_path, monkeypatch):
    path = tmp_path / "report.json"
    sentinel = tmp_path / "unrelated.json"
    sentinel.write_text("keep")

    def failed_link(*args):
        raise OSError("simulated filesystem error")

    monkeypatch.setattr(os, "link", failed_link)
    with pytest.raises(OSError, match="simulated filesystem"):
        atomic_json(path, {"result": [1, 2]}, exclusive=True)
    assert list(tmp_path.iterdir()) == [sentinel]
    assert sentinel.read_text() == "keep"


def test_independent_processes_publish_one_complete_report(tmp_path):
    path = tmp_path / "report.json"
    script = """
import sys
from pathlib import Path
from agent.rsi.engine.storage import atomic_json
sys.stdin.read(1)
try:
    atomic_json(Path(sys.argv[1]), {'writer': int(sys.argv[2]), 'rows': list(range(5000))}, exclusive=True)
except FileExistsError:
    raise SystemExit(17)
"""
    workers = []
    try:
        for index in range(8):
            workers.append(subprocess.Popen(
                [sys.executable, "-c", script, str(path), str(index)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            ))
        for worker in workers:
            worker.stdin.write("1")
            worker.stdin.flush()
        for worker in workers:
            _, stderr = worker.communicate(timeout=15)
            assert worker.returncode in {0, 17}, stderr
        codes = [worker.returncode for worker in workers]
        assert codes.count(0) == 1 and codes.count(17) == 7
        data = json.loads(path.read_text())
        assert data == {"writer": codes.index(0), "rows": list(range(5000))}
        assert path.stat().st_nlink == 1
        assert path.stat().st_mode & 0o777 == 0o600
        assert list(tmp_path.iterdir()) == [path]
    finally:
        for worker in workers:
            if worker.poll() is None:
                worker.kill()
            worker.wait(timeout=5)
