import fcntl
import subprocess
import sys

import pytest

from agent.rsi.storage import case_gpu_lock


def test_inherited_lock_is_not_reacquired_or_released(tmp_path, monkeypatch):
    path = tmp_path / "gpu.lock"
    with path.open("a") as owned:
        fcntl.flock(owned, fcntl.LOCK_EX)
        monkeypatch.setenv("VENUS_RSI_CASE_LOCK_FD", str(owned.fileno()))
        with case_gpu_lock(path):
            pass
        probe = subprocess.run(
            [
                sys.executable,
                "-c",
                "import fcntl,sys; f=open(sys.argv[1],'a'); fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)",
                str(path),
            ],
            capture_output=True,
        )
        assert probe.returncode != 0 and b"BlockingIOError" in probe.stderr


def test_inherited_wrong_file_rejected(tmp_path, monkeypatch):
    path = tmp_path / "gpu.lock"
    path.touch()
    with (tmp_path / "wrong.lock").open("a") as wrong:
        monkeypatch.setenv("VENUS_RSI_CASE_LOCK_FD", str(wrong.fileno()))
        with pytest.raises(ValueError, match="declared lock"):
            with case_gpu_lock(path):
                pass
