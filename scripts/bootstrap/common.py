from __future__ import annotations

import hashlib
import json
import os
import re
import signal
import subprocess
import tempfile
import time
from pathlib import Path
from . import progress


def digest(path: Path, algorithm="sha256") -> str:
    h = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path, default=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".init-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def inside(root: Path, value: str | Path) -> Path:
    """Validate before mutation, including symlink escapes; preserve venv paths."""
    candidate = Path(os.path.abspath(root / value))
    if not candidate.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Path leaves the project: {value}")
    return candidate


def add_missing(old: dict, new: dict) -> dict:
    """Existing administrator paths, limits and disabled switches always win."""
    result = dict(old)
    for key, value in new.items():
        if key not in result:
            result[key] = value
        elif isinstance(value, dict) and isinstance(result[key], dict):
            result[key] = add_missing(result[key], value)
    return result


class InstallLock:
    """Kernel-owned lock; a crashed installer does not leave a stale lock."""
    def __init__(self, path):
        self.path = path

    def __enter__(self):
        import fcntl
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("a+")
        try:
            fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.stream.close()
            raise RuntimeError("Another initializer is running in this checkout") from None
        return self

    def __exit__(self, *_):
        self.stream.close()


def clean_env():
    env = {k: v for k, v in os.environ.items()
           if not any(x in k.upper() for x in ("TOKEN", "API_KEY", "SECRET", "PASSWORD"))}
    for key in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"):
        env.pop(key, None)
    env.update(PYTHON_DOTENV_DISABLED="1", PYTHONUNBUFFERED="1", PYTHONNOUSERSITE="1",
               GIT_TERMINAL_PROMPT="0", PIP_NO_INPUT="1", PIP_DISABLE_PIP_VERSION_CHECK="1",
               HF_HUB_DISABLE_IMPLICIT_TOKEN="1", OMP_NUM_THREADS="2",
               OPENBLAS_NUM_THREADS="2", MAX_JOBS="2", MAKEFLAGS="-j2")
    return env


class Runner:
    def __init__(self, root: Path, log: Path):
        self.root, self.log = root, log

    def __call__(self, argv, *, cwd=None, timeout=3600, extra_env=None):
        self.log.parent.mkdir(parents=True, exist_ok=True)
        args = [str(x) for x in argv]
        env = clean_env()
        env.update(extra_env or {})
        # Logs stay on the installation host. Never print environment variables.
        with self.log.open("a", encoding="utf-8") as log:
            log.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] {json.dumps(args)}\n")
            log.flush()
            process = subprocess.Popen(args, cwd=cwd or self.root, env=env,
                                       stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            try:
                started = time.monotonic()
                label = Path(args[0]).name
                if '-m' in args:
                    label += ' ' + args[args.index('-m') + 1]
                elif len(args) > 1 and args[1] in ('ci', 'run', 'install', 'pip', 'venv', 'create', 'fetch', 'checkout'):
                    label += ' ' + args[1]
                while True:
                    elapsed = time.monotonic() - started
                    progress.ui.update(label, elapsed)
                    try:
                        code = process.wait(timeout=min(1, max(0.01, timeout - elapsed)))
                        break
                    except subprocess.TimeoutExpired:
                        if time.monotonic() - started >= timeout:
                            raise
                progress.ui.line(f'  {label}: {"done" if code == 0 else "failed"} ({time.monotonic() - started:.0f}s)')
            except BaseException:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=5)
                except (ProcessLookupError, subprocess.TimeoutExpired):
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait()
                raise
        if code:
            with self.log.open('rb') as stream:
                stream.seek(max(0, self.log.stat().st_size - 16384))
                tail = stream.read().decode('utf-8', errors='replace')
            # Surface a concise cause without copying arbitrary package output,
            # URLs, environment values or credentials into the terminal summary.
            statuses = re.findall(r'\b(401|403|404|408|429|5\d\d) (?:Client|Server) Error', tail)
            cause = ''
            if statuses:
                cause = f'; HTTP {statuses[-1]}'
                if statuses[-1] == '429':
                    cause += ' download rate limit; retry later'
            raise RuntimeError(f"{Path(args[0]).name} exited {code}{cause}; see {self.log}")


def capture(argv, cwd=None, timeout=30):
    return subprocess.check_output([str(x) for x in argv], cwd=cwd,
                                   env=clean_env(), text=True, stderr=subprocess.PIPE,
                                   timeout=timeout).strip()
