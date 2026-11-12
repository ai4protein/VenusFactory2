"""Loopback-only MCP service, reusable only within the same runtime generation."""
import hashlib
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import time
from urllib.error import HTTPError
from urllib.request import urlopen

from backend.runtime_identity import loaded_source_revision

ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ID = hashlib.sha256(str(ROOT).encode()).hexdigest()[:16]
STARTUP_TIMEOUT = 45


def port():
    return int(os.getenv("MCP_HTTP_PORT", "8080"))


def base_url():
    return "http://127.0.0.1:" + str(port())


def control_dir():
    from backend.web_v2.job_queue import JobQueue
    path = JobQueue().root / "mcp-service" / str(port())
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    return path


def identity(*, scheduler_only=False):
    runtime = [str(control_dir().resolve()), os.getenv("WEBUI_V2_MODE", "local").strip().lower(),
               sys.executable, sys.prefix, sys.version, "scheduler-only" if scheduler_only else "public-and-scheduler"]
    return {"service": "venusfactory-tools", "repository_id": REPOSITORY_ID,
            "source_id": loaded_source_revision(),
            "runtime_id": hashlib.sha256(json.dumps(runtime).encode()).hexdigest()}


def service_key():
    path = control_dir() / "service.key"
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        pass
    else:
        with os.fdopen(fd, "w") as stream:
            stream.write(secrets.token_urlsafe(48))
    return path.read_text().strip()


def health():
    with urlopen(base_url() + "/health", timeout=2) as response:
        raw = response.read(65537)
    if len(raw) > 65536:
        raise ValueError("Oversized MCP health response")
    return json.loads(raw)


def _reachable_health():
    try:
        state = health()
        if not isinstance(state, dict):
            raise RuntimeError("Existing MCP service returned invalid health data")
        return state
    except HTTPError as exc:
        raise RuntimeError(f"Existing MCP service health check failed (HTTP {exc.code})") from None
    except ValueError:
        raise RuntimeError("Existing MCP service returned invalid health data") from None
    except OSError:
        return None


def _check_identity(state, expected, *, pid=None):
    if not isinstance(state, dict) or any(state.get(key) != expected[key] for key in ("service", "repository_id")):
        raise RuntimeError("MCP port belongs to a different service")
    if state.get("source_id") != expected["source_id"]:
        raise RuntimeError("MCP service code version differs; stop the existing service and restart the bundle")
    if state.get("runtime_id") != expected["runtime_id"]:
        raise RuntimeError("MCP service belongs to a different runtime; use a separate MCP port")
    if type(state.get("pid")) is not int or state["pid"] < 1 or (pid is not None and state["pid"] != pid):
        raise RuntimeError("MCP service process identity differs")


def ensure_service(*, listener=None, scheduler_only=False, env_overrides=None):
    import fcntl
    from backend.web_v2.job_queue import process_token, stop_process_tree

    control = control_dir()
    expected = identity(scheduler_only=True) if scheduler_only else identity()
    if listener is not None:
        import socket
        if (listener.getsockname() != ("127.0.0.1", port())
                or not listener.getsockopt(socket.SOL_SOCKET, socket.SO_ACCEPTCONN)):
            raise ValueError("MCP listener must be a bound loopback listening socket")
    with (control / "startup.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = _reachable_health() if listener is None else None
        if state is not None:
            _check_identity(state, expected)
            if state.get("status") != "ready":
                raise RuntimeError("Existing MCP service is not ready; inspect its server log")
            return state

        service_key()
        bootstrap = control / "bootstrap.py"
        paths = [str(ROOT / "src"), str(ROOT), *(p for p in sys.path if p and Path(p).exists())]
        options = {"_VENUS_SCHEDULER_ONLY": scheduler_only, "_VENUS_LISTENER_FD": listener.fileno() if listener is not None else None}
        bootstrap.write_text("import sys,runpy,os\nsys.path=" + repr(paths) + "\nos.chdir(" + repr(str(ROOT)) +
                             ")\nrunpy.run_module(\"mcp_tools.mcp_server\",run_name=\"__main__\",init_globals=" + repr(options) + ")\n")
        env = dict(os.environ, MCP_HTTP_HOST="127.0.0.1", MCP_HTTP_PORT=str(port()), PYTHONDONTWRITEBYTECODE="1")
        if env_overrides:
            env.update(env_overrides)
        with (control / "server.log").open("ab") as log:
            proc = subprocess.Popen([sys.executable, "-I", "-u", str(bootstrap)], cwd=ROOT, env=env,
                                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                    pass_fds=(listener.fileno(),) if listener is not None else ())
        token = process_token(proc.pid)
        record = {"pid": proc.pid, "token": token, **expected}
        record_path = control / "process.json"
        try:
            pending = record_path.with_suffix(".tmp")
            pending.write_text(json.dumps(record))
            pending.chmod(0o600)
            pending.replace(record_path)
            deadline = time.monotonic() + STARTUP_TIMEOUT
            while time.monotonic() < deadline:
                if proc.poll() is not None:
                    raise RuntimeError("MCP service failed to start; inspect its server log")
                state = _reachable_health()
                if state is not None:
                    _check_identity(state, expected, pid=proc.pid)
                    if state.get("status") == "ready":
                        return state
                    if state.get("status") != "starting":
                        raise RuntimeError("MCP service failed its readiness check")
                time.sleep(.3)
            raise RuntimeError("MCP service did not become ready")
        except BaseException:
            # Only this invocation's new child is owned here. Never stop a reused
            # or conflicting listener; the operator must drain that deployment.
            stop_process_tree(proc.pid, token)
            if proc.poll() is None:
                proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
            if record_path.exists() and json.loads(record_path.read_text()) == record:
                record_path.unlink()
            raise
