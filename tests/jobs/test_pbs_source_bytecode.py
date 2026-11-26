"""Real isolated subprocess regression for immutable PBS source snapshots."""
from __future__ import annotations

import hashlib
import json
import os
import socket
import subprocess
import sys

from backend.web_v2.job_queue import process_token, stop_process_tree
from mcp_tools.scheduler import service


def fixture_package(root):
    package = root / "mcp_tools"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "probe.py").write_text("VALUE = 42\n")
    return package


def source_hashes(root):
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in root.rglob("*") if path.is_file()}


def test_real_isolated_service_never_writes_frozen_bytecode(tmp_path, monkeypatch):
    # Positive control proves PYTHONDONTWRITEBYTECODE cannot protect python -I.
    control_source = tmp_path / "positive-control"
    fixture_package(control_source)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    code = "import sys; sys.path.insert(0, sys.argv[1]); import mcp_tools.probe; print(int(sys.dont_write_bytecode))"
    result = subprocess.run([sys.executable, "-I", "-c", code, str(control_source)],
                            env=env, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0 and result.stdout.strip() == "0"
    assert list(control_source.rglob("*.pyc"))

    frozen = tmp_path / "frozen"
    package = fixture_package(frozen)
    control = tmp_path / "private-control"
    control.mkdir(mode=0o700)
    flags = control / "interpreter-flags.json"
    monkeypatch.setattr(service, "ROOT", frozen)
    monkeypatch.setattr(service, "control_dir", lambda: control)
    monkeypatch.delenv("PYTHONDONTWRITEBYTECODE", raising=False)
    expected = service.identity(scheduler_only=True)
    source = """
import json, os, socket, sys
from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer
from mcp_tools import probe
assert probe.VALUE == 42
Path(FLAGS).write_text(json.dumps({'isolated': sys.flags.isolated, 'no_bytecode': sys.dont_write_bytecode}))
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        data = json.dumps(dict(EXPECTED, pid=os.getpid(), status='ready')).encode()
        self.send_response(200)
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)
    def log_message(self, *args):
        pass
server = HTTPServer(('127.0.0.1', 0), Handler, bind_and_activate=False)
server.socket.close()
server.socket = socket.socket(fileno=_VENUS_LISTENER_FD)
server.server_address = server.socket.getsockname()
server.serve_forever()
""".replace("FLAGS", repr(str(flags))).replace("EXPECTED", repr(expected))
    (package / "mcp_server.py").write_text(source)
    before = source_hashes(frozen)
    state = None
    try:
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen(128)
            monkeypatch.setenv("MCP_HTTP_PORT", str(listener.getsockname()[1]))
            state = service.ensure_service(listener=listener, scheduler_only=True)
        assert state["status"] == "ready"
        assert json.loads(flags.read_text()) == {"isolated": 1, "no_bytecode": True}
        assert source_hashes(frozen) == before
        assert not list(frozen.rglob("__pycache__"))
    finally:
        if state is not None:
            stop_process_tree(state["pid"], process_token(state["pid"]))
            os.waitpid(state["pid"], 0)
    assert source_hashes(frozen) == before


def test_actual_runtime_command_forces_no_bytecode_in_frozen_source(tmp_path, monkeypatch):
    from mcp_tools.models import execution

    frozen = tmp_path / "frozen"
    frozen.mkdir()
    (frozen / "model_probe.py").write_text("VALUE = 42\n")
    before = source_hashes(frozen)
    monkeypatch.setattr(execution, "PROJECT", frozen)
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "0")
    script = "import json,sys,model_probe; print(json.dumps({'value': model_probe.VALUE, 'no_bytecode': sys.dont_write_bytecode}))"
    spec = execution.command("fixture", ["-c", script], settings={
        "fixture": {"python": sys.executable, "timeout_seconds": 10},
    })
    assert spec.env["PYTHONDONTWRITEBYTECODE"] == "1"
    assert json.loads(execution.execute(spec)) == {"value": 42, "no_bytecode": True}
    assert source_hashes(frozen) == before
    assert not list(frozen.rglob("__pycache__"))
