"""Service reuse must preserve code, mode and readiness boundaries."""
import pytest
from mcp_tools.scheduler import service


@pytest.fixture
def current_service(tmp_path, monkeypatch):
    expected = {"service": "venusfactory-tools", "repository_id": service.REPOSITORY_ID,
                "source_id": "current-source", "runtime_id": "current-runtime"}
    state = {**expected, "status": "ready", "pid": 1234}
    monkeypatch.setattr(service, "control_dir", lambda: tmp_path)
    monkeypatch.setattr(service, "identity", lambda: expected, raising=False)
    monkeypatch.setattr(service, "health", lambda: dict(state))
    def never_spawn(*args, **kwargs):
        pytest.fail("A service already owns this port; do not spawn another")
    monkeypatch.setattr(service.subprocess, "Popen", never_spawn)
    return state


def test_same_ready_service_is_reused(current_service):
    assert service.ensure_service() == current_service


@pytest.mark.parametrize("field,value", [
    ("source_id", "previous-source"),
    ("runtime_id", "another-runtime"),
    ("service", "unrelated-server"),
    ("repository_id", "another-checkout"),
])
def test_incompatible_service_is_not_reused_or_replaced(current_service, field, value):
    current_service[field] = value
    with pytest.raises(RuntimeError):
        service.ensure_service()


def test_legacy_service_without_revision_requires_restart(current_service):
    current_service.pop("source_id")
    current_service.pop("runtime_id")
    with pytest.raises(RuntimeError):
        service.ensure_service()


def test_unready_service_is_not_returned(current_service):
    current_service["status"] = "failed"
    with pytest.raises(RuntimeError):
        service.ensure_service()


@pytest.mark.parametrize("state", [None, [], False, "wrong", {"service": "wrong"}])
def test_malformed_listener_is_not_replaced(current_service, monkeypatch, state):
    monkeypatch.setattr(service, "health", lambda: state)
    with pytest.raises(RuntimeError):
        service.ensure_service()


def test_http_failure_does_not_spawn_over_existing_listener(current_service, monkeypatch):
    from urllib.error import HTTPError
    def failed():
        raise HTTPError("http://127.0.0.1/health", 503, "unavailable", {}, None)
    monkeypatch.setattr(service, "health", failed)
    with pytest.raises(RuntimeError, match="HTTP 503"):
        service.ensure_service()


def test_runtime_identity_separates_modes_queues_and_python(tmp_path, monkeypatch):
    queue = [tmp_path / "first"]
    monkeypatch.setattr(service, "control_dir", lambda: queue[0])
    monkeypatch.setenv("WEBUI_V2_MODE", "online")
    monkeypatch.setenv("DMX_API_KEY", "synthetic-private-marker")
    first = service.identity()
    assert "synthetic-private-marker" not in str(first)
    assert str(tmp_path) not in str(first)
    monkeypatch.setenv("WEBUI_V2_MODE", "local")
    assert service.identity()["runtime_id"] != first["runtime_id"]
    monkeypatch.setenv("WEBUI_V2_MODE", "online")
    queue[0] = tmp_path / "second"
    assert service.identity()["runtime_id"] != first["runtime_id"]
    queue[0] = tmp_path / "first"
    monkeypatch.setattr(service.sys, "prefix", "/another-python-environment")
    assert service.identity()["runtime_id"] != first["runtime_id"]


@pytest.fixture
def child_service(tmp_path, monkeypatch):
    import json
    import socket
    import subprocess
    import sys
    from backend.web_v2.job_queue import process_token, stop_process_tree

    expected = {"service": "venusfactory-tools", "repository_id": service.REPOSITORY_ID,
                "source_id": "fixture-source", "runtime_id": "fixture-runtime"}
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    monkeypatch.setenv("MCP_HTTP_PORT", str(port))
    monkeypatch.setattr(service, "control_dir", lambda: tmp_path)
    monkeypatch.setattr(service, "identity", lambda: expected)
    monkeypatch.setattr(service, "STARTUP_TIMEOUT", 5)
    source = tmp_path / "fixture_service.py"
    payload = {**expected, "status": "ready"}
    source.write_text("""
import json, os
from http.server import BaseHTTPRequestHandler, HTTPServer
payload = """ + repr(payload) + """
payload['pid'] = os.getpid()
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        raw = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header('Content-Length', str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)
    def log_message(self, *args): pass
HTTPServer(('127.0.0.1', """ + str(port) + """), Handler).serve_forever()
""")
    original_popen = subprocess.Popen
    children = []
    def spawn(command, **kwargs):
        child = original_popen([sys.executable, "-u", str(source)], **kwargs)
        children.append(child)
        return child
    monkeypatch.setattr(service.subprocess, "Popen", spawn)
    try:
        yield tmp_path, expected, children, source
    finally:
        for child in children:
            if child.poll() is None:
                stop_process_tree(child.pid, process_token(child.pid))
            child.wait(timeout=5)


def test_new_service_is_ready_then_reused_with_its_pid(child_service):
    import json
    folder, expected, children, source = child_service
    state = service.ensure_service()
    assert state == {**expected, "status": "ready", "pid": children[0].pid}
    assert service.ensure_service() == state
    assert len(children) == 1
    record = json.loads((folder / "process.json").read_text())
    assert record["source_id"] == expected["source_id"]
    assert record["token"]
    assert (folder / "process.json").stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize("failure", ["timeout", "exit", "version", "pid"])
def test_failed_start_reaps_owned_child_and_removes_process_record(child_service, monkeypatch, failure):
    folder, expected, children, source = child_service
    if failure == "timeout":
        source.write_text("import time; time.sleep(60)")
        monkeypatch.setattr(service, "STARTUP_TIMEOUT", .4)
    elif failure == "exit":
        source.write_text("raise SystemExit(1)")
    elif failure == "version":
        source.write_text(source.read_text().replace("fixture-source", "old-source"))
    else:
        source.write_text(source.read_text().replace("os.getpid()", "os.getpid() + 1"))
    with pytest.raises(RuntimeError):
        service.ensure_service()
    assert len(children) == 1 and children[0].poll() is not None
    assert not (folder / "process.json").exists()
