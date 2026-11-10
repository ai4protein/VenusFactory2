import importlib.util
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location("deployment_services", Path(__file__).resolve().parents[1] / "scripts/deploy_services.py")
deployment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(deployment)


def test_both_modes_override_inherited_paths_and_use_distinct_ports(tmp_path):
    args = deployment.arguments(["--runtime-root", str(tmp_path)])
    plans = deployment.service_plan(args, {"VENUS_JOB_QUEUE_ROOT": "/old/shared", "VENUS_USAGE_DB": "/old/shared.db"})
    assert [p["mode"] for p in plans] == ["online", "local"]
    assert len({p[k] for p in plans for k in ["port", "mcp_port"]}) == 4
    for plan in plans:
        assert Path(plan["env"]["VENUS_JOB_QUEUE_ROOT"]) == tmp_path / plan["mode"] / "queue"
        assert Path(plan["env"]["VENUS_USAGE_DB"]).is_relative_to(tmp_path / plan["mode"])
        assert "--mode" in plan["command"] and plan["mode"] in plan["command"]
        assert Path(plan["env"]["KIMI_CODE_HOME"]) == tmp_path / plan["mode"] / "control/kimi"


def test_harness_settings_are_private_and_do_not_copy_global_sessions_or_mcp(tmp_path):
    source = tmp_path / "personal"
    source.mkdir()
    for name in ["config.toml", "mcp.json", "sessions.json"]:
        (source / name).write_text("fixture")
    plans = deployment.service_plan(deployment.arguments(["--runtime-root", str(tmp_path / "runtime")]), {})
    for plan in plans:
        deployment.prepare_harness_home(plan, source)
        home = Path(plan["env"]["KIMI_CODE_HOME"])
        assert (home / "config.toml").read_text() == "fixture"
        assert not (home / "mcp.json").exists()
        assert not (home / "sessions.json").exists()
        (home / "config.toml").write_text("mode-specific")
        deployment.prepare_harness_home(plan, source)
        assert (home / "config.toml").read_text() == "mode-specific"


def test_port_collision_and_partial_tunnel_configuration_fail_before_launch():
    with pytest.raises(ValueError, match="distinct"):
        deployment.service_plan(deployment.arguments(["--online-port", "7862"]))
    with pytest.raises(SystemExit):
        deployment.arguments(["--frp-config", "private.ini"])


def test_stopping_bundle_leaves_unrelated_process_alive():
    import subprocess
    import sys
    managed = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        deployment.stop_children([managed])
        assert managed.poll() is not None
        assert unrelated.poll() is None
    finally:
        for child in [managed, unrelated]:
            if child.poll() is None:
                child.terminate()
            child.wait(timeout=5)


@pytest.fixture
def readiness_server():
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    responses = {
        "/health": (200, {"success": True}),
        "/api/chat/agent-execution-options": (200, {"harnesses": [], "models": [], "combinations": []}),
        "/api/chat/science-goals/catalog?view=composer": (200, {"drivers": [], "execution": {}}),
        "/api/report/options": (200, {"models": [], "analyses": [], "chapters": []}),
        "/api/model-platform/catalog?limit=1": (200, {"items": [], "categories": {}, "total": 0}),
        "/api/model-platform/workflow-components?limit=1": (200, {"items": [], "categories": {}, "kinds": {}, "total": 0}),
    }
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append(self.path)
            status, body = responses.get(self.path, (404, {}))
            payload = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_port, responses, requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.mark.parametrize("endpoint", [
    "/api/chat/agent-execution-options",
    "/api/chat/science-goals/catalog?view=composer",
    "/api/report/options",
    "/api/model-platform/catalog?limit=1",
    "/api/model-platform/workflow-components?limit=1",
])
def test_health_success_does_not_hide_broken_workspace_catalog(readiness_server, endpoint):
    port, responses, requests = readiness_server
    original = responses[endpoint]
    responses[endpoint] = (500, {"detail": "ImportError"})
    assert not deployment.service_ready(port, "api")
    assert endpoint in requests
    responses[endpoint] = original
    assert deployment.service_ready(port, "api")


def test_initial_installation_can_start_before_api_keys_are_configured(readiness_server):
    port, responses, requests = readiness_server
    assert deployment.service_ready(port, "api")
    assert requests == list(responses)


@pytest.mark.parametrize("endpoint,body", [
    ("/health", []),
    ("/api/chat/agent-execution-options", {"success": True}),
    ("/api/chat/science-goals/catalog?view=composer", None),
    ("/api/report/options", {"models": [], "analyses": [], "chapters": {}}),
    ("/api/model-platform/catalog?limit=1", {"items": [], "categories": {}, "total": True}),
    ("/api/model-platform/workflow-components?limit=1", {"items": [], "categories": {}, "total": 0}),
])
def test_malformed_catalog_is_not_ready(readiness_server, endpoint, body):
    port, responses, _ = readiness_server
    responses[endpoint] = (200, body)
    assert not deployment.service_ready(port, "api")


def test_mcp_readiness_does_not_request_web_catalogs(readiness_server):
    port, responses, requests = readiness_server
    responses["/health"] = (200, {"status": "ready"})
    assert deployment.service_ready(port, "mcp")
    assert requests == ["/health"]


def test_deployment_identity_tracks_source_and_frontend_without_credentials(tmp_path, monkeypatch):
    import hashlib
    import subprocess

    root = tmp_path / "release"
    root.mkdir()
    dist = tmp_path / "frontend"
    dist.mkdir()
    (dist / "index.html").write_text("fixture frontend")
    (root / "app.py").write_text("version = 1\n")
    for args in (("init",), ("add", "app.py"),
                 ("-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "-m", "fixture")):
        subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)
    monkeypatch.setenv("DMX_API_KEY", "synthetic-do-not-record")
    first = deployment.deployment_identity(dist, root)
    assert len(first["source_revision"]) == 40
    assert first["source_dirty"] is False
    assert first["frontend_index_sha256"] == hashlib.sha256(b"fixture frontend").hexdigest()
    assert "synthetic-do-not-record" not in str(first)
    (root / "app.py").write_text("version = 2\n")
    (dist / "index.html").write_text("second frontend")
    second = deployment.deployment_identity(dist, root)
    assert second["source_revision"] == first["source_revision"]
    assert second["source_dirty"] is True
    assert second["frontend_index_sha256"] != first["frontend_index_sha256"]


def test_source_archive_can_start_without_a_git_repository(tmp_path):
    root, dist = tmp_path / "release", tmp_path / "frontend"
    root.mkdir(); dist.mkdir()
    (dist / "index.html").write_text("fixture")
    identity = deployment.deployment_identity(dist, root)
    assert identity["source_revision"] is None
    assert identity["source_dirty"] is None
    assert len(identity["frontend_index_sha256"]) == 64


def test_readiness_diagnostic_identifies_endpoint_without_response_content(readiness_server):
    port, responses, _ = readiness_server
    endpoint = "/api/report/options"
    responses[endpoint] = (500, {"detail": "private configuration must not be logged"})
    failure = deployment.readiness_failure(port, "api")
    assert failure == {"path": endpoint, "reason": "HTTP 500"}
    assert "private" not in str(failure)


def test_readiness_rejects_oversized_response(readiness_server):
    port, responses, _ = readiness_server
    responses["/health"] = (200, {"success": True, "padding": "x" * deployment._MAX_READINESS_BYTES})
    assert deployment.readiness_failure(port, "api") == {"path": "/health", "reason": "invalid JSON response"}


def test_failed_catalog_prevents_tunnel_and_records_startup_failure(tmp_path, monkeypatch):
    import json

    dist = tmp_path / "frontend"
    dist.mkdir()
    (dist / "index.html").write_text("fixture")
    runtime = tmp_path / "runtime"
    plan = {"mode": "online", "port": 46001, "mcp_port": 46002,
            "runtime": runtime / "online", "command": ["fixture-service"], "env": {}}
    monkeypatch.setattr(deployment, "service_plan", lambda args: [plan])
    monkeypatch.setattr(deployment, "prepare_harness_home", lambda plan: None)
    monkeypatch.setattr(deployment, "deployment_identity", lambda dist: {"source_revision": "fixture"})
    monkeypatch.setattr(deployment, "remember_children", lambda *args: None)
    monkeypatch.setattr(deployment.signal, "signal", lambda *args: None)
    clock = iter([0, 121])
    monkeypatch.setattr(deployment.time, "monotonic", lambda: next(clock))

    class Socket:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def setsockopt(self, *args): pass
        def bind(self, *args): pass

    class Process:
        pid = 12345
        def poll(self): return None

    launched, stopped = [], []
    monkeypatch.setattr(deployment.socket, "socket", lambda: Socket())
    monkeypatch.setattr(deployment.subprocess, "Popen", lambda command, **kwargs: (launched.append(command), Process())[1])
    monkeypatch.setattr(deployment, "stop_children", lambda children, owned: stopped.extend(children))
    failure = {"path": "/api/report/options", "reason": "HTTP 500"}
    monkeypatch.setattr(deployment, "readiness_failure", lambda port, kind: failure if kind == "api" else None)
    with pytest.raises(RuntimeError, match=r"api:46001 /api/report/options.*HTTP 500"):
        deployment.main(["--mode", "online", "--runtime-root", str(runtime),
                         "--frontend-dist", str(dist), "--frp-executable", str(tmp_path / "frpc"),
                         "--frp-config", str(tmp_path / "frpc.ini")])
    assert launched == [["fixture-service"]]
    assert len(stopped) == 1
    state = json.loads((runtime / "services/deployment.json").read_text())
    assert state["status"] == "failed"
    assert state["pending_checks"] == [{"port": 46001, "kind": "api", **failure}]
