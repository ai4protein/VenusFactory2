import os
from pathlib import Path
import subprocess
import sys

import pytest
from fastapi.testclient import TestClient
from fastmcp import FastMCP


def test_health_binds_the_started_application_identity(monkeypatch):
    from mcp_tools.scheduler import mcp_service
    current = {"service": "venusfactory-tools", "repository_id": "repository",
               "source_id": "source", "runtime_id": "runtime"}
    monkeypatch.setattr(mcp_service, "identity", lambda: dict(current))
    app = mcp_service.application(FastMCP("fixture"))
    client = TestClient(app)
    first = client.get("/health").json()
    assert first.items() >= current.items()
    assert first["status"] == "ready" and first["pid"] > 0
    current["source_id"] = "changed-files"
    assert client.get("/health").json() == first


@pytest.mark.parametrize("mode", ["online", "local"])
def test_real_mcp_startup_and_reuse_with_external_network_blocked(tmp_path, mode):
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, WEBUI_V2_MODE=mode, PYTHON_DOTENV_DISABLED="1",
               VENUS_RUNTIME_ROOT=str(tmp_path / "runtime"),
               VENUS_MODEL_CONFIG=str(root / "model-config.yaml.example"),
               VENUS_MODEL_RUNTIMES_CONFIG=str(tmp_path / "absent-runtimes.json"),
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", VENUS_WORKER_AUTOSTART="0")
    from backend.web_v2.runtime_paths import AREAS
    for name in AREAS:
        env.pop(name, None)
    code = r"""
import asyncio, json, os, socket, subprocess
from pathlib import Path
from backend.web_v2.runtime_paths import configure_runtime_paths
configure_runtime_paths()
from backend.web_v2.job_queue import process_token, stop_process_tree
from mcp_tools.scheduler import service

# Apply the same outbound guard to both parent and real MCP child.
guard = (
    'import socket\n'
    '_original_connect = socket.socket.connect\n'
    'def loopback_only(self, address):\n'
    '    if isinstance(address, tuple) and address[0] not in ("127.0.0.1", "::1", "localhost"):\n'
    '        raise AssertionError("MCP startup must not contact external services")\n'
    '    return _original_connect(self, address)\n'
    'socket.socket.connect = loopback_only\n'
)
exec(guard)
with socket.socket() as probe:
    probe.bind(('127.0.0.1', 0))
    os.environ['MCP_HTTP_PORT'] = str(probe.getsockname()[1])
original_popen = subprocess.Popen
children = []
def guarded_spawn(command, **kwargs):
    bootstrap = Path(command[-1])
    bootstrap.write_text(guard + bootstrap.read_text())
    child = original_popen(command, **kwargs)
    children.append(child)
    return child
service.subprocess.Popen = guarded_spawn
try:
    first = service.ensure_service()
    assert first['status'] == 'ready'
    assert len(children) == 1 and first['pid'] == children[0].pid
    assert first == service.ensure_service()
    assert len(children) == 1
    assert set(first) == {'service', 'repository_id', 'source_id', 'runtime_id', 'pid', 'status'}
    service.subprocess.Popen = original_popen
    from fastmcp import Client
    async def inventory():
        async with Client(service.base_url() + '/mcp') as client:
            return await client.list_tools()
    tools = asyncio.run(inventory())
    assert tools
    print(json.dumps({'mode': os.environ['WEBUI_V2_MODE'], 'ready': True,
                      'tool_count': len(tools), 'reused_pid': first['pid']}))
finally:
    service.subprocess.Popen = original_popen
    for child in children:
        if child.poll() is None:
            stop_process_tree(child.pid, process_token(child.pid))
        child.wait(timeout=5)
"""
    result = subprocess.run([sys.executable, "-c", code], cwd=root, env=env,
                            capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
