"""Idempotently register VenusFactory's MCP server with kimi-code.

kimi-code reads MCP server declarations from (in order, last write wins per name):
  - $KIMI_CODE_HOME/mcp.json  (or ~/.kimi-code/mcp.json)
  - <cwd>/.kimi-code/mcp.json
  - <cwd>/.mcp.json           (Claude-compatible, also honored)

Schema is the Claude-compatible `mcpServers` map. We register one entry
pointing at the HTTP MCP server already exposed by src/mcp_server.py.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

VENUSFACTORY_MCP_NAME = "venusfactory"
DEFAULT_MCP_URL = "http://127.0.0.1:8080/mcp"


def _config_path() -> Path:
    home = os.environ.get("KIMI_CODE_HOME") or str(Path.home() / ".kimi-code")
    return Path(home) / "mcp.json"


def ensure_registered(mcp_url: str = DEFAULT_MCP_URL, name: str = VENUSFACTORY_MCP_NAME) -> Path:
    """Write `name` -> `{url: mcp_url}` into the user-global kimi mcp.json.

    Preserves any other servers already configured. Returns the file path.
    Idempotent: if the entry already matches, no write happens.
    """
    try:
        from agent.kimi_skills import ensure_kimi_project_skills
        ensure_kimi_project_skills()
    except Exception:
        pass

    path = _config_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    data: dict[str, Any] = {"mcpServers": {}}
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8")) or {}
        except (json.JSONDecodeError, OSError):
            data = {"mcpServers": {}}
    if not isinstance(data, dict):
        data = {"mcpServers": {}}
    servers = data.setdefault("mcpServers", {})
    if not isinstance(servers, dict):
        servers = {}
        data["mcpServers"] = servers

    desired = {"url": mcp_url}
    if servers.get(name) == desired:
        return path

    servers[name] = desired
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return path
