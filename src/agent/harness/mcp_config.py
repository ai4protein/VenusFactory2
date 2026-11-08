"""Portable scientific MCP connection, independent of model/provider configuration.

This module also loads inside the standalone DeepSeek worker. Keep it stdlib-only.
Configuration compatibility is not native runtime or scientific conformance.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import PurePosixPath


def scientific_server(
    *,
    entry,
    tools,
    python="/opt/venus-runtime/bin/python",
    capability="/capabilities",
    workspace="/workspace",
    clean_pythonpath=False,
):
    if not isinstance(entry, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*\.py", entry):
        raise ValueError("Scientific MCP entry must be a Python filename inside the admitted capability")
    for value in (python, capability, workspace):
        if not isinstance(value, str) or not PurePosixPath(value).is_absolute() or any(ord(c) < 32 for c in value):
            raise ValueError("Scientific runtime paths must be absolute Linux paths")
    names = list(tools)
    if any(not isinstance(n, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", n) for n in names) or len(
        names
    ) != len(set(names)):
        raise ValueError("Invalid or repeated scientific tool name")
    args = ["-u", str(PurePosixPath(capability) / entry)]
    command = python
    if clean_pythonpath:
        command, args = "/usr/bin/env", ["-u", "PYTHONPATH", python, *args]
    return {
        "name": "venusfactory",
        "command": command,
        "args": args,
        "env": {"VENUS_RSI_WORKSPACE": workspace, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"},
        "tools": sorted(names),
    }


def claude_mcp_configuration(server):
    """MCP connection only: no inference credentials, model defaults or launch."""
    return {
        "mcpServers": {
            server["name"]: {
                "type": "stdio",
                "command": server["command"],
                "args": list(server["args"]),
                "env": dict(server["env"]),
            }
        }
    }


def codex_mcp_configuration(server, wall_seconds):
    if (
        isinstance(wall_seconds, bool)
        or not isinstance(wall_seconds, (int, float))
        or not math.isfinite(wall_seconds)
        or wall_seconds <= 0
    ):
        raise ValueError("Invalid scientific tool timeout")

    def q(value):
        return json.dumps(value, ensure_ascii=False)

    prefix = "mcp_servers." + server["name"]
    lines = [
        f"[{prefix}]",
        "command = " + q(server["command"]),
        "args = " + q(server["args"]),
        "required = true",
        "enabled_tools = " + q(server["tools"]),
        "startup_timeout_sec = 30",
        f"tool_timeout_sec = {int(wall_seconds) + 1}",
        f"[{prefix}.env]",
    ]
    lines += [name + " = " + q(value) for name, value in sorted(server["env"].items())]
    for name in server["tools"]:
        lines += [f"[{prefix}.tools.{q(name)}]", 'approval_mode = "approve"']
    return "\n".join(lines) + "\n"


def deepseek_mcp_plugin(server, wall_seconds=60):
    if (isinstance(wall_seconds, bool) or not isinstance(wall_seconds, (int, float))
            or not math.isfinite(wall_seconds) or wall_seconds <= 0):
        raise ValueError("Invalid scientific tool timeout")
    return {
        "id": server["name"],
        "name": "@deepseek-ai/dsh-mcp-client",
        "config": {
            "serverName": server["name"],
            "transport": "stdio",
            "command": server["command"],
            "args": list(server["args"]),
            "env": dict(server["env"]),
            "failOnStartupError": True,
            "toolCallTimeoutMs": max(1, math.ceil(wall_seconds * 1000)),
        },
    }


def connection_manifest(server):
    payload = json.dumps(server, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return {
        "schema": "scientific_mcp_connection.v1",
        "server": server,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "status": "configuration_only",
        "native_runtime_verified": False,
        "model_calls": 0,
        "registered_models": 0,
        "tool_admission": "Native adapters and the host retain tool admission and receipt verification; a connection file alone does not enforce it.",
        "inference": "Not configured. Use an independently admitted free provider; no implicit model or credential fallback.",
    }
