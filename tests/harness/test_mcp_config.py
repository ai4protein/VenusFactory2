"""Real stdio protocol checks, without a model or native client subscription."""

import json
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from agent.harness.contracts import Budget, ModelBinding, Profile
from agent.harness.mcp_config import (
    claude_mcp_configuration,
    codex_mcp_configuration,
    connection_manifest,
    deepseek_mcp_plugin,
    scientific_server,
)
from agent.harness.registry import builtin_registry


@pytest.mark.asyncio
@pytest.mark.parametrize("client", ["claude-code", "codex", "deepseek-harness"])
async def test_exported_connection_reaches_real_mcp_server(tmp_path, client):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    cap = tmp_path / 'cap with "quotes"'
    cap.mkdir()
    (cap / "generated_tools.py").write_text(
        "from mcp.server.fastmcp import FastMCP\n"
        'mcp = FastMCP("offline-science")\n'
        "@mcp.tool()\n"
        "def protein_length(sequence: str) -> dict:\n"
        '    return {"length": len(sequence), "source": "offline_fixture"}\n'
        "mcp.run()\n"
    )
    server = scientific_server(
        entry="generated_tools.py",
        tools=["protein_length"],
        python=sys.executable,
        capability=str(cap),
        workspace=str(tmp_path),
        clean_pythonpath=client == "deepseek-harness",
    )
    if client == "claude-code":
        config = json.loads(json.dumps(claude_mcp_configuration(server)))["mcpServers"]["venusfactory"]
    elif client == "codex":
        config = tomllib.loads(codex_mcp_configuration(server, 12))["mcp_servers"]["venusfactory"]
        assert config["enabled_tools"] == ["protein_length"]
    else:
        config = deepseek_mcp_plugin(server)["config"]
        assert config["failOnStartupError"] is True
    async with stdio_client(
        StdioServerParameters(
            command=config["command"],
            args=config["args"],
            env=config["env"],
        )
    ) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            names = {t.name for t in (await session.list_tools()).tools}
            assert names == {"protein_length"}
            result = await session.call_tool("protein_length", {"sequence": "ACDEFG"})
            assert result.isError is False
            assert json.loads(result.content[0].text) == {"length": 6, "source": "offline_fixture"}
            missing = await session.call_tool("unregistered_tool", {})
            assert missing.isError is True


@pytest.mark.parametrize(
    "patch",
    [
        {"entry": "../generated_tools.py"},
        {"python": "python"},
        {"workspace": "/tmp\nworkspace"},
        {"tools": ["x", "x"]},
        {"tools": ["x.*"]},
        {"tools": [1]},
    ],
)
def test_invalid_tool_connections_are_rejected(patch):
    values = {"entry": "generated_tools.py", "tools": ["inspect_model_code_task"]}
    values.update(patch)
    with pytest.raises(ValueError):
        scientific_server(**values)


@pytest.mark.parametrize("wall", [0, -1, float("nan"), float("inf"), True])
def test_invalid_tool_timeout_is_rejected(wall):
    with pytest.raises(ValueError):
        codex_mcp_configuration(scientific_server(entry="generated_tools.py", tools=[]), wall)


def test_connection_digest_changes_with_task_scope():
    a = scientific_server(entry="generated_tools.py", tools=["inspect_model_code_task"])
    b = scientific_server(entry="generated_tools.py", tools=["predict_with_registered_model"])
    assert connection_manifest(a)["sha256"] != connection_manifest(b)["sha256"]
    assert connection_manifest(a)["native_runtime_verified"] is False
    assert connection_manifest(a)["registered_models"] == 0


@pytest.mark.parametrize("name,mode", [("claude-code", "science_agent"), ("expert-claude-code", "science_expert")])
def test_claude_native_mode_resolves_without_model_or_harness_fallback(tmp_path, name, mode):
    executable = tmp_path / "claude"
    executable.write_text("offline fixture")
    profile = Profile(
        mode, name, ModelBinding("dmx", "unisound-u2-flash", "https://www.dmxapi.cn/v1"), Budget(thinking="disabled")
    )
    registry = builtin_registry()
    adapter, check = registry.resolve(profile, {"claude_bin": str(executable)})
    assert check["ready"]
    assert type(adapter).__name__ == ("ClaudeAdapter" if mode == "science_agent" else "NativeExpertTeamAdapter")
    assert check["profile"]["model"] == profile.public()["model"]
    assert check["adapter"]["status"] == "experimental_messages_bridge"
    with pytest.raises(ValueError, match="runtime_missing"):
        registry.resolve(profile, {"kimi_bin": str(executable)})


@pytest.mark.parametrize("mode,name", [("science_agent", "claude-code"), ("science_expert", "expert-claude-code")])
@pytest.mark.parametrize("thinking", [None, "enabled"])
def test_claude_registry_rejects_controls_before_runtime_resolution(tmp_path, mode, name, thinking):
    executable = tmp_path / "claude"
    executable.write_text("offline fixture")
    profile = Profile(
        mode, name, ModelBinding("dmx", "unisound-u2-flash", "https://www.dmxapi.cn/v1"), Budget(thinking=thinking)
    )
    with pytest.raises(ValueError, match="claude_controls_incompatible"):
        builtin_registry().resolve(profile, {"claude_bin": str(executable)})


def test_cli_exports_without_credentials_or_overwriting_existing_files(tmp_path):
    root = Path(__file__).resolve().parents[2]
    target = tmp_path / "export"
    command = [
        sys.executable,
        str(root / "scripts/harness/export_scientific_mcp.py"),
        "--output",
        str(target),
        "--python",
        sys.executable,
        "--capability",
        "/capabilities",
        "--workspace",
        "/workspace",
        "--tool",
        "inspect_model_code_task",
    ]
    first = subprocess.run(command, capture_output=True, text=True, check=True)
    assert json.loads(first.stdout)["external_model_calls"] == 0
    assert {p.name for p in target.iterdir()} == {
        "claude.mcp.json",
        "codex.mcp.toml",
        "deepseek.plugin.json",
        "connection_manifest.json",
    }
    previous = {p.name: p.read_bytes() for p in target.iterdir()}
    second = subprocess.run(command, capture_output=True, text=True)
    assert second.returncode != 0
    assert {p.name: p.read_bytes() for p in target.iterdir()} == previous

@pytest.mark.parametrize("seconds", [0.25, 120, 900])
def test_deepseek_tool_timeout_matches_admitted_turn(seconds):
    server = scientific_server(entry="web_tools.py", tools=["call_tool"])
    config = deepseek_mcp_plugin(server, seconds)["config"]
    assert config["toolCallTimeoutMs"] == int(seconds * 1000)


@pytest.mark.parametrize("seconds", [0, -1, True, float("nan"), float("inf"), "900"])
def test_deepseek_rejects_unbounded_or_invalid_timeout(seconds):
    server = scientific_server(entry="web_tools.py", tools=["call_tool"])
    with pytest.raises(ValueError):
        deepseek_mcp_plugin(server, seconds)
