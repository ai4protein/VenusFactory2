"""Export portable science-tool connections without launching a model or client."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from agent.harness.mcp_config import (
    claude_mcp_configuration,
    codex_mcp_configuration,
    connection_manifest,
    deepseek_mcp_plugin,
    scientific_server,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--capability", required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--entry", default="generated_tools.py")
    parser.add_argument("--tool", action="append", required=True)
    parser.add_argument("--wall-seconds", type=float, default=900)
    args = parser.parse_args()
    server = scientific_server(
        entry=args.entry, tools=args.tool, python=args.python, capability=args.capability, workspace=args.workspace
    )
    files = {
        "claude.mcp.json": json.dumps(claude_mcp_configuration(server), indent=2),
        "codex.mcp.toml": codex_mcp_configuration(server, args.wall_seconds),
        "deepseek.plugin.json": json.dumps(deepseek_mcp_plugin(server), indent=2),
        "connection_manifest.json": json.dumps(connection_manifest(server), indent=2),
    }
    args.output.mkdir(parents=True, exist_ok=False)
    for name, content in files.items():
        (args.output / name).write_text(content + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "configuration_only",
                "output": str(args.output.resolve()),
                "external_model_calls": 0,
                "native_runtime_verified": False,
            }
        )
    )


if __name__ == "__main__":
    main()
