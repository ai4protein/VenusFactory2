"""Real, bounded MCP/inference conformance check; not a protein performance experiment."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import tomllib
from dataclasses import replace
from pathlib import Path

from agent.harness import Budget, ModelBinding, Profile, Task, builtin_registry
from agent.harness.host import execute, write_json
from agent.harness.installation import default_runtimes as discover_runtimes

ROOT = Path(__file__).resolve().parents[2]


def default_runtimes():
    """Compatibility entry point for existing research CLI commands."""
    return discover_runtimes(ROOT)


def credential_from_toml(path, model):
    # Explicit migration bridge. Never choose a model from this legacy runtime configuration.
    cfg = tomllib.loads(Path(path).read_text())
    provider = cfg["providers"][model.provider]
    if provider["base_url"].rstrip("/") != model.base_url.rstrip("/"):
        raise ValueError("Credential source endpoint differs from requested provider")
    return provider["api_key"]


async def main(args):
    model = ModelBinding(**json.loads(args.model.read_text()))
    key = (
        credential_from_toml(args.credential_toml, model)
        if args.credential_toml
        else os.environ.get(model.credential_env)
    )
    args.output.mkdir(parents=True, exist_ok=False)
    runtimes = default_runtimes()
    if args.runtimes:
        runtimes.update(json.loads(args.runtimes.read_text()))
    base = Profile(
        "science_agent",
        "kimi-code",
        model,
        Budget(wall_seconds=240, model_calls=4, tool_calls=2, output_tokens=2048, thinking="disabled"),
    )
    registry = builtin_registry()
    profiles = [replace(base, harness=name) for name in args.harnesses]
    report = {
        "kind": "runtime_conformance_probe",
        "performance_comparison": False,
        "preflight": [registry.preflight(p, runtimes) for p in profiles],
        "runs": [],
    }
    write_json(args.output / "report.json", report)
    if not args.execute:
        return
    for i, profile in enumerate(profiles):
        if not report["preflight"][i]["ready"]:
            continue
        folder = args.output / profile.harness
        folder.mkdir()
        work = folder / "workspace"
        work.mkdir()
        capability = folder / "capabilities"
        capability.mkdir()
        # One deterministic tool writes independently checkable evidence.
        (capability / "generated_tools.py").write_text(
            "from fastmcp import FastMCP\nfrom pathlib import Path\nimport json\n"
            'mcp = FastMCP("venusfactory")\n@mcp.tool()\ndef protein_probe(sequence: str) -> dict:\n'
            '    """Count residues in a supplied protein sequence; save the measured result."""\n'
            '    result = {"sequence": sequence, "length": len(sequence), "cysteines": sequence.count("C")}\n'
            '    Path("/workspace/probe.json").write_text(json.dumps(result))\n    return result\n'
            'if __name__ == "__main__": mcp.run()\n'
        )
        task = Task(
            work,
            capability,
            folder / "execution",
            "Call protein_probe once with sequence ACDEFGHIKC. Report its returned length and cysteine count.",
            "Complete this deterministic protein-tool integration check.",
            frozenset({"protein_probe"}),
            mcp_entry="generated_tools.py",
            phase="probe",
        )
        result = await execute(profile, task, runtimes, credential=key)
        artifact = json.loads((work / "probe.json").read_text()) if (work / "probe.json").exists() else None
        passed = (
            result["status"] == "completed"
            and result["audit"]["passed"]
            and artifact == {"sequence": "ACDEFGHIKC", "length": 10, "cysteines": 2}
        )
        report["runs"].append(
            {
                "harness": profile.harness,
                "status": result["status"],
                "artifact": artifact,
                "conformance_passed": passed,
                "wall_seconds": result.get("wall_seconds"),
            }
        )
        write_json(args.output / "report.json", report)
        print(json.dumps(report["runs"][-1]), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--runtimes", type=Path)
    parser.add_argument("--credential-toml", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--harnesses", nargs="+", default=["kimi-code", "deepseek-harness", "codex"])
    parser.add_argument("--execute", action="store_true")
    asyncio.run(main(parser.parse_args()))
