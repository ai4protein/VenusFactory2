"""Predeclared GLM four-role/native-executor conformance, not efficacy research."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

from run_harness_probe import ROOT, credential_from_toml, default_runtimes

from agent.harness import Budget, ModelBinding, Profile, Task, builtin_registry
from agent.harness.contracts import file_hash
from agent.harness.host import execute, write_json
from agent.harness.native_expert_team import NATIVE_EXPERT_HARNESSES, POLICY
from agent.rsi.source_snapshot import research_implementation_paths

TOOL = '''from fastmcp import FastMCP
from pathlib import Path
import json
mcp = FastMCP("venusfactory")
@mcp.tool()
def protein_probe(sequence: str) -> dict:
    """Measure sequence length and cysteine count and persist the actual result."""
    value = {"sequence": sequence, "length": len(sequence), "cysteines": sequence.count("C")}
    Path("/workspace/probe.json").write_text(json.dumps(value))
    return value
if __name__ == "__main__":
    mcp.run()
'''


async def run(args):
    model = ModelBinding(**json.loads(args.model.read_text()))
    credential = credential_from_toml(args.credential_toml, model)
    output = args.output.resolve()
    output.relative_to((ROOT / "research_runs/rsi").resolve())
    output.mkdir(parents=True, exist_ok=False)
    runtime = default_runtimes()
    profiles = [
        Profile(
            "science_expert",
            harness,
            model,
            Budget(
                wall_seconds=600,
                request_seconds=180,
                model_calls=12,
                tool_calls=2,
                output_tokens=2048,
                thinking="disabled",
            ),
        )
        for harness in NATIVE_EXPERT_HARNESSES
    ]
    protocol = {
        "kind": "native_expert_conformance.v1",
        "coordination_policy": POLICY,
        "profiles": [p.public() for p in profiles],
        "preflight": [builtin_registry().preflight(p, runtime) for p in profiles],
        "implementation": {str(p.relative_to(ROOT)): file_hash(p) for p in research_implementation_paths(ROOT)},
        "expected": {"sequence": "ACDEFGHIKC", "length": 10, "cysteines": 2},
        "required_roles": ["PI", "CB", "MLS", "SC"],
        "allocation_policy": "One fresh session per configured executor; all outcomes retained; no retries",
        "scope": "Engineering conformance only; no protein-model, biological or causal harness-performance claim",
    }
    if not all(p["ready"] for p in protocol["preflight"]):
        write_json(output / "protocol.json", protocol)
        raise ValueError("A declared native executor is unavailable")
    write_json(output / "protocol.json", protocol)
    report = {
        "status": "running",
        "protocol_sha256": file_hash(output / "protocol.json"),
        "started_unix": time.time(),
        "allocations": [],
    }
    write_json(output / "report.json", report)
    try:
        for profile in profiles:
            folder = output / profile.harness
            folder.mkdir()
            work, capability = folder / "workspace", folder / "capabilities"
            work.mkdir()
            capability.mkdir()
            (capability / "generated_tools.py").write_text(TOOL)
            write_json(
                capability / "scientific_task.json",
                {
                    "schema": 2,
                    "kind": "reproduction",
                    "context_tool": "protein_probe",
                    "artifact_file": "probe.json",
                    "required_tools": ["protein_probe"],
                    "completion_repair_limit": 0,
                },
            )
            row = {"harness": profile.harness, "status": "running"}
            report["allocations"].append(row)
            write_json(output / "report.json", report)
            task = Task(
                work,
                capability,
                folder / "execution",
                "Measure sequence ACDEFGHIKC with protein_probe exactly once and report its returned "
                "length and cysteine count. The four roles must plan, execute and review the observed "
                "result. This is a deterministic tool-contract check; do not claim biochemical activity.",
                "Use the supplied sequence and scientific tool. No human feedback is needed.",
                frozenset({"protein_probe"}),
                phase="chat",
            )
            result = await execute(profile, task, runtime, credential=credential)
            artifact = json.loads((work / "probe.json").read_text()) if (work / "probe.json").is_file() else None
            history = (
                json.loads((task.output / "role_history.json").read_text())
                if (task.output / "role_history.json").is_file()
                else []
            )
            row.update(
                status=result["status"],
                artifact=artifact,
                role_ids=result.get("role_ids"),
                calls=result["audit"]["requests"]["model_calls"],
                tools=result["audit"]["events"]["tool_calls"],
                conformance_passed=(
                    result["status"] == "completed"
                    and result["audit"]["passed"]
                    and result.get("four_role_path_completed") is True
                    and result.get("finalizer_succeeded") is True
                    and result.get("role_ids") == protocol["required_roles"]
                    and artifact == protocol["expected"]
                    and result["audit"]["events"]["tool_calls"] == 1
                    and len(history) == 4
                ),
            )
            write_json(output / "report.json", report)
            print(json.dumps(row), flush=True)
        report["status"] = (
            "completed" if all(r["conformance_passed"] for r in report["allocations"]) else "completed_with_failures"
        )
        for name, sha in protocol["implementation"].items():
            if file_hash(ROOT / name) != sha:
                raise ValueError("Conformance implementation changed")
    except BaseException as exc:
        report.update(status="failed", error_type=type(exc).__name__)
        raise
    finally:
        report["finished_unix"] = time.time()
        write_json(output / "report.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--credential-toml", type=Path, required=True)
    asyncio.run(run(parser.parse_args()))
