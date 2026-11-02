"""Predeclared conversational conformance matrix, never a scientific efficacy study."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from run_harness_probe import default_runtimes

from agent.harness import Budget, ModelBinding, Profile, Task, builtin_registry
from agent.harness.catalog import resolve_credential
from agent.harness.contracts import file_hash
from agent.harness.host import execute, write_json

ROOT = Path(__file__).resolve().parents[2]
CASES = {
    "direct_answer": {
        "prompt": 'What one-letter code denotes cysteine? Answer only JSON: {"residue": "the code"}.',
        "tools": [],
        "answer": {"residue": "C"},
        "tool_calls": 0,
    },
    "clarify_missing_sequence": {
        "prompt": "Count the cysteines in my protein. I have not supplied its sequence. "
        'If the sequence is missing, ask for it using JSON {"needs_input":"protein_sequence"}; '
        "otherwise return JSON with length and cysteines. Never invent a sequence.",
        "tools": ["protein_probe"],
        "answer": {"needs_input": "protein_sequence"},
        "tool_calls": 0,
    },
    "measure_sequence": {
        "prompt": "Measure the sequence length and cysteine count for ACDEFGHIKC using the available "
        "measurement capability. Return only JSON with the measured length and cysteines.",
        "tools": ["protein_probe"],
        "answer": {"length": 10, "cysteines": 2},
        "tool_calls": 1,
    },
}
TOOL_SOURCE = '''from fastmcp import FastMCP
from pathlib import Path
import json
mcp = FastMCP("venusfactory")
@mcp.tool()
def protein_probe(sequence: str) -> dict:
    """Measure residue length and cysteine count in a supplied sequence."""
    result = {"sequence": sequence, "length": len(sequence), "cysteines": sequence.count("C")}
    Path("/workspace/probe.json").write_text(json.dumps(result))
    return result
if __name__ == "__main__": mcp.run()
'''


def response_json(result, output):
    value = result.get("text")
    report = output / "final_report.json"
    if value is None and report.exists():
        value = json.loads(report.read_text())["content"]
    if isinstance(value, list):
        value = "".join(x.get("text", "") for x in value if isinstance(x, dict) and x.get("type") == "text")
    text = str(value or "").strip()
    if text.startswith("```") and text.endswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        return None


def assess(result, folder, case):
    artifact_path = folder / "workspace/probe.json"
    artifact = json.loads(artifact_path.read_text()) if artifact_path.exists() else None
    answer = response_json(result, folder / "execution")
    expected_artifact = {"sequence": "ACDEFGHIKC", "length": 10, "cysteines": 2} if case["tool_calls"] else None
    checks = {
        "completed": result["status"] == "completed",
        "execution_audit": result["audit"]["passed"],
        "answer": answer == case["answer"],
        "tool_count": result["audit"]["events"]["tool_calls"] == case["tool_calls"],
        "measured_artifact": artifact == expected_artifact,
    }
    return {"passed": all(checks.values()), "checks": checks, "answer": answer, "artifact": artifact}


async def main(args):
    model_bindings = [ModelBinding(**json.loads(path.read_text())) for path in args.models]
    if len({(m.provider, m.model) for m in model_bindings}) != len(model_bindings):
        raise ValueError("Repeated model identity")
    runtimes = default_runtimes()
    registry = builtin_registry()
    profiles = [
        Profile(
            "science_agent",
            h,
            m,
            Budget(
                wall_seconds=240,
                request_seconds=120,
                model_calls=4,
                tool_calls=2,
                output_tokens=2048,
                thinking="disabled",
            ),
        )
        for m in model_bindings
        for h in args.harnesses
    ]
    args.output.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__), ROOT / "scripts/rsi/run_harness_probe.py"]
    sources += sorted((ROOT / "src/agent/harness").glob("*.py"))
    sources += [
        ROOT / ("src/agent/" + n)
        for n in [
            "kimi_client.py",
            "rsi/engine/runtime.py",
            "rsi/engine/sandbox.py",
            "rsi/engine/scientist_runtime.py",
            "rsi/engine/adaptive_runtime.py",
            "rsi/engine/langgraph_tools.py",
            "rsi/engine/model_gateway.py",
            "rsi/engine/model_policy.py",
        "rsi/engine/tool_contracts.py",
        ]
    ]
    hashes = {str(p.relative_to(ROOT)): file_hash(p) for p in sources}
    for name in hashes:
        dst = args.output / "implementation" / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes((ROOT / name).read_bytes())
    allocations = [
        {
            "id": f"m{mi:02d}-{h}-{case}",
            "profile": p.public(),
            "case": case,
            "preflight": registry.preflight(p, runtimes),
        }
        for mi, m in enumerate(model_bindings)
        for h in args.harnesses
        for p in profiles
        if p.model == m and p.harness == h
        for case in CASES
    ]
    protocol = {
        "kind": "conversational_conformance",
        "scientific_performance_comparison": False,
        "cases": CASES,
        "allocations": allocations,
        "implementation_sha256": hashes,
        "scope": "one allocation per model/harness/scenario; no replicated efficacy inference",
        "interaction": "independent chat turns, not multi-turn session migration",
        "native_system_prompts": "backend-specific and retained; not identical across harnesses",
    }
    write_json(args.output / "protocol.json", protocol)
    report = {"status": "preflight", "runs": []}
    write_json(args.output / "report.json", report)
    if not args.execute:
        return
    # Credentials are resolved once, but never written into any protocol or child profile.
    keys = {(m.provider, m.model): resolve_credential(m, args.credential_toml) for m in model_bindings}
    report["status"] = "running"
    write_json(args.output / "report.json", report)
    for item in allocations:
        if any(file_hash(ROOT / name) != sha for name, sha in hashes.items()):
            raise RuntimeError("Frozen implementation changed")
        if not item["preflight"]["ready"]:
            report["runs"].append({"id": item["id"], "status": "unsupported", "passed": False})
            write_json(args.output / "report.json", report)
            continue
        p = Profile.from_dict(item["profile"])
        case = CASES[item["case"]]
        folder = args.output / item["id"]
        workspace, capability = folder / "workspace", folder / "capabilities"
        workspace.mkdir(parents=True)
        capability.mkdir()
        (capability / "generated_tools.py").write_text(TOOL_SOURCE)
        task = Task(
            workspace,
            capability,
            folder / "execution",
            case["prompt"],
            "You are an interactive protein-research assistant.",
            frozenset(case["tools"]),
            phase="chat",
        )
        result = await execute(p, task, runtimes, credential=keys[(p.model.provider, p.model.model)])
        assessed = assess(result, folder, case)
        row = {
            "id": item["id"],
            "model": p.model.model,
            "harness": p.harness,
            "case": item["case"],
            "status": result["status"],
            "wall_seconds": result["host_wall_seconds"],
            "model_calls": result["audit"]["requests"]["model_calls"],
            "tool_calls": result["audit"]["events"]["tool_calls"],
            **assessed,
        }
        report["runs"].append(row)
        write_json(args.output / "report.json", report)
        print(json.dumps(row), flush=True)
    report["source_intact"] = all(file_hash(ROOT / name) == sha for name, sha in hashes.items())
    report["passed"] = report["source_intact"] and all(x["passed"] for x in report["runs"])
    report["status"] = "completed"
    write_json(args.output / "report.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=Path, nargs="+", required=True)
    parser.add_argument("--harnesses", nargs="+", default=["kimi-code", "deepseek-harness", "langgraph"])
    parser.add_argument("--credential-toml", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    asyncio.run(main(parser.parse_args()))
