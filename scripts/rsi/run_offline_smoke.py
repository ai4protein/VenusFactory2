#!/usr/bin/env python3
"""Execute real MCP tools in isolation with a deterministic driver; no LLM calls."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).parent))
from fastmcp import Client
from fastmcp.client.transports import StdioTransport
from run_experiment import preflight, setup

from agent.rsi.pilot import build_tasks, evaluate
from agent.rsi.sandbox import bwrap_command
from agent.rsi.storage import atomic_json


async def run(config, output, capability_dir):
    results = []
    for task in build_tasks():
        started = time.monotonic()
        run_dir = output / "runs" / task["task_id"]
        workspace = run_dir / "workspace"
        workspace.mkdir(parents=True)
        private_home = run_dir / "private_home"
        private_home.mkdir()
        for name, text in task["files"].items():
            (workspace / name).write_text(text)
        cmd = bwrap_command(
            workspace, private_home, capability_dir, Path(config["kimi_bin"]), Path(config["python_root"]), 0
        )
        cmd = cmd[: cmd.index("--") + 1] + ["/opt/venus-runtime/bin/python", "-u", "/capabilities/pilot_tools.py"]
        events = []
        try:
            async with asyncio.timeout(60):
                with (run_dir / "mcp.log").open("w") as log:
                    transport = StdioTransport(command=cmd[0], args=cmd[1:], keep_alive=False, log_file=log)
                    async with Client(transport, timeout=20) as client:

                        async def call(name, _events=events, _client=client, **kwargs):
                            _events.append(
                                {"kind": "tool_call_start", "tool_name": "mcp__venusfactory__" + name, "args": kwargs}
                            )
                            value = await _client.call_tool(name, kwargs)
                            data = value.data
                            if data is None:
                                data = json.loads(next(c.text for c in value.content if hasattr(c, "text")))
                            _events.append({"kind": "tool_result", "tool_name": name, "output": data})
                            return data

                        kind = task["kind"]
                        if kind == "sequence":
                            actual = (await call("sequence_properties", fasta_file="input.fasta"))["properties"]
                        elif kind == "invalid":
                            actual = await call("sequence_properties", fasta_file="input.fasta")
                        elif kind == "temperature":
                            actual = await call("convert_temperature", value=310.15, source_unit="K", target_unit="C")
                        elif kind == "mutation":
                            sequence = "ACDEFGHIKLMNPQRSTVWY"
                            actual = {
                                "valid_mutation": await call("check_mutation", sequence=sequence, mutation="C2A"),
                                "invalid_mutation": await call("check_mutation", sequence=sequence, mutation="A2V"),
                            }
                        elif kind == "ids":
                            actual = await call(
                                "audit_prediction_table", input_csv="input.csv", prediction_csv="prediction.csv"
                            )
                        elif kind == "reuse":
                            await call("fit_linear_fixture", dataset_csv="train.csv", model_id="fixture")
                            registered = await call(
                                "validate_linear_fixture", model_id="fixture", validation_csv="validation.csv"
                            )
                            prediction = await call(
                                "predict_linear_fixture", model_id="fixture", input_csv="test_inputs.csv"
                            )
                            wrong = await call(
                                "predict_linear_fixture",
                                model_id="fixture",
                                input_csv="test_inputs.csv",
                                task="unrelated_protein_property",
                            )
                            actual = {
                                "registered": registered["registered"],
                                "predictions": prediction["predictions"],
                                "incompatible_reuse_rejected": not wrong["valid"],
                            }
                        atomic_json(workspace / "output.json", actual)
            run_result = {"status": "completed", "driver": "deterministic_mcp", "model_calls": 0}
        except Exception as exc:
            run_result = {
                "status": "runtime_error",
                "driver": "deterministic_mcp",
                "model_calls": 0,
                "error": f"{type(exc).__name__}: {exc}",
            }
        (run_dir / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
        atomic_json(run_dir / "run.json", run_result)
        result = evaluate(task, workspace, run_result, run_dir / "events.jsonl")
        result.update(driver="deterministic_mcp", wall_seconds=round(time.monotonic() - started, 3))
        atomic_json(run_dir / "evaluation.json", result)
        results.append(result)
        print(json.dumps(result), flush=True)
    summary = {
        "state": "completed",
        "driver": "deterministic_mcp",
        "model_calls": 0,
        "tasks": 6,
        "passed": sum(x["passed"] for x in results),
        "results": results,
        "claim_scope": "Tool wiring and scientific artifact validation only; no autonomous agent or RSI performance claim",
    }
    atomic_json(output / "summary.json", summary)
    atomic_json(output / "status.json", {k: summary[k] for k in ["state", "driver", "model_calls", "tasks", "passed"]})
    return 0 if summary["passed"] == 6 else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.relative_to((REPO / "research_runs/rsi").resolve())
    config = json.loads((REPO / "configs/rsi/engineering_smoke.json").read_text())
    config.update(suite="offline_mcp_smoke", claim_scope="Deterministic tool integration; no LLM calls")
    baseline, capabilities, version = setup(config, output)
    preflight(config, output, capabilities)
    print(f"RUN_DIR={output}", flush=True)
    return asyncio.run(run(config, output, capabilities))


if __name__ == "__main__":
    raise SystemExit(main())
