#!/usr/bin/env python3
"""Launch bounded Kimi engineering tasks; all outputs are local to this run.

Not a formal B1/B4 comparison. The immutable capability bundle contains only
public tools. Task expectations and scoring code stay outside its namespace.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from agent.rsi.engine.baseline import capture_baseline
from agent.rsi.engine.execution import execute_task
from agent.rsi.engine.model_policy import read_model_configuration
from agent.rsi.campaigns.pilot import build_tasks
from agent.rsi.engine.sandbox import bwrap_command
from agent.rsi.engine.storage import atomic_json, sha256_file
from agent.rsi.engine.versions import VersionStore

SYSTEM_PROMPT = """You are the VenusFactory2 protein-engineering assistant running a bounded engineering pilot.
Use only tools actually present in your catalog. The private venusfactory MCP provides
sequence_properties, audit_prediction_table, convert_temperature, check_mutation,
fit_linear_fixture, validate_linear_fixture and predict_linear_fixture.
This is a deliberately small CPU fixture tool surface, not the full deployed VF system.
All task files are under /workspace. Use the native Write tool to save /workspace/output.json.
For Write use the exact argument names in its schema. Output plain JSON in that file.
Compute the requested quantities using the specified tools. Do not fabricate execution.
Keep tool results and sample IDs intact. Do not access runtime configuration or credentials.
Do not use Bash, network tools, subagents, interactive questions, or modify capability code.
No approval is needed for the task itself. Complete it autonomously from the supplied inputs.
"""


def setup(config: dict, output: Path) -> tuple[dict, Path, dict]:
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "config.json", config)
    baseline = capture_baseline(REPO, Path(config["kimi_bin"]), Path(config["kimi_config"]), output / "baseline.json")
    read_model_configuration(
        Path(config["kimi_config"]), expected=baseline["model"], required=config.get("required_model")
    )
    baseline["runtime_overrides"] = {
        "max_output_size": 4096,
        "thinking": "medium",
        "tool_policy": "pilot_mcp_and_workspace_files",
    }
    baseline["evaluator_sha256"] = sha256_file(REPO / "src/agent/rsi/engine/diagnostic_audit.py")
    baseline["legacy_evaluator_sha256"] = sha256_file(REPO / "src/agent/rsi/campaigns/pilot.py")
    atomic_json(output / "baseline.json", baseline)
    source = output / "initial_capabilities"
    source.mkdir()
    shutil.copyfile(REPO / "src/agent/rsi/engine/pilot_tools.py", source / "pilot_tools.py")
    shutil.copyfile(REPO / "mcp_tools/tools/predict/features/calculate_physchem.py", source / "calculate_physchem.py")
    (source / "executor_prompt.md").write_text(SYSTEM_PROMPT)
    store = VersionStore(output / "version_store")
    version = store.snapshot(source, metadata={"scope": "engineering", "base_commit": baseline["git_commit"]})
    capability_dir = store.root / "versions" / version["version_id"]
    atomic_json(output / "status.json", {"state": "preflight", "completed": 0})
    return baseline, capability_dir, version


def preflight(config: dict, output: Path, capability_dir: Path) -> dict:
    work = output / "preflight_workspace"
    private_home = output / "preflight_private_home"
    work.mkdir()
    private_home.mkdir()
    hidden = output / "control_canary.txt"
    hidden.write_text("must not be visible inside the agent namespace")
    (work / "visible.txt").write_text("visible")
    command = bwrap_command(
        work, private_home, capability_dir, Path(config["kimi_bin"]), Path(config["python_root"]), 0
    )
    command = command[: command.index("--") + 1]
    code = (
        "from pathlib import Path; import json; "
        "assert Path('/workspace/visible.txt').read_text()=='visible'; "
        f"assert not Path({str(hidden)!r}).exists(); "
        f"assert not Path({str(REPO)!r}).exists(); "
        "import Bio, numpy, fastmcp; print(json.dumps({'filesystem_isolation':True,'runtime_imports':True}))"
    )
    check = subprocess.run(
        command + ["/opt/venus-runtime/bin/python", "-c", code], capture_output=True, text=True, timeout=40
    )
    result = {
        "returncode": check.returncode,
        "passed": check.returncode == 0,
        "stdout": check.stdout[-3000:],
        "stderr": check.stderr[-3000:],
        "network_isolated": False,
        "mounted_repository": False,
    }
    atomic_json(output / "preflight.json", result)
    if not result["passed"]:
        raise RuntimeError("isolation/import preflight failed; see preflight.json")
    return result


async def experiment(
    config: dict, output: Path, baseline: dict, capability_dir: Path, version: dict, limit: int | None
):
    tasks = build_tasks(config["variants"])
    if limit is not None:
        tasks = tasks[:limit]
    atomic_json(output / "tasks_private.json", tasks)
    semaphore = asyncio.Semaphore(config["concurrency"])
    results = []
    blocked = asyncio.Event()
    skipped = []

    async def execute(task):
        async with semaphore:
            if blocked.is_set():
                skipped.append({"task_id": task["task_id"], "reason": "provider_authentication_failed"})
                return
            run_dir = output / "runs" / task["task_id"]
            run, score = await execute_task(
                task,
                config=config,
                baseline=baseline,
                capability_dir=capability_dir,
                version_id=version["version_id"],
                run_dir=run_dir,
            )
            if run.get("error_code") == "provider_authentication_failed":
                blocked.set()
            results.append(score)
            atomic_json(
                output / "status.json",
                {
                    "state": "running",
                    "completed": len(results),
                    "total": len(tasks),
                    "passed": sum(x["passed"] for x in results),
                    "last_task": task["task_id"],
                },
            )
            print(
                json.dumps(
                    {
                        "task": task["task_id"],
                        "status": run["status"],
                        "passed": score["passed"],
                        "reason": score["reason"],
                    }
                ),
                flush=True,
            )

    await asyncio.gather(*(execute(task) for task in tasks))
    summary = {
        "state": "blocked" if blocked.is_set() else "completed",
        "block_reason": "provider_authentication_failed" if blocked.is_set() else None,
        "attempted": len(results),
        "skipped": skipped,
        "suite": config["suite"],
        "claim_scope": config["claim_scope"],
        "tasks": len(tasks),
        "passed": sum(x["passed"] for x in results),
        "failed": sum(not x["passed"] for x in results),
        "model": baseline["model"],
        "kimi_version": baseline["kimi_version"],
        "version_id": version["version_id"],
        "usage_complete": False,
        "results": sorted(results, key=lambda x: x["task_id"]),
    }
    atomic_json(output / "summary.json", summary)
    atomic_json(
        output / "status.json",
        {k: summary[k] for k in ("state", "tasks", "attempted", "passed", "failed", "block_reason")},
    )
    return 0 if summary["failed"] == 0 and not blocked.is_set() else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=REPO / "configs/rsi/engineering_smoke_glm.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if not 1 <= config["variants"] <= 5 or not 1 <= config["concurrency"] <= 2:
        raise ValueError("engineering pilot bounds: variants <=5, concurrency <=2")
    if not 0 < config["task_timeout_s"] <= 300 or not 0 < config["max_tool_calls"] <= 20:
        raise ValueError("task budget outside engineering pilot bounds")
    if args.limit is not None and args.limit < 1:
        raise ValueError("limit must be positive")
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S")
    output = (args.output or REPO / "research_runs/rsi" / f"{stamp}_{config['suite']}").resolve()
    output.relative_to((REPO / "research_runs/rsi").resolve())
    baseline, capability_dir, version = setup(config, output)
    print(f"RUN_DIR={output}", flush=True)
    try:
        preflight(config, output, capability_dir)
        if args.preflight_only:
            atomic_json(output / "status.json", {"state": "preflight_passed"})
            return 0
        return asyncio.run(experiment(config, output, baseline, capability_dir, version, args.limit))
    except BaseException as exc:
        atomic_json(output / "status.json", {"state": "failed", "error": f"{type(exc).__name__}: {exc}"})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
