#!/usr/bin/env python3
"""One GLM-proposed policy vs its parent on six new engineering variants.

This is a diagnostic, not recursive self-improvement or a formal holdout study.
Proposal sees only the old development partition, never evaluator expectations.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import shutil
import sys
import tomllib
from pathlib import Path

import httpx
from run_experiment import REPO, preflight, setup

sys.path.insert(0, str(REPO / "src"))
from agent.rsi.execution import execute_task
from agent.rsi.model_policy import read_model_configuration
from agent.rsi.pilot import build_tasks
from agent.rsi.storage import atomic_json, sha256_file
from agent.rsi.versions import VersionStore


def shape(value):
    if isinstance(value, dict):
        return {k: shape(v) for k, v in value.items()}
    if isinstance(value, list):
        return {"type": "array", "example_element_type": shape(value[0]) if value else None}
    return type(value).__name__


def propose(config, source, output, baseline):
    tasks = json.loads((source / "tasks_private.json").read_text())
    evidence = []
    for task in tasks:
        if task["split"] != "development":
            continue
        score = json.loads((source / "runs" / task["task_id"] / "evaluation.json").read_text())
        run = json.loads((source / "runs" / task["task_id"] / "run.json").read_text())
        evidence.append(
            {
                "task_kind": task["kind"],
                "request": task["prompt"],
                "passed": score["passed"],
                "reason": score["reason"],
                "runtime_status": run["status"],
                "error_code": run.get("error_code"),
                "output_shape": shape(score["actual"]) if "actual" in score else None,
            }
        )
    request = {
        "instruction": "Propose ONE reusable executor-policy addition for a tool-using protein assistant. "
        "Use only the supplied development evidence. Output JSON with exactly hypothesis (string), "
        "executor_policy (string, at most 350 words), expected_failure_modes (list of strings). "
        "No specific task IDs, sample answers, calculated numbers, hidden scoring logic, new tools, "
        "shell commands, or network access. Keep the existing seven tools and native file tools. "
        "Focus on a general workflow that preserves tool-result structure and verifies saved artifacts. "
        "Do not claim to fix provider errors. This is a single candidate, no optimization loop.",
        "development_evidence": evidence,
    }
    atomic_json(output / "proposal_input.json", request)
    config_text, identity = read_model_configuration(
        Path(config["kimi_config"]), expected=baseline["model"], required=config.get("required_model")
    )
    private = tomllib.loads(config_text)
    atomic_json(output / "proposer_model_binding.json", identity)
    selected = private["models"][private["default_model"]]
    provider = private["providers"][selected["provider"]]
    if selected["model"] != "glm-4-flash":
        raise ValueError("This experiment requires GLM")
    with httpx.Client(timeout=90) as client:
        response = client.post(
            provider["base_url"].rstrip("/") + "/chat/completions",
            headers={"Authorization": "Bearer " + provider["api_key"]},
            json={
                "model": selected["model"],
                "messages": [{"role": "user", "content": json.dumps(request)}],
                "temperature": 0.2,
                "max_tokens": 1800,
            },
        )
    response.raise_for_status()
    data = response.json()
    raw = data["choices"][0]["message"].get("content") or ""
    atomic_json(
        output / "proposal_response.json",
        {
            "content": raw,
            "usage": data.get("usage"),
            "model": data.get("model"),
            "finish_reason": data["choices"][0].get("finish_reason"),
            "requests": 1,
        },
    )
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    result = json.loads(text)
    if set(result) != {"hypothesis", "executor_policy", "expected_failure_modes"}:
        raise ValueError("invalid proposal fields")
    if not isinstance(result["hypothesis"], str) or not result["hypothesis"].strip():
        raise ValueError("missing hypothesis")
    policy = result["executor_policy"]
    if not isinstance(policy, str) or not 20 <= len(policy) <= 5000 or len(policy.split()) > 350:
        raise ValueError("invalid policy size")
    if not isinstance(result["expected_failure_modes"], list) or not all(
        isinstance(x, str) for x in result["expected_failure_modes"]
    ):
        raise ValueError("invalid failure modes")
    atomic_json(output / "proposal.json", result)
    return result


async def compare(config, output, baseline, bundles):
    # Variant 5 was not in the previous 0..4 pilot. Same families: diagnostic only.
    tasks = [dict(t, split="diagnostic") for t in build_tasks(6) if t["variant"] == 5]
    atomic_json(output / "tasks_private.json", tasks)
    order = []
    rng = random.Random(20261010)
    for task in tasks:
        arms = list(bundles)
        rng.shuffle(arms)
        order.extend((arm, task) for arm in arms)
    atomic_json(
        output / "comparison_plan.json",
        {
            "seed": 20261010,
            "pairs": 6,
            "calls_per_arm": 6,
            "concurrency": 2,
            "order": [[a, t["task_id"]] for a, t in order],
            "task_timeout_s": config["task_timeout_s"],
            "max_tool_calls": config["max_tool_calls"],
            "transport_retries": 0,
            "manual_proposal_edits": 0,
            "automatic_promotion": False,
            "claim_scope": "One proposal, six same-family diagnostic pairs; no RSI efficacy inference.",
        },
    )
    semaphore = asyncio.Semaphore(2)
    results = []

    async def execute(arm, task):
        async with semaphore:
            capdir, version = bundles[arm]
            run_dir = output / "runs" / arm / task["task_id"]
            _, score = await execute_task(
                task,
                config=config,
                baseline=baseline,
                capability_dir=capdir,
                version_id=version["version_id"],
                run_dir=run_dir,
                metadata={"arm": arm},
            )
            score["arm"] = arm
            atomic_json(run_dir / "evaluation.json", score)
            results.append(score)
            atomic_json(output / "status.json", {"state": "running", "completed": len(results), "total": 12})
            print(json.dumps({k: score[k] for k in ("arm", "task_id", "passed", "reason", "error_code")}), flush=True)

    await asyncio.gather(*(execute(a, t) for a, t in order))
    pairs = []
    for task in tasks:
        pair = {a: next(s for s in results if s["arm"] == a and s["task_id"] == task["task_id"]) for a in bundles}
        pairs.append(
            {
                "task_id": task["task_id"],
                "kind": task["kind"],
                "parent_passed": pair["parent"]["passed"],
                "candidate_passed": pair["candidate"]["passed"],
            }
        )
    summary = {
        "state": "completed",
        "model": baseline["model"],
        "pairs": pairs,
        "by_arm": {
            a: {
                "passed": sum(s["passed"] for s in results if s["arm"] == a),
                "tasks": 6,
                "runtime_completed": sum(s["runtime_status"] == "completed" for s in results if s["arm"] == a),
            }
            for a in bundles
        },
        "all_sandboxes_stopped": all(s["sandbox_stopped"] for s in results),
        "automatic_promotion": False,
        "usage_complete": False,
        "claim_scope": "Diagnostic only. GLM generated one policy; U did not evolve. No significance claim.",
        "results": results,
    }
    atomic_json(output / "summary.json", summary)
    atomic_json(output / "status.json", {"state": "completed", "completed": 12, "by_arm": summary["by_arm"]})
    print(json.dumps({k: summary[k] for k in ("state", "by_arm", "all_sandboxes_stopped")}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.relative_to((REPO / "research_runs/rsi").resolve())
    config = json.loads((REPO / "configs/rsi/engineering_smoke_glm.json").read_text())
    config.update(
        suite="glm_policy_probe", concurrency=2, claim_scope="One GLM proposal and six diagnostic pairs, no recursive U"
    )
    baseline, capdir, parent = setup(config, output)
    print(f"RUN_DIR={output}", flush=True)
    try:
        preflight(config, output, capdir)
        proposal = propose(config, REPO / "research_runs/rsi/20261010_glm_autonomous_pilot30_v1", output, baseline)
        source = output / "candidate_source"
        source.mkdir()
        for path in capdir.iterdir():
            if path.name != "manifest.json":
                shutil.copyfile(path, source / path.name)
        with (source / "executor_prompt.md").open("a") as stream:
            stream.write("\nGLM-proposed diagnostic policy:\n" + proposal["executor_policy"] + "\n")
        store = VersionStore(output / "version_store")
        candidate = store.snapshot(
            source,
            parent=parent["version_id"],
            metadata={
                "scope": "diagnostic",
                "proposer_model": "glm-4-flash",
                "manual_edits": 0,
                "proposal_sha256": sha256_file(output / "proposal.json"),
            },
        )
        asyncio.run(
            compare(
                config,
                output,
                baseline,
                {
                    "parent": (capdir, parent),
                    "candidate": (store.root / "versions" / candidate["version_id"], candidate),
                },
            )
        )
    except BaseException as exc:
        atomic_json(output / "status.json", {"state": "failed", "error": f"{type(exc).__name__}: {exc}"})
        raise


if __name__ == "__main__":
    main()
