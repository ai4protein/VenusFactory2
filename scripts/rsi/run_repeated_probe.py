#!/usr/bin/env python3
"""72 bounded GLM engineering runs: 3 fixed policies x 12 tasks x 2 repeats.

No new proposal is optimized. No candidate is promoted. Tasks are synthetic
same-family diagnostics, not a hidden protein benchmark or an RSI evaluation.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import shutil
from collections import Counter
from pathlib import Path

from run_experiment import REPO, preflight, setup

from agent.rsi.execution import execute_task
from agent.rsi.pilot import build_tasks
from agent.rsi.storage import atomic_json, sha256_file
from agent.rsi.versions import VersionStore

CONTRACT_POLICY = """Result delivery procedure (human-authored engineering control):
Before each scientific tool call, preserve the exact task input, sequence, sample IDs and units.
Use successful tool results as evidence. A valid=false domain result can be the expected rejection;
it is not permission to omit the result. Keep complete result objects, all required keys,
nested objects, empty lists, booleans, and numeric types. Do not replace objects with summaries.
When the task requests named results, place each complete tool result under its requested key.
Always call the native Write tool to create /workspace/output.json containing plain JSON.
Read the saved file with Read, compare it with the task's requested structure and tool results,
and repair any missing or incorrectly typed fields using available file tools within the budget.
Do not claim a file was saved without a successful write. Do not ask for human intervention.
Do not fabricate tool results or missing predictions, change tool code, or use unavailable tools.
"""


def bundle(store, output, capdir, parent, arm, addition, origin):
    source = output / (arm + "_source")
    source.mkdir()
    for path in capdir.iterdir():
        if path.name != "manifest.json":
            shutil.copyfile(path, source / path.name)
    with (source / "executor_prompt.md").open("a") as stream:
        stream.write("\n" + addition + "\n")
    version = store.snapshot(
        source, parent=parent["version_id"], metadata={"scope": "diagnostic", "arm": arm, "origin": origin}
    )
    return store.root / "versions" / version["version_id"], version


async def experiment(config, output, baseline, bundles):
    tasks = [dict(t, split="diagnostic") for t in build_tasks(8) if t["variant"] in (6, 7)]
    atomic_json(output / "tasks_private.json", tasks)
    rng = random.Random(20261010)
    blocks = [(repeat, task) for repeat in range(2) for task in tasks]
    rng.shuffle(blocks)
    order = []
    for repeat, task in blocks:
        arms = list(bundles)
        rng.shuffle(arms)
        order.extend((arm, repeat, task) for arm in arms)
    atomic_json(
        output / "comparison_plan.json",
        {
            "runs": 72,
            "arms": list(bundles),
            "tasks_per_arm": 12,
            "repeats": 2,
            "variants": [6, 7],
            "task_families": 6,
            "order_seed": 20261010,
            "order": [[a, rep, t["task_id"]] for a, rep, t in order],
            "concurrency": 2,
            "timeout_s": 180,
            "max_tool_calls": 12,
            "transport_retries": 0,
            "automatic_promotion": False,
            "candidate_regeneration": False,
            "primary": "external audit: matching input and successful results + correct artifact + completed runtime",
            "secondary": [
                "legacy_score",
                "tool_chain_correct",
                "artifact_schema_valid",
                "artifact_values_correct",
                "provider_failure",
                "delivery_gap",
            ],
            "claim_scope": "Same-family synthetic diagnostic; repeats are not independent lineages or biological evidence.",
        },
    )
    semaphore = asyncio.Semaphore(2)
    results, skipped = [], []
    blocked = asyncio.Event()

    async def execute(arm, repeat, task):
        async with semaphore:
            if blocked.is_set():
                skipped.append({"arm": arm, "repeat": repeat, "task_id": task["task_id"]})
                return
            capdir, version = bundles[arm]
            run_dir = output / "runs" / arm / (task["task_id"] + f"-r{repeat}")
            run, score = await execute_task(
                task,
                config=config,
                baseline=baseline,
                capability_dir=capdir,
                version_id=version["version_id"],
                run_dir=run_dir,
                metadata={"arm": arm, "repeat": repeat},
            )
            if run.get("error_code") == "provider_authentication_failed":
                blocked.set()
            score.update(arm=arm, repeat=repeat)
            atomic_json(run_dir / "evaluation.json", score)
            results.append(score)
            atomic_json(
                output / "status.json",
                {
                    "state": "running",
                    "completed": len(results),
                    "total": 72,
                    "by_arm": {
                        a: {
                            "attempted": sum(x["arm"] == a for x in results),
                            "passed": sum(x["passed"] for x in results if x["arm"] == a),
                        }
                        for a in bundles
                    },
                },
            )
            print(
                json.dumps(
                    {
                        "n": len(results),
                        "arm": arm,
                        "task": task["task_id"],
                        "repeat": repeat,
                        "passed": score["passed"],
                        "tool_chain": score["tool_chain_correct"],
                        "artifact": score["artifact_values_correct"],
                        "error": score["error_code"],
                    }
                ),
                flush=True,
            )

    await asyncio.gather(*(execute(a, rep, t) for a, rep, t in order))
    metrics = [
        "passed",
        "legacy_passed",
        "tool_chain_correct",
        "artifact_exists",
        "artifact_schema_valid",
        "artifact_values_correct",
        "runtime_completed",
        "computation_delivery_gap",
    ]
    summary = {
        "state": "blocked" if blocked.is_set() else "completed",
        "planned": 72,
        "attempted": len(results),
        "skipped": skipped,
        "model": baseline["model"],
        "all_sandboxes_stopped": all(x["sandbox_stopped"] for x in results),
        "automatic_promotion": False,
        "usage_complete": False,
        "by_arm": {
            a: {
                "planned": 24,
                "attempted": sum(x["arm"] == a for x in results),
                **{m: sum(x[m] for x in results if x["arm"] == a) for m in metrics},
                "errors": dict(Counter(x["error_code"] or x["runtime_status"] for x in results if x["arm"] == a)),
            }
            for a in bundles
        },
        "results": results,
        "claim_scope": "72 bounded synthetic runs; no autonomous recursive update, plugin implementation or biological efficacy claim.",
    }
    atomic_json(output / "summary.json", summary)
    atomic_json(output / "status.json", {k: summary[k] for k in ("state", "planned", "attempted", "by_arm")})
    print(json.dumps({k: summary[k] for k in ("state", "by_arm", "all_sandboxes_stopped")}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.relative_to((REPO / "research_runs/rsi").resolve())
    config = json.loads((REPO / "configs/rsi/engineering_smoke_glm.json").read_text())
    config.update(suite="glm_repeated_diagnostic", concurrency=2, claim_scope="72 synthetic diagnostic runs")
    baseline, capdir, parent = setup(config, output)
    assert baseline["model"]["model"] == "glm-4-flash"
    baseline["diagnostic_audit_sha256"] = sha256_file(REPO / "src/agent/rsi/diagnostic_audit.py")
    atomic_json(output / "baseline.json", baseline)
    print(f"RUN_DIR={output}", flush=True)
    try:
        preflight(config, output, capdir)
        previous = REPO / "research_runs/rsi/20261010_glm_policy_probe_v1/proposal.json"
        proposal = json.loads(previous.read_text())
        atomic_json(
            output / "fixed_policies.json",
            {
                "glm_candidate": proposal,
                "source_proposal_sha256": sha256_file(previous),
                "contract_control": CONTRACT_POLICY,
                "origins": {
                    "parent": "existing pilot",
                    "glm_candidate": "previous single GLM proposal, unchanged",
                    "contract_control": "human-authored engineering control; not RSI",
                },
            },
        )
        store = VersionStore(output / "version_store")
        bundles = {
            "parent": (capdir, parent),
            "glm_candidate": bundle(
                store, output, capdir, parent, "glm_candidate", proposal["executor_policy"], "prior GLM proposal"
            ),
            "contract_control": bundle(
                store, output, capdir, parent, "contract_control", CONTRACT_POLICY, "human-authored"
            ),
        }
        asyncio.run(experiment(config, output, baseline, bundles))
    except BaseException as exc:
        atomic_json(output / "status.json", {"state": "failed", "error": f"{type(exc).__name__}: {exc}"})
        raise


if __name__ == "__main__":
    main()
