"""Matched complete-objective controls: expert goals, single-agent persistence,
and conversational Agent under a predeclared scripted user-turn policy.

Cold-start repeats on reused protein partitions estimate engineering variance;
these are not independent biological lineages or confirmatory RSI evidence.
"""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import json
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from run_harness_probe import ROOT, credential_from_toml, default_runtimes
from run_science_goal import GeneratedModelGoal, prepare, read

from agent.harness import Profile, builtin_registry
from agent.harness.contracts import digest, file_hash
from agent.harness.host import write_json
from agent.rsi.engine.model_plugins import get_model_plugin
from agent.science_goal import GoalStore, run_goal

CONDITIONS = ("science_goal", "persistent_agent_ablation", "scripted_agent_turns")
MODES = {c: "science_expert" if c == "science_goal" else "science_agent" for c in CONDITIONS}
CASES = ("alde_parpgb", "mcba_itopride")
BUDGET = {"max_cycles": 4, "model_calls": 72, "tool_calls": 36, "wall_seconds": 2400}
OBJECTIVE = (
    "Develop deterministic protein prediction model code for the supplied assay, "
    "independently verify its training and validation, then reuse the accepted "
    "model in a fresh execution session to predict every unseen input. "
    "Both independent model acceptance and verified fresh-session reuse are required."
)
TURN_POLICY = {
    "id": "verified_stage_turns.v1",
    "initial": "Supply the complete objective, declared criteria and current-stage task.",
    "after_rejection": "Supply the same verified rejection summary and request another development attempt within remaining budget.",
    "after_acceptance": "Request fresh-session reuse of the accepted model on supplied unlabelled inputs.",
    "after_failed_reuse": "Request another reuse attempt with the same information and remaining budget.",
    "future_feedback": "Never provide final labels, scores or baseline results to any executor.",
    "external_turns": "Every episode in conversational control; only initial episode in persistent conditions.",
    "internal_expert_gates": "Fixed scripted task-context and plan confirmations, recorded separately from external turns.",
    "human_interventions": 0,
    "interpretation": "Scripted assistance policy, not a real human interaction study or automatic product mode switching.",
}


def schedule(cases, repeats):
    rows = []
    for repeat in range(repeats):
        order = tuple(cases) if repeat % 2 == 0 else tuple(reversed(cases))
        for case in order:
            offset = (repeat + tuple(cases).index(case)) % len(CONDITIONS)
            for condition in CONDITIONS[offset:] + CONDITIONS[:offset]:
                rows.append(
                    {"id": f"{case}_r{repeat:02d}_{condition}", "case": case, "repeat": repeat, "condition": condition}
                )
    return rows


def implementation():
    paths = [
        Path(__file__),
        ROOT / "scripts/rsi/run_science_goal.py",
        ROOT / "scripts/rsi/run_harness_probe.py",
        ROOT / "scripts/rsi/run_protein_lifecycle.py",
        ROOT / "scripts/rsi/run_generated_lifecycle.py",
        ROOT / "scripts/rsi/run_paired_generated.py",
        ROOT / "src/agent/science_goal.py",
        ROOT / "src/agent/research_modes.py",
        *sorted((ROOT / "src/agent/harness").glob("*.py")),
        *sorted((ROOT / "src/agent/rsi").rglob("*.py")),
        *sorted((ROOT / "src/agent/graph").rglob("*.py")),
    ]
    return {str(p.relative_to(ROOT)): file_hash(p) for p in paths}


def verify_protocol(root):
    protocol = read(root / "protocol.json")
    if digest(protocol["design"]) != protocol["design_sha256"]:
        raise ValueError("Comparison design changed")
    for name, sha in protocol["implementation"].items():
        if file_hash(ROOT / name) != sha or file_hash(root / "implementation" / name) != sha:
            raise ValueError("Frozen comparison implementation changed")
    for row in protocol["design"]["schedule"]:
        arm = root / "arms" / row["id"]
        if file_hash(arm / "protocol.json") != protocol["arm_protocols"][row["id"]]:
            raise ValueError("Allocation protocol changed")
        state = GoalStore(arm).read()
        if state["definition_sha256"] != protocol["goal_definitions"][row["id"]]:
            raise ValueError("Allocation definition changed")
    return protocol


def create(args):
    base = Profile.from_dict(read(args.profile))
    if base.harness != "langgraph":
        raise ValueError("This controlled comparison uses the validated common execution-library family")
    profiles = {c: replace(base, mode=MODES[c]) for c in CONDITIONS}
    for p in profiles.values():
        if not builtin_registry().preflight(p, default_runtimes())["ready"]:
            raise ValueError("Unsupported comparison profile")
    args.output.mkdir(parents=True, exist_ok=False)
    design = {
        "cases": args.cases,
        "repeats": args.repeats,
        "schedule": schedule(args.cases, args.repeats),
        "profiles": {c: p.public() for c, p in profiles.items()},
        "goal_budget": BUDGET,
        "objective": OBJECTIVE,
        "user_turn_policy": TURN_POLICY,
        "primary": "fraction of all allocated objectives with independently accepted model and verified fresh-session reuse",
        "secondary": [
            "all allocated call/time costs, including failures",
            "episode counts and recovery",
            "scripted external turns and internal gates",
            "future RMSE conditional on complete-objective success",
        ],
        "comparisons": {
            "product_modes": "science_goal versus scripted_agent_turns under declared assistance policy",
            "coordination": "science_goal versus persistent_agent_ablation with identical persistence and scientific stage policy",
            "continuation_context": "persistent_agent_ablation versus scripted_agent_turns; both share the evaluator and budget supervisor, so this tests context/turn policy, not all consequences of persistence",
        },
        "failure_policy": "No extra allocations or automatic allocation retries. Within-objective retries obey the fixed stage policy and cumulative budget; every failed episode remains recorded. Unknown crash usage retains its reservation.",
        "selection_feedback": "Binary criterion results and coarse failure summaries only; no validation metric in executor context",
        "runtime_concurrency": args.concurrency,
        "randomness": "Fresh isolated sessions, API alias and temperature fixed; no claim that provider RNG or serving weights are controlled",
        "unit": "Cold-start run repeated on a fixed protein partition; not an independent protein family or an accumulated capability lineage",
        "scope": "Exploratory engineering comparison on previously used retrospective partitions; no confirmatory biological or cross-family RSI inference",
    }
    hashes = implementation()
    for name in hashes:
        dst = args.output / "implementation" / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes((ROOT / name).read_bytes())
    (args.output / "profiles").mkdir()
    for condition, p in profiles.items():
        write_json(args.output / "profiles" / (condition + ".json"), p.public())
    arm_hashes, definitions, data = {}, {}, {}
    for row in design["schedule"]:
        arm = args.output / "arms" / row["id"]
        store = prepare(
            SimpleNamespace(
                output=arm,
                profile=args.output / "profiles" / (row["condition"] + ".json"),
                objective=OBJECTIVE,
                case=row["case"],
                condition=row["condition"],
                source_cache=args.source_cache,
                **BUDGET,
            )
        )
        protocol = read(arm / "protocol.json")
        protocol.update(
            implementation=hashes, comparison_allocation=row, user_turn_policy=TURN_POLICY, scope_note=design["scope"]
        )
        for name in hashes:
            dst = arm / "implementation" / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes((ROOT / name).read_bytes())
        write_json(arm / "protocol.json", protocol)
        if row["case"] in data and data[row["case"]] != protocol["data_hashes"]:
            raise ValueError("Case data differs across paired conditions")
        data[row["case"]] = protocol["data_hashes"]
        arm_hashes[row["id"]] = file_hash(arm / "protocol.json")
        definitions[row["id"]] = store.read()["definition_sha256"]
    write_json(
        args.output / "protocol.json",
        {
            "design": design,
            "design_sha256": digest(design),
            "implementation": hashes,
            "arm_protocols": arm_hashes,
            "goal_definitions": definitions,
            "paired_data_hashes": data,
        },
    )
    write_json(args.output / "report.json", {"status": "prepared", "allocations": []})
    print(
        json.dumps({"status": "prepared", "allocations": len(arm_hashes), "design_sha256": digest(design)}), flush=True
    )


async def worker(args):
    root = args.output.resolve()
    protocol = verify_protocol(root)
    row = next(r for r in protocol["design"]["schedule"] if r["id"] == args.allocation)
    folder = root / "arms" / row["id"]
    store = GoalStore(folder)
    original = store.read()
    if original["cycles"] or original["inflight"] or original["status"] != "active":
        raise ValueError("Allocation already attempted; no automatic restart")
    profile = Profile.from_dict(original["definition"]["profile"])
    key = credential_from_toml(args.credential_toml, profile.model)
    plugin = GeneratedModelGoal(store, key)
    started = time.monotonic()
    turns = []

    async def cycle(index, p, state, context):
        turn = {
            "index": index,
            "external_user_turn": index == 0 or row["condition"] == "scripted_agent_turns",
            "human_intervention": False,
            "condition": row["condition"],
            "context": context,
            "context_sha256": digest(context),
            "checkpoint": state,
            "episode_profile": p.public(),
        }
        (folder / "turns").mkdir(exist_ok=True)
        write_json(folder / "turns" / f"turn_{index:03d}.json", turn)
        turns.append(turn)
        return await plugin.cycle(index, p, state, context)

    try:
        state = await run_goal(store, cycle, plugin.evaluate)
        result = {
            "status": state["status"],
            "criteria": state["assessment"],
            "usage": state["usage"],
            "episodes": len(state["cycles"]),
        }
        if state["status"] == "completed":
            lc, mid = plugin.lifecycle(state)
            result["future_metrics"] = get_model_plugin("generated_python.v1").evaluate(
                lc._model(mid), plugin.data["future"], plugin.scope
            )
            write_json(folder / "future_metrics.json", result["future_metrics"])
    except Exception as exc:
        state = store.mutate(lambda s: s.update(status="interrupted"))
        result = {
            "status": "worker_error",
            "error_type": type(exc).__name__,
            "goal_status": state["status"],
            "usage": state["usage"],
        }
    records = [read(p) for p in sorted((folder / "episodes").glob("*/execution/run.json"))]
    result.update(
        allocation=row,
        all_elapsed_seconds=time.monotonic() - started,
        external_scripted_turns=sum(t["external_user_turn"] for t in turns),
        internal_scripted_gates=sum(len(r.get("scripted_feedback", [])) for r in records),
        human_interventions=sum(r.get("human_interventions", 0) for r in records),
        runtime_failures=[
            {"episode": i, "status": r["status"]} for i, r in enumerate(records) if r["status"] != "completed"
        ],
        implementation_unchanged=all(file_hash(ROOT / n) == h for n, h in protocol["implementation"].items()),
    )
    write_json(folder / "comparison_result.json", result)
    print(
        json.dumps({"allocation": row["id"], "status": result["status"], "episodes": len(state["cycles"])}), flush=True
    )


async def run(args):
    root = args.output.resolve()
    with (root / ".campaign.lock").open("a") as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        protocol = verify_protocol(root)
        if read(root / "report.json")["status"] != "prepared":
            raise ValueError("Campaign already attempted; inspect existing workers instead of restarting")
        for profile_value in protocol["design"]["profiles"].values():
            if not credential_from_toml(args.credential_toml, Profile.from_dict(profile_value).model):
                raise ValueError("Comparison credential is unavailable")
        rows = protocol["design"]["schedule"]
        report = {
            "status": "running",
            "allocations": [],
            "planned": len(rows),
            "design_sha256": protocol["design_sha256"],
        }
        write_json(root / "report.json", report)
        sem = asyncio.Semaphore(protocol["design"]["runtime_concurrency"])

        async def one(row):
            async with sem:
                folder = root / "arms" / row["id"]
                record = {**row, "status": "running"}
                report["allocations"].append(record)
                env = {
                    "PATH": str(Path(sys.executable).parent) + ":/usr/bin:/bin",
                    "PYTHONPATH": str(ROOT / "src"),
                    "HOME": str(Path.home()),
                    "LANG": "C.UTF-8",
                    "OPENBLAS_NUM_THREADS": "1",
                    "OMP_NUM_THREADS": "1",
                }
                worker_started = time.monotonic()
                with (folder / "worker.log").open("wb") as log:
                    proc = await asyncio.create_subprocess_exec(
                        sys.executable,
                        "-u",
                        str(Path(__file__)),
                        "worker",
                        "--output",
                        str(root),
                        "--allocation",
                        row["id"],
                        "--credential-toml",
                        str(args.credential_toml),
                        cwd=ROOT,
                        env=env,
                        stdin=subprocess.DEVNULL,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                    )
                    record["pid"] = proc.pid
                    write_json(root / "report.json", report)
                    code = await proc.wait()
                result = folder / "comparison_result.json"
                record.update(
                    status=read(result)["status"] if result.exists() else "worker_missing_result",
                    exit_code=code,
                    worker_wall_seconds=time.monotonic() - worker_started,
                )
                write_json(root / "report.json", report)
                print(json.dumps({"allocation": row["id"], "status": record["status"]}), flush=True)

        await asyncio.gather(*(one(row) for row in rows))
        report["status"] = "completed"
        report["all_allocations_attempted"] = len(report["allocations"]) == len(rows)
        report["independent_audit_required"] = True
        report["reported_complete_objectives"] = sum(r["status"] == "completed" for r in report["allocations"])
        write_json(root / "report.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["create", "run", "worker"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", type=Path, default=ROOT / "configs/harness/glm51_science_goal.json")
    parser.add_argument("--credential-toml", type=Path)
    parser.add_argument("--source-cache", type=Path, default=ROOT / "research_runs/rsi/20261011_paper_case_audit")
    parser.add_argument("--cases", choices=CASES, nargs="+", default=list(CASES))
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--allocation")
    args = parser.parse_args()
    if not 1 <= args.repeats <= 5 or not 1 <= args.concurrency <= 2 or len(set(args.cases)) != len(args.cases):
        parser.error("invalid fixed allocation plan")
    if args.action != "create" and args.credential_toml is None:
        parser.error("Execution requires --credential-toml")
    if args.action == "create":
        create(args)
    elif args.action == "worker":
        asyncio.run(worker(args))
    else:
        asyncio.run(run(args))
