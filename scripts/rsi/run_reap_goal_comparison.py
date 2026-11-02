"""Matched complete REAP objectives with equal feedback and deferred future scoring.

Three partition seeds, two proteins and three declared interaction conditions.
Conversational control uses scripted external turns, not an actual user study.
"""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import json
import math
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from run_harness_probe import ROOT, credential_from_toml, default_runtimes
from run_reap_generated_matrix import FAMILIES, PLUGIN
from run_reap_science_goal import ReapModelGoal, prepare, publish_completed_source, read

from agent.harness import Profile, builtin_registry
from agent.harness.contracts import digest, file_hash
from agent.harness.host import write_json
from agent.rsi.model_lifecycle import ModelLifecycle
from agent.rsi.model_plugins import get_model_plugin
from agent.science_goal import GoalStore, run_goal

CONDITIONS = ("science_goal", "persistent_agent_ablation", "scripted_agent_turns")
MODES = {c: "science_expert" if c == "science_goal" else "science_agent" for c in CONDITIONS}
CASES = FAMILIES
BUDGET = {"max_cycles": 6, "model_calls": 96, "tool_calls": 48, "wall_seconds": 2700}
OBJECTIVE = (
    "Develop deterministic protein prediction model code for the supplied assay, "
    "independently verify its training and validation, then reuse the accepted "
    "model in a fresh execution session to predict every unseen input. "
    "Both independent model acceptance and verified fresh-session reuse are required."
)
TURN_POLICY = {
    "id": "verified_stage_turns.v1",
    "initial": "Supply the complete objective, declared criteria and current-stage task.",
    "after_rejection": "Supply the same last two independently verified source implementations and aggregate development-acceptance metrics, then request another attempt within remaining budget.",
    "after_acceptance": "Request fresh-session reuse of the accepted model on supplied unlabelled inputs.",
    "after_failed_reuse": "Request another reuse attempt with the same information and remaining budget.",
    "future_feedback": "Never provide final labels, scores or baseline results to any executor.",
    "external_turns": "Every episode in conversational control; only initial episode in persistent conditions.",
    "internal_expert_gates": "Fixed scripted task-context and plan confirmations, recorded separately from external turns.",
    "human_interventions": 0,
    "interpretation": "Scripted assistance policy, not a real human interaction study or automatic product mode switching.",
}


def process_start(pid):
    try:
        return (Path("/proc") / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()[19]
    except FileNotFoundError:
        return None


async def wait_for_dependency(root, protocol):
    design = protocol["design"]
    if not design["start_after"]:
        return
    folder = Path(design["start_after"])
    while True:
        if file_hash(folder / "protocol.json") != design["dependency_protocol_sha256"]:
            raise ValueError("Dependency protocol changed")
        state = read(folder / "report.json")["status"]
        if state == "campaign_completed":
            write_json(
                root / "dependency_receipt.json",
                {
                    "path": str(folder),
                    "report_sha256": file_hash(folder / "report.json"),
                    "finished_before_this_campaign": True,
                },
            )
            return
        if (
            state != "running"
            or process_start(design["dependency_pid"]) != design["dependency_process_start"]
            or design["dependency_process_start"] is None
        ):
            write_json(
                root / "report.json",
                {"status": "dependency_not_completed", "dependency_status": state, "planned": 18, "allocations": []},
            )
            raise ValueError("Dependency is not completed or its original controller is no longer alive")
        write_json(
            root / "report.json",
            {"status": "waiting_for_dependency", "dependency": str(folder), "planned": 18, "allocations": []},
        )
        await asyncio.sleep(30)


def select_model(store, state, *, metric="mse_skill_vs_training_mean"):
    candidates = []
    for cycle in state["cycles"]:
        folder = store.root / "episodes" / f"cycle_{cycle['index']:03d}"
        path = folder / "acceptance.json"
        if not path.exists():
            continue
        assessment = read(path)
        controller = folder / "controller"
        lifecycle = ModelLifecycle(controller, read(controller / "policy.json"))
        mid = assessment["model_id"]
        lifecycle._model(mid)
        current = lifecycle._state()
        if (
            digest(assessment) != current["candidates"][mid]["assessment_sha256"]
            or assessment != read(controller / ("assessment_" + mid + ".json"))
            or not assessment["training_replayed"]
        ):
            raise ValueError("Development-selected source lacks independent assessment binding")
        registered = mid in current["published"] and current["candidates"][mid]["status"] == "registered"
        skill = assessment["metrics"][metric]
        if type(assessment["passed"]) is not bool or registered != assessment["passed"] or not math.isfinite(skill):
            raise ValueError("Development assessment publication or metric is inconsistent")
        candidates.append(
            {
                "cycle": cycle["index"],
                "controller": str(controller.relative_to(store.root)),
                "model_id": mid,
                "accepted": assessment["passed"],
                "development_skill": skill,
                "assessment_sha256": digest(assessment),
            }
        )
    incumbent = state["progress"].get("accepted_model")
    if incumbent:
        selected = next(
            (
                c
                for c in candidates
                if c["model_id"] == incumbent["model_id"]
                and c["controller"] == incumbent["controller"]
                and c["accepted"]
            ),
            None,
        )
        if selected is None:
            raise ValueError("Goal incumbent is not independently accepted")
        return {**selected, "selection_reason": "goal_incumbent"}
    accepted = [c for c in candidates if c["accepted"]]
    if accepted:
        return {**min(accepted, key=lambda c: c["cycle"]), "selection_reason": "first_independently_accepted"}
    if candidates:
        return {
            **max(candidates, key=lambda c: (c["development_skill"], -c["cycle"])),
            "selection_reason": "best_development_submission",
        }
    return None


def freeze_goal_barrier(root, allocations, *, expected_count=18):
    if (
        expected_count < 1
        or len(allocations) != expected_count
        or len({r["id"] for r in allocations}) != expected_count
    ):
        raise ValueError(f"Exactly {expected_count} goal allocations required")
    records = []
    for row in allocations:
        folder = root / "arms" / row["id"]
        state, result = GoalStore(folder).read(), read(folder / "comparison_result.json")
        if (
            state["status"] not in {"completed", "budget_exhausted"}
            or state["inflight"] is not None
            or result["status"] != state["status"]
            or not result["implementation_unchanged"]
        ):
            raise ValueError("Every goal must reach an intact terminal state before future scoring")
        if (folder / "future_metrics.json").exists() or (folder / "selected_future_metrics.json").exists():
            raise ValueError("Future metrics appeared before the global goal barrier")
        records.append(
            {
                "id": row["id"],
                "goal_sha256": file_hash(folder / "goal.json"),
                "comparison_result_sha256": file_hash(folder / "comparison_result.json"),
                "status": state["status"],
            }
        )
    write_json(
        root / "development_barrier.json",
        {
            "allocations": records,
            "count": expected_count,
            "future_metrics_present": False,
            "created_at_unix": time.time(),
        },
    )
    return file_hash(root / "development_barrier.json")


def measured_calls(folder):
    calls = {"observed_model_calls": 0, "observed_tool_calls": 0}
    for filename, field, event, key in [
        ("model_requests.jsonl", "event", "request", "observed_model_calls"),
        ("events.jsonl", "kind", "tool_call_start", "observed_tool_calls"),
    ]:
        for path in (folder / "episodes").glob("*/execution/" + filename):
            calls[key] += sum(
                json.loads(line).get(field) == event for line in path.read_text().splitlines() if line.strip()
            )
    return calls


def contrast_rows(rows):
    table = {(r["case"], r["seed"], r["condition"]): r for r in rows}
    pairs = [
        ("science_goal", "scripted_agent_turns"),
        ("science_goal", "persistent_agent_ablation"),
        ("persistent_agent_ablation", "scripted_agent_turns"),
    ]
    contrasts = []
    for family, seed in sorted({k[:2] for k in table}):
        for left, right in pairs:
            a, b = table[(family, seed, left)], table[(family, seed, right)]
            contrasts.append(
                {
                    "case": family,
                    "seed": seed,
                    "left": left,
                    "right": right,
                    "complete_objective_difference": int(a["complete_objective"]) - int(b["complete_objective"]),
                    "ranking_utility_difference": a["ranking_utility"] - b["ranking_utility"],
                    "observed_model_calls_difference": a["observed_model_calls"] - b["observed_model_calls"],
                    "external_scripted_turns_difference": a["external_scripted_turns"] - b["external_scripted_turns"],
                }
            )
    return contrasts


def score_all(root, allocations, *, contrasts=contrast_rows):
    barrier = read(root / "development_barrier.json")
    if barrier["count"] != len(allocations) or barrier["future_metrics_present"] is not False:
        raise ValueError("Scoring requires the full terminal barrier")
    for allocation, binding in zip(allocations, barrier["allocations"], strict=True):
        folder = root / "arms" / allocation["id"]
        if (
            binding["id"] != allocation["id"]
            or file_hash(folder / "goal.json") != binding["goal_sha256"]
            or file_hash(folder / "comparison_result.json") != binding["comparison_result_sha256"]
        ):
            raise ValueError("Goal changed after the global scoring barrier")
    rows = []
    for allocation in allocations:
        folder = root / "arms" / allocation["id"]
        store = GoalStore(folder)
        state = store.read()
        result = read(folder / "comparison_result.json")
        selected = select_model(store, state)
        if selected != result["selected_model"]:
            raise ValueError("Development-only model selection changed")
        metrics = None
        scored = {"status": "no_verified_candidate", "selected_model": selected}
        if selected:
            protocol = read(folder / "protocol.json")
            controller = folder / selected["controller"]
            lifecycle = ModelLifecycle(controller, read(controller / "policy.json"))
            try:
                metrics = get_model_plugin(PLUGIN).evaluate(
                    lifecycle._model(selected["model_id"]), read(folder / "data/future.json"), protocol["scope"]
                )
                scored.update(status="completed", metrics=metrics)
                if state["status"] == "completed":
                    write_json(folder / "future_metrics.json", metrics)
            except Exception as exc:
                scored.update(status="scoring_failed", error_type=type(exc).__name__)
        write_json(folder / "selected_future_metrics.json", scored)
        rows.append(
            {
                **allocation,
                "status": state["status"],
                "complete_objective": state["status"] == "completed",
                "selected_model": selected,
                "future_scoring_status": scored["status"],
                "spearman": metrics["spearman"] if metrics else None,
                "mse_skill": metrics["mse_skill_vs_training_mean"] if metrics else None,
                "ranking_utility": metrics["spearman"] if metrics else -1.0,
                "utility_is_failure_floor": metrics is None,
                "episodes": len(state["cycles"]),
                "external_scripted_turns": result["external_scripted_turns"],
                "internal_scripted_gates": result["internal_scripted_gates"],
                "human_interventions": result["human_interventions"],
                "charged_usage": state["usage"],
                **measured_calls(folder),
            }
        )
    return {
        "rows": rows,
        "paired_contrasts": contrasts(rows),
        "scope": "Matched exploratory goal completion and development-selected future performance under scripted assistance. Three overlapping splits and a shared future cohort are not biological replicates or RSI efficacy evidence.",
    }


def schedule(cases, seeds):
    if (
        tuple(cases) != CASES
        or len(seeds) != 3
        or len(set(seeds)) != 3
        or any(type(x) is not int or x not in range(4, 10) for x in seeds)
    ):
        raise ValueError("Both REAP proteins and three unique partition seeds in 4..9 required")
    rows = []
    for repeat, seed in enumerate(seeds):
        order = tuple(cases) if repeat % 2 == 0 else tuple(reversed(cases))
        for case in order:
            offset = (repeat + tuple(cases).index(case)) % len(CONDITIONS)
            for condition in CONDITIONS[offset:] + CONDITIONS[:offset]:
                rows.append(
                    {
                        "id": f"{case}_seed{seed:02d}_{condition}",
                        "case": case,
                        "seed": seed,
                        "repeat": repeat,
                        "condition": condition,
                    }
                )
    return rows


def implementation():
    paths = [
        Path(__file__),
        ROOT / "scripts/rsi/run_science_goal.py",
        ROOT / "scripts/rsi/run_reap_science_goal.py",
        ROOT / "scripts/rsi/run_reap_generated_matrix.py",
        ROOT / "src/agent/kimi_client.py",
        *sorted((ROOT / "src/agent/prompts").glob("*.md")),
        ROOT / "scripts/rsi/run_harness_probe.py",
        ROOT / "scripts/rsi/run_protein_lifecycle.py",
        ROOT / "scripts/rsi/run_generated_lifecycle.py",
        ROOT / "scripts/rsi/run_paired_generated.py",
        ROOT / "src/agent/science_goal.py",
        ROOT / "src/agent/research_modes.py",
        *sorted((ROOT / "src/agent/harness").glob("*.py")),
        *sorted((ROOT / "src/agent/rsi").glob("*.py")),
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
    if base.model.model != "glm-5.2":
        raise ValueError("This GLM experiment is bound to glm-5.2")
    if base.harness != "langgraph":
        raise ValueError("This controlled comparison uses the validated common execution-library family")
    profiles = {c: replace(base, mode=MODES[c]) for c in CONDITIONS}
    for p in profiles.values():
        if not builtin_registry().preflight(p, default_runtimes())["ready"]:
            raise ValueError("Unsupported comparison profile")
    args.output.mkdir(parents=True, exist_ok=False)
    design = {
        "cases": args.cases,
        "seeds": args.seeds,
        "schedule": schedule(args.cases, args.seeds),
        "profiles": {c: p.public() for c, p in profiles.items()},
        "goal_budget": BUDGET,
        "objective": OBJECTIVE,
        "user_turn_policy": TURN_POLICY,
        "primary": "fraction of all allocated objectives with independently accepted model and verified fresh-session reuse",
        "secondary": [
            "all allocated call/time costs, including failures",
            "episode counts and recovery",
            "scripted external turns and internal gates",
            "higher-order Spearman/MSE skill of the development-selected source for every allocation",
            "predeclared utility -1 when no valid future score exists, distinct from observed correlation",
        ],
        "comparisons": {
            "product_modes": "science_goal versus scripted_agent_turns under declared assistance policy",
            "coordination": "science_goal versus persistent_agent_ablation with identical persistence and scientific stage policy",
            "continuation_context": "persistent_agent_ablation versus scripted_agent_turns; both share the evaluator and budget supervisor, so this tests context/turn policy, not all consequences of persistence",
        },
        "failure_policy": "No extra allocations or automatic allocation retries. Within-objective retries obey the fixed stage policy and cumulative budget; every failed episode remains recorded. Unknown crash usage retains its reservation.",
        "selection_feedback": "Identical last-two-source and aggregate development-acceptance feedback for all conditions; two internal training-only evaluations and one independent acceptance per development episode.",
        "selection_policy": "Keep the independently accepted model if any; otherwise choose the independently replayed submission with maximal development acceptance MSE skill, breaking ties by earlier episode. Never select using future metrics.",
        "future_policy": "All 18 complete-objective allocations must reach terminal status before any higher-order future score is computed. Rejected but independently verified selected models remain in the scientific outcome denominator.",
        "start_after": str(args.after_campaign.resolve()) if args.after_campaign else None,
        "dependency_protocol_sha256": file_hash(args.after_campaign / "protocol.json") if args.after_campaign else None,
        "dependency_pid": args.after_pid,
        "dependency_process_start": process_start(args.after_pid) if args.after_pid else None,
        "runtime_concurrency": args.concurrency,
        "randomness": "Fresh isolated sessions, API alias and temperature fixed; no claim that provider RNG or serving weights are controlled",
        "unit": "Cold-start goal on one of three overlapping single-mutant partitions; higher-order future cohort shared across seeds. Not independent biological lineages or accumulated capability experience.",
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
                family=row["case"],
                seed=row["seed"],
                condition=row["condition"],
                admission=args.admission,
                features=args.features,
                baseline=args.baseline,
                **BUDGET,
            )
        )
        protocol = read(arm / "protocol.json")
        protocol.update(
            implementation=hashes,
            comparison_allocation=row,
            user_turn_policy=TURN_POLICY,
            scope_note=design["scope"],
            future_policy=design["future_policy"],
        )
        for name in hashes:
            dst = arm / "implementation" / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes((ROOT / name).read_bytes())
        write_json(arm / "protocol.json", protocol)
        data_key = row["case"] + ":" + str(row["seed"])
        if data_key in data and data[data_key] != protocol["data_hashes"]:
            raise ValueError("Case data differs across paired conditions")
        data[data_key] = protocol["data_hashes"]
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


async def worker(args, *, goal_factory=None, select=select_model, publish=publish_completed_source):
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
    if goal_factory is not None:
        plugin = goal_factory(store, key)
    elif "improver" in read(folder / "protocol.json"):
        from run_reap_improver_goal import PolicyReapGoal

        plugin = PolicyReapGoal(store, key)
    else:
        plugin = ReapModelGoal(store, key)
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
            publish(store, plugin, state)
        result["selected_model"] = select(store, state)

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


def finish_campaign(root, rows, report, *, analyze, barrier):
    """Persist postprocessing progress and failures without altering development evidence."""
    report.update(status="postprocessing", postprocessing_stage="development_barrier", future_scoring_completed=False)
    write_json(root / "report.json", report)
    try:
        report["development_barrier_sha256"] = barrier(root, rows)
        report["postprocessing_stage"] = "analysis"
        write_json(root / "report.json", report)
        analysis = analyze(root, rows)
        report["postprocessing_stage"] = "publish_analysis"
        write_json(root / "report.json", report)
        write_json(root / "analysis.json", analysis)
        report.update(
            status="campaign_completed",
            postprocessing_stage="completed",
            analysis_sha256=file_hash(root / "analysis.json"),
            future_scoring_completed=True,
            finished_unix=time.time(),
        )
    except Exception as exc:
        report.update(
            status="postprocessing_failed",
            postprocessing_error={"type": type(exc).__name__, "message": str(exc)[:4000]},
            future_scoring_completed=False,
            finished_unix=time.time(),
        )
        write_json(root / "report.json", report)
        raise
    write_json(root / "report.json", report)


async def run(args, *, worker_script=None, analyze=score_all, barrier=freeze_goal_barrier):
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
        await wait_for_dependency(root, protocol)
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
                        str(worker_script or Path(__file__)),
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
        report["all_allocations_attempted"] = len(report["allocations"]) == len(rows)
        report["reported_complete_objectives"] = sum(r["status"] == "completed" for r in report["allocations"])
        report["independent_audit_required"] = True
        if not report["all_allocations_attempted"] or any(
            x["status"] not in {"completed", "budget_exhausted"} for x in report["allocations"]
        ):
            report["status"] = "campaign_incomplete"
            write_json(root / "report.json", report)
            return
        finish_campaign(root, rows, report, analyze=analyze, barrier=barrier)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["create", "run", "worker"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", type=Path, default=ROOT / "configs/harness/dmx_glm52.json")
    parser.add_argument("--credential-toml", type=Path)
    parser.add_argument("--admission", type=Path)
    parser.add_argument("--features", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--after-campaign", type=Path)
    parser.add_argument("--after-pid", type=int)
    parser.add_argument("--cases", choices=CASES, nargs="+", default=list(CASES))
    parser.add_argument("--seeds", type=int, nargs="+", default=[4, 5, 6])
    parser.add_argument("--concurrency", type=int, choices=[1], default=1)
    parser.add_argument("--allocation")
    args = parser.parse_args()
    if args.action == "create" and (
        not all([args.admission, args.features, args.baseline]) or bool(args.after_campaign) != bool(args.after_pid)
    ):
        parser.error("Create requires source paths and a paired dependency path/PID if used")
    if args.action != "create" and args.credential_toml is None:
        parser.error("Execution requires --credential-toml")
    if args.action == "create":
        create(args)
    elif args.action == "worker":
        asyncio.run(worker(args))
    else:
        asyncio.run(run(args))
