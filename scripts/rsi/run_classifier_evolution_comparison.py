"""Matched classifier lineage comparison with source, metric-only and restart controls.

Equal total budgets are caps, not a claim of identical realized cost. Fresh API
repetitions share retrospective cohorts; they are not biological replicates.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import run_reap_goal_comparison as common
from audit_science_evolution import audit
from run_classifier_evolution import (
    HISTORY_POLICIES,
    EvolvingClassifierGoal,
    implementation_paths,
    prepare,
    score_terminal_campaign,
)
from run_harness_probe import ROOT, default_runtimes
from run_science_goal import read

from agent.harness import Profile, builtin_registry
from agent.harness.contracts import digest, file_hash
from agent.harness.host import write_json

BUDGET = {"max_cycles": 6, "model_calls": 64, "tool_calls": 32, "wall_seconds": 1800}
CASES = ("nucb", "gfp")
OBJECTIVE = (
    "Author and test deterministic protein classification code from the supplied training rows. "
    "Obtain two independently accepted implementations with increasing development Brier skill, "
    "and reuse each in a new session. All attempts share one total budget. "
    "The host controls the permitted history; do not claim future-set or cross-protein improvement."
)


def schedule(repeats):
    if type(repeats) is not int or not 1 <= repeats <= 5:
        raise ValueError("Choose one to five declared API repeats")
    cells = [(c, m, h) for c in CASES for m in common.CONDITIONS for h in HISTORY_POLICIES]
    rows = []
    for repeat in range(repeats):
        offset = (repeat * 5) % len(cells)
        ordered = cells[offset:] + cells[:offset]
        for case, mode, history in ordered:
            rows.append(
                {
                    "id": f"r{repeat:02d}_{case}_{mode}_{history}",
                    "repeat": repeat,
                    "case": case,
                    "condition": mode,
                    "history_policy": history,
                }
            )
    return rows


def select_model(store, state):
    records = state["progress"].get("generations", [])
    if not records:
        return None
    record = records[-1]
    return {key: record[key] for key in ["controller", "model_id", "generation", "source_sha256", "experience_id"]}


def publish(store, plugin, state):
    # Every admitted generation was already registered by the shared controller.
    return None


def contrasts(rows):
    lookup = {(r["repeat"], r["case"], r["condition"], r["history_policy"]): r for r in rows}
    result = []
    for repeat, case in sorted({(r["repeat"], r["case"]) for r in rows}):
        pairs = []
        for mode in common.CONDITIONS:
            for a, b in [
                ("source_and_metrics", "metrics_only"),
                ("metrics_only", "independent_restart"),
                ("source_and_metrics", "independent_restart"),
            ]:
                pairs.append(((mode, a), (mode, b), "history"))
        for history in HISTORY_POLICIES:
            for a, b in [
                ("science_goal", "scripted_agent_turns"),
                ("science_goal", "persistent_agent_ablation"),
                ("persistent_agent_ablation", "scripted_agent_turns"),
            ]:
                pairs.append(((a, history), (b, history), "mode"))
        for (am, ah), (bm, bh), kind in pairs:
            a, b = lookup[(repeat, case, am, ah)], lookup[(repeat, case, bm, bh)]
            result.append(
                {
                    "repeat": repeat,
                    "case": case,
                    "kind": kind,
                    "left": a["id"],
                    "right": b["id"],
                    "complete_objective_difference": int(a["complete_objective"]) - int(b["complete_objective"]),
                    "selected_future_skill_difference": a["selected_future_skill"] - b["selected_future_skill"],
                    "observed_model_calls_difference": a["observed_model_calls"] - b["observed_model_calls"],
                }
            )
    return result


def score_all(root, allocations):
    # Audit executable lineage and reuse before any final labels are scored.
    for row in allocations:
        folder = root / "arms" / row["id"]
        write_json(folder / "lineage_audit.json", audit(folder, score_future=False))
    scored = score_terminal_campaign([root / "arms" / row["id"] for row in allocations], root / "terminal_scoring")
    rows = []
    for allocation, value in zip(allocations, scored, strict=True):
        folder = root / "arms" / allocation["id"]
        result = read(folder / "comparison_result.json")
        rows.append(
            {
                **allocation,
                **value,
                "complete_objective": value["status"] == "completed",
                "external_scripted_turns": result["external_scripted_turns"],
                "internal_scripted_gates": result["internal_scripted_gates"],
                "human_interventions": result["human_interventions"],
                **common.measured_calls(folder),
            }
        )
    return {
        "rows": rows,
        "paired_contrasts": contrasts(rows),
        "scope": "Retrospective paired API allocations. Source inheritance, metric feedback and extra search are separated; no biological replication, general RSI efficacy or self-improver claim.",
        "budget_scope": "Same total caps and per-episode controls; realized calls and time reported separately.",
        "denominator": "Every allocation retained; no accepted model gives declared training-prior fallback zero. Observed negative scores are retained.",
    }


def comparison_profiles(base):
    if base.model.model != "glm-5.2" or base.harness != "langgraph":
        raise ValueError("This experiment freezes GLM-5.2 on the common LangGraph family")
    base = replace(
        base,
        budget=replace(
            base.budget,
            wall_seconds=600,
            request_seconds=300,
            model_calls=24,
            tool_calls=12,
            output_tokens=12288,
            temperature=0.1,
        ),
    )
    profiles = {c: replace(base, mode=common.MODES[c]) for c in common.CONDITIONS}
    return profiles


def create(args):
    profiles = comparison_profiles(Profile.from_dict(read(args.profile)))
    for p in profiles.values():
        if not builtin_registry().preflight(p, default_runtimes())["ready"]:
            raise ValueError("Unsupported comparison profile")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    (root / "profiles").mkdir()
    for condition, p in profiles.items():
        write_json(root / "profiles" / (condition + ".json"), p.public())
    design = {
        "schedule": schedule(args.repeats),
        "profiles": {c: p.public() for c, p in profiles.items()},
        "goal_budget": BUDGET,
        "objective": OBJECTIVE,
        "generations": 2,
        "runtime_concurrency": 1,
        "start_after": None,
        "primary": "All-allocation verified two-version completion and development-selected future Brier skill",
        "history": {
            "source_and_metrics": "Accepted source via experience plus last two independently assessed sources and aggregate development scores",
            "metrics_only": "Same aggregate development scores and source fingerprints, without source code, method summaries or fitted parameters",
            "independent_restart": "No cross-episode source, scores or summaries; same training-only within-episode feedback, host selection and total budget",
        },
        "context": "Identical objective/episode context; earlier free-text checkpoints are not provided in any arm",
        "parent_semantics": "Code inheritance parents exist only in the source arm; all arms separately record the selection predecessor",
        "selection": "Last independently accepted source version; zero-skill prior fallback if none. Never choose by future score.",
        "future_policy": "All declared allocations terminal before any future score; no allocation replacement or restart",
        "mode_scope": "Science expert goal, persistent Agent coordination ablation and scripted Agent turns. Scripted turns are not a user study.",
        "repetition_scope": "Fresh API runs, shared retrospective public cohorts; no claim of biological independence or fixed provider RNG.",
        "scope": "Mechanistic exploratory comparison; no claim of autonomous improver evolution or general cross-protein benefit",
    }
    paths = implementation_paths()
    hashes = {str(p.relative_to(ROOT)): file_hash(p) for p in sorted(paths)}
    for name in hashes:
        dest = root / "implementation" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((ROOT / name).read_bytes())
    arms, definitions, data = {}, {}, {}
    for row in design["schedule"]:
        assets = args.nucb_assets if row["case"] == "nucb" else args.gfp_assets
        store = prepare(
            SimpleNamespace(
                output=root / "arms" / row["id"],
                profile=root / "profiles" / (row["condition"] + ".json"),
                objective=OBJECTIVE,
                case=row["case"],
                condition=row["condition"],
                history_policy=row["history_policy"],
                generations=2,
                assets=assets,
                rebuilt=args.nucb_rebuilt,
                **BUDGET,
            )
        )
        path = store.root / "protocol.json"
        protocol = read(path)
        protocol.update(
            implementation=hashes,
            comparison_allocation=row,
            scope_note=design["scope"],
            future_policy=design["future_policy"],
        )
        write_json(path, protocol)
        if row["case"] in data and data[row["case"]] != protocol["data_hashes"]:
            raise ValueError("Paired data differs")
        data[row["case"]] = protocol["data_hashes"]
        arms[row["id"]] = file_hash(path)
        definitions[row["id"]] = store.read()["definition_sha256"]
    write_json(
        root / "protocol.json",
        {
            "design": design,
            "design_sha256": digest(design),
            "implementation": hashes,
            "arm_protocols": arms,
            "goal_definitions": definitions,
            "paired_data_hashes": data,
        },
    )
    write_json(root / "report.json", {"status": "prepared", "allocations": []})
    print("prepared", len(arms), digest(design), flush=True)


def barrier(root, rows):
    return common.freeze_goal_barrier(root, rows, expected_count=len(rows))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["create", "run", "worker"])
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--profile", type=Path)
    p.add_argument("--credential-toml", type=Path)
    p.add_argument("--nucb-assets", type=Path)
    p.add_argument("--nucb-rebuilt", type=Path)
    p.add_argument("--gfp-assets", type=Path)
    p.add_argument("--repeats", type=int, choices=range(1, 6), default=1)
    p.add_argument("--allocation")
    args = p.parse_args()
    if args.action == "create":
        if not all([args.profile, args.nucb_assets, args.nucb_rebuilt, args.gfp_assets]):
            p.error("Create requires a profile and all public source paths")
        create(args)
    elif args.action == "worker":
        asyncio.run(common.worker(args, goal_factory=EvolvingClassifierGoal, select=select_model, publish=publish))
    else:
        asyncio.run(common.run(args, worker_script=Path(__file__).resolve(), analyze=score_all, barrier=barrier))
