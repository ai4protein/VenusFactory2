"""NucB whole-goal comparison: three execution policies by verified feedback.

Three independent API runs per condition share the same retrospective public
G1/G2, G3 and G4 cohorts. They are runtime repetitions, not biological replicates.
No source or history from a previous campaign is supplied to an executor.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import run_reap_goal_comparison as common
from run_harness_probe import ROOT, default_runtimes
from run_nucb_science_goal import PLUGIN, NucbModelGoal, prepare, read
from run_reap_science_goal import publish_completed_source

from agent.harness import Profile, builtin_registry
from agent.harness.contracts import digest, file_hash
from agent.harness.host import write_json
from agent.rsi.model_lifecycle import ModelLifecycle
from agent.rsi.model_plugins import get_model_plugin
from agent.science_goal import GoalStore

FEEDBACK = ("none", "source_and_metrics")
BUDGET = {"max_cycles": 6, "model_calls": 64, "tool_calls": 32, "wall_seconds": 1800}
METRIC = "brier_skill_vs_training_prior"
OBJECTIVE = (
    "Develop a deterministic full-sequence NucB probability model from the supplied public G1/G2 "
    "measurements, obtain independent G3 development acceptance, and reuse the accepted model "
    "in a fresh session on every supplied unlabelled input. Both verified acceptance and fresh-session "
    "reuse are required. All episodes, failures and model revisions share one total budget."
)


def schedule():
    conditions = [(mode, feedback) for mode in common.CONDITIONS for feedback in FEEDBACK]
    rows = []
    for repeat in range(3):
        ordered = conditions[repeat * 2 :] + conditions[: repeat * 2]
        for condition, feedback in ordered:
            rows.append(
                {
                    "id": f"r{repeat:02d}_{condition}_{feedback}",
                    "case": "nucb",
                    "repeat": repeat,
                    "condition": condition,
                    "feedback": feedback,
                }
            )
    return rows


def select_model(store, state):
    return common.select_model(store, state, metric=METRIC)


def publish(store, plugin, state):
    return publish_completed_source(
        store,
        plugin,
        state,
        summary="Autonomously generated NucB classifier source independently accepted and reused in a fresh goal episode. Source registration establishes provenance and availability, not transferable efficacy.",
    )


def contrasts(rows):
    lookup = {(r["repeat"], r["condition"], r["feedback"]): r for r in rows}
    result = []
    for repeat in range(3):
        for condition in common.CONDITIONS:
            left, right = (lookup[(repeat, condition, f)] for f in FEEDBACK)
            result.append(
                {
                    "repeat": repeat,
                    "condition": condition,
                    "comparison": "source_and_metrics_minus_none",
                    "complete_objective_difference": int(right["complete_objective"]) - int(left["complete_objective"]),
                    "brier_skill_difference": right["all_allocation_brier_skill"] - left["all_allocation_brier_skill"],
                }
            )
        for feedback in FEEDBACK:
            scientist = lookup[(repeat, "science_goal", feedback)]
            for condition in ["persistent_agent_ablation", "scripted_agent_turns"]:
                other = lookup[(repeat, condition, feedback)]
                result.append(
                    {
                        "repeat": repeat,
                        "feedback": feedback,
                        "comparison": "science_goal_minus_" + condition,
                        "complete_objective_difference": int(scientist["complete_objective"])
                        - int(other["complete_objective"]),
                        "brier_skill_difference": scientist["all_allocation_brier_skill"]
                        - other["all_allocation_brier_skill"],
                    }
                )
    return result


def score_all(root, allocations):
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
                plugin = get_model_plugin(PLUGIN)
                model = lifecycle._model(selected["model_id"])
                rebuilt = plugin.fit(read(folder / "data/training.json"), protocol["scope"], model["recipe"])
                if digest(rebuilt) != digest(model):
                    raise ValueError("Fresh classifier refit changed")
                metrics = plugin.evaluate(rebuilt, read(folder / "data/future.json"), protocol["scope"])
                scored.update(status="completed", metrics=metrics, training_replayed=True)
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
                "raw_metrics": metrics,
                "all_allocation_brier_skill": metrics[METRIC] if metrics else 0.0,
                "uses_prior_fallback": metrics is None,
                "episodes": len(state["cycles"]),
                "source_revision_episodes": sum(
                    bool(c.get("source_bytes_changed_from_feedback")) for c in state["cycles"]
                ),
                "external_scripted_turns": result["external_scripted_turns"],
                "internal_scripted_gates": result["internal_scripted_gates"],
                "human_interventions": result["human_interventions"],
                "charged_usage": state["usage"],
                **common.measured_calls(folder),
            }
        )
    return {
        "rows": rows,
        "paired_contrasts": contrasts(rows),
        "scope": "Exploratory repeated API runs on one previously examined public protein cohort. Common G4 labels are withheld from every executor, not unseen to investigators. No biological replication, prospective validation or cross-protein efficacy claim.",
        "fallback": "Missing or unscorable models receive training-prior Brier skill zero as an explicit allocation convention; observed negative skills are retained without clipping.",
    }


def create(args):
    base = Profile.from_dict(read(args.profile))
    if base.model.model != "glm-5.2" or base.harness != "langgraph":
        raise ValueError("This matched experiment is bound to GLM-5.2 and LangGraph")
    base = replace(
        base,
        budget=replace(
            base.budget,
            wall_seconds=900,
            request_seconds=300,
            model_calls=24,
            tool_calls=12,
            output_tokens=12288,
            temperature=0.1,
            thinking="enabled",
        ),
    )
    profiles = {c: replace(base, mode=common.MODES[c]) for c in common.CONDITIONS}
    for profile in profiles.values():
        if not builtin_registry().preflight(profile, default_runtimes())["ready"]:
            raise ValueError("Unsupported experiment profile")
    args.output.mkdir(parents=True, exist_ok=False)
    design = {
        "schedule": schedule(),
        "profiles": {c: p.public() for c, p in profiles.items()},
        "goal_budget": BUDGET,
        "objective": OBJECTIVE,
        "runtime_concurrency": 1,
        "start_after": None,
        "primary": "All-allocated fraction with independently accepted model and verified fresh-session reuse",
        "feedback_ablation": "none retains categorical stage outcomes and within-episode training feedback; source_and_metrics additionally receives the last two independently assessed sources and aggregate G3 metrics from this SAME goal. No prior campaign sources, private history or fitted donor weights.",
        "information": "Same public training rows, initial prompt, tools and acceptance criteria across all conditions. No G4 labels or scores are supplied. G3 feedback is adaptive development selection.",
        "mode_contrast": "Expert-team persistent goal vs scripted conversational turns, plus persistent-Agent coordination ablation. Scripted turns are not a human interaction study.",
        "selection_policy": "Keep accepted incumbent; otherwise earliest accepted candidate; otherwise maximal independently verified G3 Brier skill, breaking ties by earlier episode. No future selection.",
        "future_policy": "All 18 allocations must reach intact terminal states before any G4 scoring. Include rejected selected models and failed allocations. No revision after future scoring.",
        "failure_policy": "No allocation retries. Failed episodes and unknown usage reservations consume the same fixed budget. Missing model fallback is zero skill; observed negatives are retained.",
        "randomness": "Three fresh API runs per condition with fixed alias/temperature and rotated order. No control claim for provider weights or RNG. Same data cohort, not three biological replicates.",
        "scope": "Retrospective exploratory single-protein test of within-goal revision and execution policy, not accumulated cross-protein RSI efficacy.",
    }
    hashes = common.implementation()
    for path in [
        Path(__file__).resolve(),
        ROOT / "scripts/rsi/run_nucb_science_goal.py",
        ROOT / "scripts/rsi/run_nucb_generated_matrix.py",
    ]:
        hashes[str(path.relative_to(ROOT))] = file_hash(path)
    for name in hashes:
        dst = args.output / "implementation" / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes((ROOT / name).read_bytes())
    (args.output / "profiles").mkdir()
    for condition, profile in profiles.items():
        write_json(args.output / "profiles" / (condition + ".json"), profile.public())
    arm_hashes, definitions = {}, {}
    data_binding = None
    for row in design["schedule"]:
        arm = args.output / "arms" / row["id"]
        store = prepare(
            SimpleNamespace(
                output=arm,
                profile=args.output / "profiles" / (row["condition"] + ".json"),
                objective=OBJECTIVE,
                condition=row["condition"],
                feedback=row["feedback"],
                assets=args.assets,
                rebuilt=args.rebuilt,
                **BUDGET,
            )
        )
        protocol = read(arm / "protocol.json")
        protocol.update(
            implementation=hashes,
            comparison_allocation=row,
            scope_note=design["scope"],
            future_policy=design["future_policy"],
            acceptance_role="G3 is adaptive development selection. Only source_and_metrics receives aggregate metrics and source from this same goal; never raw validation rows or G4 outcomes.",
        )
        for name in hashes:
            dst = arm / "implementation" / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes((ROOT / name).read_bytes())
        write_json(arm / "protocol.json", protocol)
        if data_binding is not None and data_binding != protocol["data_hashes"]:
            raise ValueError("Paired data changed")
        data_binding = protocol["data_hashes"]
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
            "paired_data_hashes": data_binding,
        },
    )
    write_json(args.output / "report.json", {"status": "prepared", "allocations": []})
    print("prepared", len(arm_hashes), digest(design), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["create", "run", "worker"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--assets", type=Path)
    parser.add_argument("--rebuilt", type=Path)
    parser.add_argument("--credential-toml", type=Path)
    parser.add_argument("--allocation")
    args = parser.parse_args()
    if args.action == "create":
        if not all([args.profile, args.assets, args.rebuilt]):
            parser.error("Create requires public source paths and a profile")
        create(args)
    elif args.action == "worker":
        asyncio.run(common.worker(args, goal_factory=NucbModelGoal, select=select_model, publish=publish))
    else:
        if not args.credential_toml:
            parser.error("Run requires credential binding")
        asyncio.run(common.run(args, worker_script=Path(__file__).resolve(), analyze=score_all))
