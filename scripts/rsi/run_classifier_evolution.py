"""Source-verified multi-generation classification on public NucB and GFP data.

Each successor must improve the plugin's development metric, change executable
source, and be independently reused. All rejected attempts retain their original
cost and artifacts. Future labels/metrics never enter development episodes.
This is retrospective adaptive selection, not evidence of algorithmic novelty.
"""

from __future__ import annotations

import sys as _entry_sys
from pathlib import Path as _EntryPath
_entry_root = _EntryPath(__file__).resolve().parents[2]
_entry_sys.path.insert(0, str(_entry_root))


import argparse
import asyncio
from pathlib import Path

from run_generated_lifecycle import SYSTEM
from run_harness_probe import ROOT
from scripts.rsi.cases.run_htfunclib_generated import PROMPT as GFP_PROMPT
from scripts.rsi.cases.run_nucb_science_goal import NucbModelGoal, goal_development_prompt
from run_science_evolution import EvolvingModelGoal
from run_science_goal import GeneratedModelGoal, read

from agent.harness import Profile
from agent.harness.catalog import resolve_credential
from agent.harness.contracts import digest, file_hash
from agent.harness.host import write_json
from agent.rsi.cases.htfunclib_model_case import prepare_htfunclib_model_case
from agent.rsi.engine.lineage_promotion import new_evolution
from agent.rsi.engine.model_plugins import get_model_plugin
from agent.rsi.cases.nucb_case import prepare_nucb_case
from agent.rsi.engine.source_snapshot import research_implementation_paths
from agent.science_goal import GoalSpec, GoalStore, goal_cycle_prompt, run_goal

PLUGINS = {"nucb": "generated_classifier.v1", "gfp": "generated_classifier.large.v1"}
CRITERIA = ("accepted_model", "fresh_session_reuse", "verified_lineage")
HISTORY_POLICIES = ("source_and_metrics", "metrics_only", "independent_restart")


def implementation_paths():
    """Include runtime dependencies, agent code and role prompts."""
    return research_implementation_paths(ROOT)


class ClassifierModelGoal(NucbModelGoal):
    def __init__(self, store, credential):
        GeneratedModelGoal.__init__(self, store, credential)
        expected = PLUGINS[self.protocol["case"]]
        if (
            self.model_plugin_id != expected
            or self.protocol["model_plugin_sha256"] != get_model_plugin(expected).source_sha256
        ):
            raise ValueError("Goal classifier binding changed")

    def development_action(self):
        if self.protocol["case"] == "nucb":
            return goal_development_prompt()
        return GFP_PROMPT + (
            " Read the permitted model experience. Between episodes, the host may provide "
            "aggregate development-selection feedback and source generated earlier in THIS goal. "
            "Use it to propose the next implementation; no earlier private campaign history is supplied."
        )

    def acceptance_policy(self, state):
        plugin = get_model_plugin(self.model_plugin_id)
        return {
            "schema": 1,
            "scope": self.scope,
            "plugin_id": plugin.id,
            "plugin_sha256": plugin.source_sha256,
            "training_sha256": digest(self.data["training"]),
            "validation_sha256": digest(self.data["validation"]),
            "minimum_skill": 0.0,
            "max_candidates": 1,
        }


class EvolvingClassifierGoal(EvolvingModelGoal, ClassifierModelGoal):
    """Common budget/acceptance controller; explicit history interventions."""

    def history_policy(self):
        value = self.protocol.get("history_policy", "source_and_metrics")
        if value not in HISTORY_POLICIES:
            raise ValueError("Unknown classifier history intervention")
        return value

    def lineage_parents(self, records):
        return super().lineage_parents(records) if self.history_policy() == "source_and_metrics" else []

    def development_experience(self, state):
        if self.history_policy() == "source_and_metrics":
            return super().development_experience(state)
        return self.experience().project("none", [])

    def development_feedback(self, state):
        if self.history_policy() == "independent_restart":
            return []
        values = super().development_feedback(state)
        if self.history_policy() == "metrics_only":
            values = [{key: value for key, value in row.items() if key != "source_code"} for row in values]
        return values

    def generation_context(self, generation):
        if self.protocol.get("schema", 2) < 3:
            return super().generation_context(generation)
        disclosure = {
            "source_and_metrics": "You may inherit this goal's accepted source and its last two development feedback records.",
            "metrics_only": "Only aggregate development feedback and source fingerprints are supplied. Earlier source implementations and fitted states are withheld; author a fresh implementation.",
            "independent_restart": "Develop independently from the training data. No previous model source, summaries, development scores or fitted states are supplied.",
        }[self.history_policy()]
        return (
            f"\nRequested accepted version {generation} of {self.target}. {disclosure} "
            "Read the permitted model experience once. Choose and test a deterministic implementation. "
            "The same independent development-improvement and new-session reuse criteria apply in all arms. "
            "Do not claim algorithmic novelty or future-set improvement."
        )

    async def cycle(self, index, profile, state, context):
        if self.protocol.get("schema", 2) >= 3:
            # Shared checkpoint prose could leak an implementation/rationale to a
            # no-source arm. All interventions receive the same bounded context.
            context = goal_cycle_prompt(
                {
                    **state,
                    "cycles": [],
                    "progress": {"verified_versions": len(state["progress"].get("generations", []))},
                }
            )
        outcome = await super().cycle(index, profile, state, context)
        write_json(
            self.store.root / "episodes" / f"cycle_{index:03d}" / "goal_context.json",
            {"text": context, "sha256": digest(context), "history_policy": self.history_policy()},
        )
        outcome["history_policy"] = self.history_policy()
        write_json(self.store.root / "episodes" / f"cycle_{index:03d}" / "outcome.json", outcome)
        return outcome


def default_assets(case):
    base = ROOT / "research_runs/rsi"
    if case == "nucb":
        return base / "case_assets/nuclease_design/702284847c4558567d8288169a86dc147cfaab32"
    if case == "gfp":
        return base / "case_assets/htFuncLib/inspection_api/random_forest"
    raise ValueError("Unknown classifier case")


def prepare(args):
    if not args.profile or not args.objective:
        raise ValueError("Create requires a profile and explicit objective")
    if args.case not in PLUGINS or args.generations not in {2, 3}:
        raise ValueError("Choose a registered classifier case and two or three generations")
    if args.max_cycles < 2 * args.generations:
        raise ValueError("Each generation requires development and fresh-session reuse episodes")
    history = getattr(args, "history_policy", "source_and_metrics")
    if history not in HISTORY_POLICIES:
        raise ValueError("Unknown classifier history intervention")
    assets = args.assets or default_assets(args.case)
    if args.case == "nucb":
        rebuilt = args.rebuilt or ROOT / "research_runs/rsi/20261011_nucb_raw_reproduction_v2"
        scope, train, val, future, source, split = prepare_nucb_case(assets, rebuilt)
    else:
        scope, train, val, future, source, split = prepare_htfunclib_model_case(assets)
    profile = Profile.from_dict(read(args.profile))
    plugin = get_model_plugin(PLUGINS[args.case])
    store = GoalStore.create(
        args.output,
        GoalSpec(
            args.objective,
            CRITERIA,
            max_cycles=args.max_cycles,
            model_calls=args.model_calls,
            tool_calls=args.tool_calls,
            wall_seconds=args.wall_seconds,
        ),
        profile,
        condition=args.condition,
    )
    (store.root / "data").mkdir()
    data = {"training": train, "validation": val, "future": future}
    for name, rows in data.items():
        write_json(store.root / "data" / (name + ".json"), rows)
    paths = implementation_paths()
    hashes = {str(p.relative_to(ROOT)): file_hash(p) for p in paths}
    for p in paths:
        dst = store.root / "implementation" / p.relative_to(ROOT)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(p.read_bytes())
    write_json(
        store.root / "protocol.json",
        {
            "schema": 3,
            "history_policy": history,
            "parent_semantics": "Source-visible accepted predecessors only. Selection predecessor is tracked separately; source-withheld arms have no inherited-code parents.",
            "case": args.case,
            "scope": scope,
            "source": source,
            "split": split,
            "model_plugin_id": plugin.id,
            "model_plugin_sha256": plugin.source_sha256,
            "implementation": hashes,
            "data_hashes": {n: digest(rows) for n, rows in data.items()},
            "development_feedback": "none" if history == "independent_restart" else "source_and_metrics",
            "instructions_sha256": digest(SYSTEM),
            "evolution": {
                **new_evolution(args.generations),
                "metric": plugin.acceptance_metric,
                "promotion": "Strict development skill improvement plus changed executable source and fresh-session reuse",
                "source_gate": "AST hash excluding formatting, comments and docstrings; not proof of algorithmic novelty",
            },
            "future_policy": "No automatic future scoring. Explicit host scoring only after all declared allocations are terminal.",
            "selection_policy": "Last independently accepted generation; preserve first accepted generation as frozen-source control. No accepted model means training-prior fallback (skill zero).",
            "scope_note": "Retrospective public-data functionality pilot; adaptive development selection, not confirmatory efficacy.",
            "feedback_scope": "Only this goal's prior source and aggregate development feedback; no prior private campaign history or fitted donor weights",
        },
    )
    return store


def score_terminal_campaign(roots, output):
    """Seal all caller-declared allocations before evaluating any future labels.

    Caller must freeze the allocation list before execution. This helper refuses
    active goals and retains failed/partial goals in the result denominator.
    """
    if output.exists():
        raise FileExistsError(output)
    if not roots or len({p.resolve() for p in roots}) != len(roots):
        raise ValueError("Require distinct declared allocations")
    sealed, loaded = [], []
    for root in roots:
        store = GoalStore(root)
        state = store.read()
        if state["status"] not in {"completed", "budget_exhausted", "cancelled"} or state["inflight"]:
            raise ValueError("Every declared allocation must be terminal before future scoring")
        driver = EvolvingClassifierGoal(store, None)
        checks = driver.evaluate(state)
        if (state["status"] == "completed") != all(c["passed"] for c in checks["criteria"].values()):
            raise ValueError("Terminal criteria disagree with source-verified lineage")
        sealed.append(
            {
                "root": str(store.root),
                "goal_sha256": file_hash(root / "goal.json"),
                "protocol_sha256": file_hash(root / "protocol.json"),
                "status": state["status"],
            }
        )
        loaded.append((driver, state))
    output.mkdir(parents=True, exist_ok=False)
    write_json(
        output / "barrier.json",
        {"all_allocations_terminal": True, "allocations": sealed, "future_labels_delivered_to_agents": False},
    )
    results = []
    for seal, (driver, state) in zip(sealed, loaded, strict=True):
        if file_hash(driver.store.root / "goal.json") != seal["goal_sha256"]:
            raise ValueError("Goal changed after scoring barrier")
        generations = []
        plugin = get_model_plugin(driver.model_plugin_id)
        for record in state["progress"].get("generations", []):
            lc, mid = driver.lifecycle({"progress": {"accepted_model": record}})
            model = lc._model(mid)
            rebuilt = plugin.fit(driver.data["training"], driver.scope, model["recipe"])
            if digest(model) != digest(rebuilt):
                raise ValueError("Independent terminal training replay differs")
            assessment = read(lc.root / ("assessment_" + mid + ".json"))
            metrics = plugin.evaluate(rebuilt, driver.data["validation"], driver.scope)
            if digest(metrics) != digest(assessment["metrics"]):
                raise ValueError("Independent development replay differs")
            future = plugin.evaluate(rebuilt, driver.data["future"], driver.scope)
            generations.append(
                {
                    "generation": record["generation"],
                    "model_id": mid,
                    "source_sha256": model["source_sha256"],
                    "development_metrics": metrics,
                    "future_metrics": future,
                    "reused": bool(record.get("reuse_receipt")),
                }
            )
        chosen = generations[-1]["future_metrics"][plugin.acceptance_metric] if generations else 0.0
        frozen = generations[0]["future_metrics"][plugin.acceptance_metric] if generations else 0.0
        results.append(
            {
                **seal,
                "case": driver.protocol["case"],
                "history_policy": driver.history_policy(),
                "usage": state["usage"],
                "generations": generations,
                "selected_future_skill": chosen,
                "frozen_first_source_skill": frozen,
                "difference_from_frozen_first": chosen - frozen,
                "failure_convention": "no independently accepted model: training-prior fallback, skill zero",
                "control_scope": "Within-run frozen first source; not an equal-budget independent-search control",
            }
        )
    write_json(
        output / "results.json",
        {
            "status": "scored",
            "allocations": results,
            "scope": "All-allocation retrospective pilot; adaptive selection and within-run control do not establish general RSI efficacy",
        },
    )
    return results


async def main(args):
    store = prepare(args) if args.action == "create" else GoalStore(args.output)
    if args.action == "pause":
        state = store.pause()
    elif args.action in {"create", "status"}:
        state = store.read()
    else:
        if args.action == "resume":
            store.resume()
        profile = Profile.from_dict(store.read()["definition"]["profile"])
        driver = EvolvingClassifierGoal(store, resolve_credential(profile.model, args.credential_toml))
        state = await run_goal(store, driver.cycle, driver.evaluate)
    print(state["status"], len(state["cycles"]), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["create", "run", "pause", "resume", "status"])
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--profile", type=Path)
    p.add_argument("--objective")
    p.add_argument("--case", choices=list(PLUGINS), default="nucb")
    p.add_argument("--assets", type=Path)
    p.add_argument("--rebuilt", type=Path)
    p.add_argument("--credential-toml", type=Path)
    p.add_argument(
        "--condition",
        choices=["science_goal", "persistent_agent_ablation", "scripted_agent_turns"],
        default="science_goal",
    )
    p.add_argument("--history-policy", choices=HISTORY_POLICIES, default="source_and_metrics")
    p.add_argument("--generations", type=int, choices=[2, 3], default=2)
    p.add_argument("--max-cycles", type=int, default=8)
    p.add_argument("--model-calls", type=int, default=96)
    p.add_argument("--tool-calls", type=int, default=48)
    p.add_argument("--wall-seconds", type=float, default=3600)
    asyncio.run(main(p.parse_args()))
