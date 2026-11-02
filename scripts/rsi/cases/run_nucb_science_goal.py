"""Persistent NucB Scientist goal with classification-aware acceptance and reuse.

G3 is a development-selection set. G4 is scored only after goal completion.
Episode failures consume the shared goal budget; accepted models survive a
failed reuse episode. This entry does not replay or alter an earlier campaign.
"""

from __future__ import annotations

# Direct-script entry point; implementation imports remain package-qualified.
import sys as _entry_sys
from pathlib import Path as _EntryPath
_entry_root = _EntryPath(__file__).resolve().parents[3]
_entry_sys.path[:0] = [str(_entry_root), str(_entry_root / 'src'), str(_entry_root / 'scripts/rsi')]


import argparse
import asyncio
import json
from pathlib import Path

from run_generated_lifecycle import SYSTEM
from run_harness_probe import ROOT
from scripts.rsi.cases.run_nucb_generated_matrix import PLUGIN, development_prompt, policy
from run_science_goal import GeneratedModelGoal, read

from agent.harness import Profile
from agent.harness.catalog import resolve_credential
from agent.harness.contracts import digest, file_hash
from agent.harness.host import write_json
from agent.rsi.engine.experience import LIMITATION, validate_projection
from agent.rsi.engine.model_plugins import get_model_plugin
from agent.rsi.cases.nucb_case import prepare_nucb_case
from agent.rsi.engine.versions import VersionStore
from agent.science_goal import GoalSpec, GoalStore, run_goal


def goal_development_prompt():
    return development_prompt("compact_probability").replace(
        "use only training feedback for model selection.",
        "use training feedback within this episode. The controller may supply aggregate G3 development feedback between episodes; this is adaptive model selection, never confirmatory evidence.",
    )


class NucbModelGoal(GeneratedModelGoal):
    def __init__(self, store, credential):
        super().__init__(store, credential)
        if (
            self.model_plugin_id != PLUGIN
            or self.protocol["model_plugin_sha256"] != get_model_plugin(PLUGIN).source_sha256
        ):
            raise ValueError("Goal classifier binding changed")

    def feedback_policy(self):
        value = self.protocol.get("development_feedback", "none")
        if value not in {"none", "source_and_metrics"}:
            raise ValueError("Unknown classifier development feedback policy")
        return value

    def development_feedback(self, state):
        if self.feedback_policy() == "none":
            return []
        feedback = []
        for cycle in state["cycles"]:
            if cycle.get("phase") != "develop":
                continue
            folder = self.store.root / "episodes" / f"cycle_{cycle['index']:03d}"
            path = folder / "acceptance.json"
            if not path.exists():
                continue
            assessment = read(path)
            controller = folder / "controller"
            mid = assessment["model_id"]
            current = read(controller / "state.json")
            if (
                digest(assessment) != current["candidates"][mid]["assessment_sha256"]
                or assessment != read(controller / ("assessment_" + mid + ".json"))
                or not assessment["training_replayed"]
            ):
                raise ValueError("Classifier development feedback assessment changed")
            VersionStore(controller / "artifacts").verify(mid)
            model = read(controller / "artifacts/versions" / mid / "model.json")
            if model["scope"] != self.scope or model["plugin_id"] != self.model_plugin_id:
                raise ValueError("Classifier development feedback scope changed")
            feedback.append(
                {
                    "cycle": cycle["index"],
                    "source_sha256": model["source_sha256"],
                    "source_code": model["recipe"]["source_code"],
                    "acceptance_passed": assessment["passed"],
                    "development_acceptance_metrics": {
                        key: assessment["metrics"][key]
                        for key in [
                            "rows",
                            "brier",
                            "training_prior_brier",
                            "brier_skill_vs_training_prior",
                            "log_loss",
                            "accuracy",
                            "balanced_accuracy",
                            "class_recall",
                        ]
                    },
                    "evidence": {"assessment_sha256": digest(assessment), "model_sha256": digest(model)},
                }
            )
        return feedback[-2:]

    async def cycle(self, index, profile, state, context):
        feedback = [] if state["progress"].get("accepted_model") else self.development_feedback(state)
        if feedback:
            context += (
                "\nPrevious independently replayed DEVELOPMENT acceptance feedback; fields follow the configured history policy. "
                "Discuss the supplied observations and choose the next implementation accordingly. These are adaptive "
                "development-selection observations, not confirmatory outcomes. No future labels, metrics, "
                "donor weights or prior-campaign model sources are provided.\n" + json.dumps(feedback)
            )
        outcome = await super().cycle(index, profile, state, context)
        folder = self.store.root / "episodes" / f"cycle_{index:03d}"
        write_json(folder / "development_feedback.json", feedback)
        outcome["development_feedback_sha256"] = digest(feedback)
        outcome["feedback_source_sha256"] = [row["source_sha256"] for row in feedback]
        candidate = folder / "workspace/candidate.json"
        if candidate.exists():
            source_hash = read(candidate)["source_sha256"]
            outcome["submitted_source_sha256"] = source_hash
            outcome["source_bytes_changed_from_feedback"] = (
                bool(feedback) and source_hash not in outcome["feedback_source_sha256"]
            )
        write_json(folder / "outcome.json", outcome)
        return outcome

    def development_action(self):
        return goal_development_prompt()

    def development_experience(self, state):
        return validate_projection({"schema": 1, "condition": "none", "entries": [], "limitation": LIMITATION})

    def acceptance_policy(self, state):
        return policy(self.scope, self.data["training"], self.data["validation"])


def prepare(args):
    if not args.profile or not args.objective or not args.assets or not args.rebuilt:
        raise ValueError("Create requires profile, explicit objective and verified public source paths")
    profile = Profile.from_dict(read(args.profile))
    scope, train, val, future, source, split = prepare_nucb_case(args.assets, args.rebuilt)
    if getattr(args, "feedback", "none") not in {"none", "source_and_metrics"}:
        raise ValueError("Unknown classifier development feedback policy")
    store = GoalStore.create(
        args.output,
        GoalSpec(
            args.objective,
            ("accepted_model", "fresh_session_reuse"),
            max_cycles=args.max_cycles,
            model_calls=args.model_calls,
            tool_calls=args.tool_calls,
            wall_seconds=args.wall_seconds,
        ),
        profile,
        condition=getattr(args, "condition", "science_goal"),
    )
    (store.root / "data").mkdir()
    for name, rows in [("training", train), ("validation", val), ("future", future)]:
        write_json(store.root / "data" / (name + ".json"), rows)
    paths = [
        Path(__file__).resolve(),
        *(
            ROOT / "scripts/rsi" / name
            for name in [
                "run_science_goal.py",
                "run_nucb_generated_matrix.py",
                "run_generated_lifecycle.py",
                "run_paired_generated.py",
                "run_harness_probe.py",
                "run_protein_lifecycle.py",
            ]
        ),
        ROOT / "src/agent/science_goal.py",
        ROOT / "src/agent/research_modes.py",
        ROOT / "src/agent/kimi_client.py",
        *sorted((ROOT / "src/agent/harness").glob("*.py")),
        *sorted((ROOT / "src/agent/rsi").rglob("*.py")),
        *sorted((ROOT / "src/agent/graph").rglob("*.py")),
        *sorted((ROOT / "src/agent/prompts").glob("*.md")),
    ]
    hashes = {str(p.relative_to(ROOT)): file_hash(p) for p in paths}
    for p in paths:
        dst = store.root / "implementation" / p.relative_to(ROOT)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(p.read_bytes())
    write_json(
        store.root / "protocol.json",
        {
            "case": "nucb",
            "development_feedback": getattr(args, "feedback", "none"),
            "scope": scope,
            "source": source,
            "split": split,
            "model_plugin_id": PLUGIN,
            "model_plugin_sha256": get_model_plugin(PLUGIN).source_sha256,
            "implementation": hashes,
            "data_hashes": {
                n: digest(rows) for n, rows in [("training", train), ("validation", val), ("future", future)]
            },
            "development_prompt_sha256": digest(goal_development_prompt()),
            "instructions_sha256": digest(SYSTEM),
            "acceptance_role": "G3 development selection; adaptive episode attempts are not unbiased acceptance replicates",
            "future_policy": "Score G4 only after the whole goal is verified; no future-label episode feedback",
            "progress_policy": "Retain independently accepted model across failed reuse episodes; all failures and physical calls consume the common goal budget",
            "scope_note": "Public retrospective single-protein persistent-goal integration; not a mode superiority or RSI efficacy comparison",
        },
    )
    return store


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
        plugin = NucbModelGoal(store, resolve_credential(profile.model, args.credential_toml))
        state = await run_goal(store, plugin.cycle, plugin.evaluate)
        if state["status"] == "completed" and not (store.root / "future_metrics.json").exists():
            lc, mid = plugin.lifecycle(state)
            write_json(
                store.root / "future_metrics.json",
                get_model_plugin(PLUGIN).evaluate(lc._model(mid), plugin.data["future"], plugin.scope),
            )
    print(state["status"], len(state["cycles"]), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["create", "run", "pause", "resume", "status"])
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--profile", type=Path)
    p.add_argument("--objective")
    p.add_argument("--assets", type=Path)
    p.add_argument("--rebuilt", type=Path)
    p.add_argument("--credential-toml", type=Path)
    p.add_argument("--feedback", choices=["none", "source_and_metrics"], default="none")
    p.add_argument(
        "--condition",
        choices=["science_goal", "persistent_agent_ablation", "scripted_agent_turns"],
        default="science_goal",
    )
    p.add_argument("--max-cycles", type=int, default=4)
    p.add_argument("--model-calls", type=int, default=72)
    p.add_argument("--tool-calls", type=int, default=36)
    p.add_argument("--wall-seconds", type=float, default=2400)
    asyncio.run(main(p.parse_args()))
