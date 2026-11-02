"""Persistent REAP goals with frozen protein features and independent acceptance.

Accepted models survive failed reuse episodes. Future labels are absent from
runtime inputs and never supplied as feedback. An explicit persistent-Agent
condition is a research ablation, not the conversational product mode.
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
from scripts.rsi.cases.run_reap_generated_matrix import FAMILIES, PLUGIN, development_prompt, policy
from run_science_goal import GeneratedModelGoal, read

from agent.harness import Profile
from agent.harness.catalog import resolve_credential
from agent.harness.contracts import digest, file_hash
from agent.harness.host import write_json
from agent.rsi.engine.experience import LIMITATION, ModelExperienceStore, validate_projection
from agent.rsi.engine.model_plugins import get_model_plugin
from agent.rsi.cases.reap_model_case import prepare_reap_model_case
from agent.rsi.engine.versions import VersionStore
from agent.science_goal import GoalSpec, GoalStore, run_goal


class ReapModelGoal(GeneratedModelGoal):
    def __init__(self, store, credential):
        super().__init__(store, credential)
        if (
            self.model_plugin_id != PLUGIN
            or self.protocol["model_plugin_sha256"] != get_model_plugin(PLUGIN).source_sha256
        ):
            raise ValueError("Goal feature-model binding changed")

    def development_action(self):
        return development_prompt(self.scope)

    def inference_inputs(self):
        return [{k: row[k] for k in ["id", "sequence", "features"]} for row in self.data["future"]]

    def development_experience(self, state):
        return validate_projection({"schema": 1, "condition": "none", "entries": [], "limitation": LIMITATION})

    def development_feedback(self, state):
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
            if digest(assessment) != current["candidates"][mid]["assessment_sha256"]:
                raise ValueError("Development feedback assessment changed")
            VersionStore(controller / "artifacts").verify(mid)
            model = read(controller / "artifacts/versions" / mid / "model.json")
            feedback.append(
                {
                    "cycle": cycle["index"],
                    "source_sha256": model["source_sha256"],
                    "source_code": model["recipe"]["source_code"],
                    "acceptance_passed": assessment["passed"],
                    "development_acceptance_metrics": {
                        key: assessment["metrics"][key]
                        for key in ["rows", "rmse", "training_mean_rmse", "mse_skill_vs_training_mean", "spearman"]
                    },
                    "evidence": {"assessment_sha256": digest(assessment), "model_sha256": digest(model)},
                }
            )
        return feedback[-2:]

    async def cycle(self, index, profile, state, context):
        feedback = [] if state["progress"].get("accepted_model") else self.development_feedback(state)
        if feedback:
            context += (
                "\nPrevious independently replayed source and aggregate DEVELOPMENT acceptance feedback. "
                "Discuss why the model failed and revise it. These are adaptive development data, "
                "not confirmatory outcomes; higher-order future scores and labels remain unavailable.\n"
                + json.dumps(feedback)
            )
        outcome = await super().cycle(index, profile, state, context)
        folder = self.store.root / "episodes" / f"cycle_{index:03d}"
        write_json(folder / "development_feedback.json", feedback)
        outcome["development_feedback_sha256"] = digest(feedback)
        write_json(folder / "outcome.json", outcome)
        return outcome

    def acceptance_policy(self, state):
        return policy(self.scope, self.data["training"], self.data["validation"])


def prepare(args):
    if not args.profile or not args.objective or not args.admission or not args.features or not args.baseline:
        raise ValueError("Create requires profile, explicit objective and verified public source paths")
    profile = Profile.from_dict(read(args.profile))
    scope, train, val, future, source, split = prepare_reap_model_case(
        args.admission, args.features, args.baseline, args.family, args.seed
    )
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
        condition=args.condition,
    )
    (store.root / "data").mkdir()
    for name, rows in [("training", train), ("validation", val), ("future", future)]:
        write_json(store.root / "data" / (name + ".json"), rows)
    paths = [
        Path(__file__),
        *(
            ROOT / "scripts/rsi" / name
            for name in [
                "run_science_goal.py",
                "run_reap_generated_matrix.py",
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
            "case": "reap",
            "scope": scope,
            "source": source,
            "split": split,
            "model_plugin_id": PLUGIN,
            "model_plugin_sha256": get_model_plugin(PLUGIN).source_sha256,
            "implementation": hashes,
            "data_hashes": {
                n: digest(rows) for n, rows in [("training", train), ("validation", val), ("future", future)]
            },
            "development_prompt_sha256": digest(development_prompt(scope)),
            "instructions_sha256": digest(SYSTEM),
            "acceptance_role": "Single-mutant acceptance is adaptive development selection; the last two independently replayed sources and aggregate acceptance metrics feed the next development episode. No future feedback or unbiased replicate claim.",
            "future_policy": "Score higher-order mutants only after the whole goal is verified; no future-label episode feedback",
            "progress_policy": "Retain independently accepted model across failed reuse episodes; all failures and physical calls consume the common goal budget",
            "scope_note": "Public retrospective feature-model goal integration; persistent-Agent is an explicit research ablation. Not full product-mode superiority, new biological discovery or RSI efficacy.",
        },
    )
    return store


def publish_completed_source(store, plugin, state, *, summary=None):
    if state["status"] != "completed" or not all(c["passed"] for c in plugin.evaluate(state)["criteria"].values()):
        raise ValueError("Verified complete goal required before experience publication")
    lc, mid = plugin.lifecycle(state)
    catalog = ModelExperienceStore(store.root / "experience")
    identity = catalog.publish(
        lc,
        mid,
        summary=summary
        or "Autonomously generated feature-model source independently accepted and reused in a fresh goal episode. Target-specific acceptance does not imply transferable efficacy.",
    )
    packet = {
        "experience_id": identity,
        "model_id": mid,
        "source_sha256": lc._model(mid)["source_sha256"],
        "projection": "Source-only; no fitted weights or future labels",
    }
    path = store.root / "registered_experience.json"
    if path.exists():
        if read(path) != packet:
            raise ValueError("Registered goal experience changed")
    else:
        write_json(path, packet)
    return packet


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
        plugin = ReapModelGoal(store, resolve_credential(profile.model, args.credential_toml))
        state = await run_goal(store, plugin.cycle, plugin.evaluate)
        if state["status"] == "completed" and not (store.root / "future_metrics.json").exists():
            lc, mid = plugin.lifecycle(state)
            write_json(
                store.root / "future_metrics.json",
                get_model_plugin(PLUGIN).evaluate(lc._model(mid), plugin.data["future"], plugin.scope),
            )
        if state["status"] == "completed":
            publish_completed_source(store, plugin, state)
    print(state["status"], len(state["cycles"]), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["create", "run", "pause", "resume", "status"])
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--profile", type=Path)
    p.add_argument("--objective")
    p.add_argument("--admission", type=Path)
    p.add_argument("--features", type=Path)
    p.add_argument("--baseline", type=Path)
    p.add_argument("--family", choices=FAMILIES, default=FAMILIES[0])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--condition", choices=["science_goal", "persistent_agent_ablation"], default="science_goal")
    p.add_argument("--credential-toml", type=Path)
    p.add_argument("--max-cycles", type=int, default=4)
    p.add_argument("--model-calls", type=int, default=72)
    p.add_argument("--tool-calls", type=int, default=36)
    p.add_argument("--wall-seconds", type=float, default=2400)
    asyncio.run(main(p.parse_args()))
