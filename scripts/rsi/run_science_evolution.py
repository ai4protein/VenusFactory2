"""Goal-driven code lineage: independently accepted generations and fresh reuse.

Repeated development validation is a selection set, not an unbiased efficacy
estimate. Future labels remain sealed until the entire goal is complete.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json

from run_harness_probe import ROOT, credential_from_toml
from run_science_goal import GeneratedModelGoal, evidence, prepare, read

from agent.harness import Profile
from agent.harness.contracts import file_hash
from agent.harness.host import write_json
from agent.rsi.engine.experience import ModelExperienceStore
from agent.rsi.engine.lineage_promotion import gain_error, new_evolution, validate_evolution
from agent.rsi.engine.model_plugins import get_model_plugin
from agent.rsi.datasets.source_lineage import program_hash
from agent.rsi.engine.training_attempt_memory import POLICIES, POLICY
from agent.science_goal import GoalStore, run_goal


class EvolvingModelGoal(GeneratedModelGoal):
    def __init__(self, store, credential):
        super().__init__(store, credential)
        validate_evolution(self.protocol)

    def promotion_gain_error(self, lc, mid, records):
        previous = self.protocol.get("model_task_minimum_skill", 0.0)
        if records:
            parent, identity = self.lifecycle({"progress": {"accepted_model": records[-1]}})
            previous = read(parent.root / ("assessment_" + identity + ".json"))["metrics"][self.acceptance_metric]
        current = read(lc.root / ("assessment_" + mid + ".json"))["metrics"][self.acceptance_metric]
        return gain_error(self.protocol, previous, current)

    @property
    def target(self):
        return self.protocol["evolution"]["generations"]

    @property
    def acceptance_metric(self):
        return get_model_plugin(self.model_plugin_id).acceptance_metric

    def promotion_error(self, model, records):
        if self.protocol["evolution"].get("schema", 1) < 2:
            return None
        candidate = program_hash(model["recipe"]["source_code"])
        for record in records:
            lc, mid = self.lifecycle({"progress": {"accepted_model": record}})
            previous = lc._model(mid)
            if candidate == program_hash(previous["recipe"]["source_code"]):
                return "Executable source is unchanged from an accepted generation"
        return None

    def lineage_parents(self, records):
        return [records[-1]["experience_id"]] if records else []

    def generation_context(self, generation):
        return (
            "\nGeneration "
            + str(generation)
            + " of "
            + str(self.target)
            + ". Call read_model_experience, discuss the previous implementation and propose a justified change. "
            "Use training-only feedback to test it. Preserve working capabilities. Independent development "
            "selection must strictly improve over the preceding generation before the host registers a successor. "
            "Do not claim algorithmic novelty or future-set improvement. If a previous attempt failed, use its checkpoint to revise your approach."
        )

    def experience(self):
        return ModelExperienceStore(self.store.root / "experience")

    def development_experience(self, state):
        return self.experience().project(
            "source", [g["experience_id"] for g in state["progress"].get("generations", [])]
        )

    def acceptance_policy(self, state):
        policy = super().acceptance_policy(state)
        previous = state["progress"].get("generations", [])
        if previous:
            lc, mid = self.lifecycle({"progress": {"accepted_model": previous[-1]}})
            report = read(lc.root / ("assessment_" + mid + ".json"))
            # A new generation must strictly improve the development selection metric.
            policy["minimum_skill"] = max(0.0, report["metrics"][self.acceptance_metric])
        return policy

    def evaluate(self, state):
        records = state["progress"].get("generations", [])
        accepted, reused, lineage = [], [], []
        parents = []
        previous_skill = self.protocol.get("model_task_minimum_skill", 0.0)
        seen_models = set()
        for record in records:
            checks = super().evaluate(
                {
                    **state,
                    "definition": {
                        **state["definition"],
                        "goal": {**state["definition"]["goal"], "criteria": ["accepted_model", "fresh_session_reuse"]},
                    },
                    "progress": {"accepted_model": record, "reuse_receipt": record.get("reuse_receipt")},
                }
            )["criteria"]
            if not checks["accepted_model"]["passed"]:
                raise ValueError("A lineage member lost its acceptance evidence")
            lc, mid = self.lifecycle({"progress": {"accepted_model": record}})
            report = read(lc.root / ("assessment_" + mid + ".json"))
            skill = report["metrics"][self.acceptance_metric]
            if (
                mid in seen_models
                or not skill > previous_skill
                or lc.policy["minimum_skill"] != previous_skill
                or gain_error(self.protocol, previous_skill, skill)
            ):
                raise ValueError("Generation promotion is not an independently verified improvement")
            model = lc._model(mid)
            if self.protocol["evolution"].get("schema", 1) >= 2:
                if (
                    record["generation"] != len(seen_models) + 1
                    or record["source_sha256"] != model["source_sha256"]
                    or record["program_sha256"] != program_hash(model["recipe"]["source_code"])
                    or self.promotion_error(model, records[: len(seen_models)])
                ):
                    raise ValueError("Generation source lineage changed or duplicates an earlier implementation")
            seen_models.add(mid)
            previous_skill = skill
            original = self.experience()._entry(record["experience_id"])
            if original["parents"] != parents or original["model_id"] != record["model_id"]:
                raise ValueError("Invalid generation ancestry")
            accepted += checks["accepted_model"]["evidence"]
            if checks["fresh_session_reuse"]["passed"]:
                reused += checks["fresh_session_reuse"]["evidence"]
            lineage.append(
                evidence(self.store, self.store.root / "experience/entries" / (record["experience_id"] + ".json"))
            )
            parents = self.lineage_parents(records[: len(seen_models)])
        enough = len(records) >= self.target
        return {
            "criteria": {
                "accepted_model": {"passed": enough, "evidence": accepted if enough else []},
                "fresh_session_reuse": {
                    "passed": enough and len(reused) == len(records),
                    "evidence": reused if enough and len(reused) == len(records) else [],
                },
                "verified_lineage": {"passed": enough, "evidence": lineage if enough else []},
            }
        }

    async def cycle(self, index, profile, state, context):
        local = copy.deepcopy(state)
        progress = local["progress"]
        records = progress.setdefault("generations", [])
        developing = not records or bool(records[-1].get("reuse_receipt"))
        generation = len(records) + 1 if developing else len(records)
        if developing:
            progress.pop("accepted_model", None)
            progress.pop("reuse_receipt", None)
            context += self.generation_context(generation)
        else:
            progress["accepted_model"] = records[-1]
            progress.pop("reuse_receipt", None)
        outcome = await super().cycle(index, profile, local, context)
        progress = outcome["progress"]
        records = progress["generations"]
        outcome["generation"] = generation
        if outcome["status"] == "completed":
            if developing:
                lc, mid = self.lifecycle({"progress": progress})
                model = lc._model(mid)
                reason = self.promotion_error(model, records) or self.promotion_gain_error(lc, mid, records)
                if reason:
                    lc.retire(mid, reason)
                    progress.pop("accepted_model", None)
                    progress.pop("reuse_receipt", None)
                    outcome.update(status="candidate_rejected", summary=reason)
                    write_json(self.store.root / "episodes" / f"cycle_{index:03d}" / "outcome.json", outcome)
                    return outcome
                provenance = read(
                    self.store.root / "episodes" / f"cycle_{index:03d}" / "workspace/source_provenance.json"
                )
                identity = self.experience().publish(
                    lc,
                    mid,
                    summary=provenance["rationale"][:6000] or "Independently tested generated source",
                    parents=self.lineage_parents(records),
                )
                record = {
                    **progress["accepted_model"],
                    "generation": generation,
                    "selection_predecessor": records[-1]["experience_id"] if records else None,
                    "experience_id": identity,
                    "source_sha256": model["source_sha256"],
                    "program_sha256": program_hash(model["recipe"]["source_code"]),
                }
                records.append(record)
                outcome["summary"] = (
                    f"Generation {generation} independently accepted and registered; fresh-session reuse is next"
                )
            else:
                records[-1]["reuse_receipt"] = progress["reuse_receipt"]
                outcome["summary"] = f"Generation {generation} reused and independently replayed; " + (
                    "next generation follows the frozen history policy"
                    if generation < self.target
                    else "all requested generations now have reuse evidence"
                )
        write_json(self.store.root / "episodes" / f"cycle_{index:03d}" / "outcome.json", outcome)
        return outcome


async def main(args):
    from agent.rsi.engine.scientist_model_library import ScientistModelLibrary

    library_root = getattr(args, "model_library", None)
    publication_library = ScientistModelLibrary(library_root) if library_root is not None else None
    if publication_library is not None:
        publication_library.goal(args.output)
    if args.action == "create":
        if args.max_cycles < 2 * args.generations:
            raise ValueError("Each generation requires development and independent reuse episodes")
        store = prepare(args)

        # Freeze the additional driver and criteria before any execution.
        def configure(state):
            from agent.harness.contracts import digest

            state["definition"]["goal"]["criteria"].append("verified_lineage")
            state["definition_sha256"] = digest(state["definition"])

        store.mutate(configure)
        protocol = read(store.root / "protocol.json")
        source = ROOT / "scripts/rsi/run_science_evolution.py"
        protocol["implementation"][str(source.relative_to(ROOT))] = file_hash(source)
        archived = store.root / "implementation" / source.relative_to(ROOT)
        archived.write_bytes(source.read_bytes())
        protocol["evolution"] = new_evolution(args.generations)
        write_json(store.root / "protocol.json", protocol)
    else:
        store = GoalStore(args.output)
    if args.action == "pause":
        state = store.pause()
    elif args.action in {"create", "status"}:
        state = store.read()
    else:
        if args.action == "resume":
            store.resume()
        profile = Profile.from_dict(store.read()["definition"]["profile"])
        key = credential_from_toml(args.credential_toml, profile.model)
        if not key:
            raise ValueError("Goal credential unresolved")
        plugin = EvolvingModelGoal(store, key)
        from agent.rsi.engine.generation_publication import publisher

        options = {"on_checkpoint": publisher(publication_library, store.root)} if publication_library else {}
        state = await run_goal(store, plugin.cycle, plugin.evaluate, **options)
        if state["status"] == "completed":
            metrics = []
            for record in state["progress"]["generations"]:
                lc, mid = plugin.lifecycle({"progress": {"accepted_model": record}})
                metrics.append(
                    {
                        "generation": record["generation"],
                        "model_id": mid,
                        "metrics": get_model_plugin(plugin.model_plugin_id).evaluate(
                            lc._model(mid), plugin.data["future"], plugin.scope
                        ),
                    }
                )
            write_json(store.root / "future_metrics.json", metrics)
    if publication_library is not None and args.action in {"run", "resume"} and state["status"] == "completed":
        publication = publication_library.publish_goal(store.root)
        if not publication["all_published"]:
            raise ValueError("Model publication failed; retained report " + publication["report_id"])
        print(json.dumps({"model_publication": publication}), flush=True)
    print(state["status"], len(state["cycles"]), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["create", "run", "pause", "resume", "status"])
    from pathlib import Path

    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--objective")
    parser.add_argument("--task-definition", type=Path, help="Host-admitted source-bound protein modelling task")
    parser.add_argument(
        "--model-library", type=Path, help="Publish each independently verified generation after fresh-session reuse"
    )
    parser.add_argument(
        "--experience-selection",
        type=Path,
        help="Frozen owner-library donor selection for none/summary/source transfer",
    )
    parser.add_argument("--case", choices=["alde_parpgb", "mcba_itopride"], default="alde_parpgb")
    parser.add_argument("--generations", type=int, choices=[2, 3], default=2)
    parser.add_argument("--attempt-memory", choices=POLICIES, default=POLICY)
    parser.add_argument("--max-cycles", type=int, default=8)
    parser.add_argument("--model-calls", type=int, default=192)
    parser.add_argument("--tool-calls", type=int, default=96)
    parser.add_argument("--wall-seconds", type=float, default=7200)
    parser.add_argument("--credential-toml", type=Path)
    parser.add_argument("--source-cache", type=Path, default=ROOT / "research_runs/rsi/20261011_paper_case_audit")
    asyncio.run(main(parser.parse_args()))
