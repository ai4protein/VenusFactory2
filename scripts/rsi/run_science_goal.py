"""Science goal entry: expert-team episodes persist across development and reuse.

This concrete domain plugin demonstrates goal continuation. It is a bounded
engineering run on prior case partitions, not a long-horizon efficacy study.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from run_generated_lifecycle import SYSTEM, policy_for, spec
from run_harness_probe import ROOT, credential_from_toml, default_runtimes
from run_paired_generated import make_bundle
from run_protein_lifecycle import audit_events, prepare_case

from agent.harness import Profile, Task
from agent.harness.contracts import digest, file_hash
from agent.harness.host import execute, write_json
from agent.rsi import context_feedback_dispatch as context_feedback
from agent.rsi import library_experience, model_tasks
from agent.rsi import training_attempt_memory as attempt_memory
from agent.rsi.datasets.library_source import REFIT_TOOL
from agent.rsi.engine import prediction_batches
from agent.rsi.engine.model_lifecycle import ModelLifecycle
from agent.rsi.engine.model_plugins import get_model_plugin
from agent.rsi.engine.provider_access import access_wait
from agent.science_goal import GoalSpec, GoalStore, run_goal


def read(path):
    return json.loads(path.read_text())


def evidence(store, path):
    return {"path": str(path.relative_to(store.root)), "sha256": file_hash(path)}


class GeneratedModelGoal:
    def __init__(self, store, credential):
        self.store = store
        self.credential = credential
        self.protocol = read(store.root / "protocol.json")
        prediction_batches.verify_goal(store.read(), self.protocol)
        attempt_memory.verify_goal(store.read(), self.protocol)
        context_feedback.verify_goal(store.read(), self.protocol)
        self.scope = self.protocol["scope"]
        self.data = {n: read(store.root / "data" / (n + ".json")) for n in ["training", "validation", "future"]}
        if "model_task" in self.protocol:
            definition = read(store.root / "model_task.json")
            if (
                model_tasks.admission(definition) != self.protocol["model_task"]
                or store.read()["definition"].get("admission") != self.protocol["model_task"]
                or definition["plugin_id"] != self.model_plugin_id
                or definition["scope"] != self.scope
                or definition["minimum_skill"] != self.protocol["model_task_minimum_skill"]
                or model_tasks.load_task(definition) != self.data
            ):
                raise ValueError("Model task admission changed")
        for n, rows in self.data.items():
            if digest(rows) != self.protocol["data_hashes"][n]:
                raise ValueError("Goal data changed")
        for name, sha in self.protocol["implementation"].items():
            if file_hash(ROOT / name) != sha:
                raise ValueError("Goal implementation changed; explicit version migration required")

        self.library_experience = library_experience.verify_goal(store.root, store.read(), self.protocol)

    @property
    def model_plugin_id(self):
        return self.protocol.get("model_plugin_id", "generated_python.v1")

    def inference_inputs(self):
        return model_tasks.inference_inputs(self.data["future"], self.model_plugin_id)

    def development_action(self):
        return "In this episode, inspect the supplied protein task, author deterministic numerical fit/predict source code, evaluate using training-only feedback, and submit one actually tested candidate. Independent controller acceptance follows this episode."

    def lifecycle(self, state):
        record = state["progress"].get("accepted_model")
        if not record:
            return None, None
        root = (self.store.root / record["controller"]).resolve()
        root.relative_to(self.store.root)
        lc = ModelLifecycle(root, read(root / "policy.json"))
        prediction_batches.require_bound_policy(lc.policy, self.protocol)
        mid = record["model_id"]
        lc._model(mid)
        return lc, mid

    def evaluate(self, state):
        checks = {n: {"passed": False, "evidence": []} for n in state["definition"]["goal"]["criteria"]}
        lc, mid = self.lifecycle(state)
        if lc:
            current = lc._state()
            report = lc._verified_assessment(mid, current, lc._model(mid))
            passed = mid in current["published"] and report["passed"] and report["training_replayed"]
            if digest(report) != current["candidates"][mid]["assessment_sha256"]:
                raise ValueError("Acceptance evidence changed")
            checks["accepted_model"] = {
                "passed": bool(passed),
                "evidence": [evidence(self.store, lc.root / ("assessment_" + mid + ".json"))] if passed else [],
            }
            receipt_name = state["progress"].get("reuse_receipt")
            if receipt_name:
                path = (self.store.root / receipt_name).resolve()
                path.relative_to(self.store.root)
                receipt = read(path)
                passed = bool(
                    passed
                    and receipt in current["events"]
                    and receipt["event"] == "reused"
                    and receipt["session_id"] != current["candidates"][mid]["session_id"]
                )
                checks["fresh_session_reuse"] = {
                    "passed": passed,
                    "evidence": [evidence(self.store, path)] if passed else [],
                }
        return {"criteria": checks}

    def development_experience(self, state):
        return None

    def acceptance_policy(self, state):
        if "model_task" in self.protocol:
            policy = model_tasks.policy_for_task(read(self.store.root / "model_task.json"), self.data)
        else:
            policy = policy_for(self.scope, self.data["training"], self.data["validation"])
        return prediction_batches.apply_policy(policy, self.protocol)

    async def cycle(self, index, profile, state, context):
        folder = self.store.root / "episodes" / f"cycle_{index:03d}"
        work = folder / "workspace"
        work.mkdir(parents=True, exist_ok=False)
        lc, mid = self.lifecycle(state)
        phase = "reuse" if lc else "develop"
        memory = None
        if attempt_memory.policy_for(self.protocol) != "none":
            memory = attempt_memory.project(self.store.root, state, self.protocol, phase)
            write_json(folder / "attempt_memory.json", memory)
            context += attempt_memory.context(memory)
        inputs = {"inputs.json": self.inference_inputs()} if lc else {"training.json": self.data["training"]}
        for name, rows in inputs.items():
            write_json(work / name, rows)
        versions, version, capability = make_bundle(
            folder / "capability",
            phase,
            inputs,
            self.scope,
            profile.public()["model"],
            lc,
            mid,
            experience=self.development_experience(state) if phase == "develop" else None,
            library_experience=self.library_experience if phase == "develop" else None,
            plugin_id=self.model_plugin_id,
            training_baseline_feedback=context_feedback.policy_for(self.protocol) if phase == "develop" else "none",
        )
        declaration = spec(phase)
        if phase == "develop" and self.development_experience(state) is not None:
            declaration["tools"].add("read_model_experience")
            declaration["required_tools"].add("read_model_experience")
        if phase == "develop" and self.library_experience is not None:
            current = library_experience.read_frozen(self.store.root, state, self.protocol)
            if current["projection"] != self.library_experience:
                raise ValueError("Admitted library experience changed during execution")
            declaration["tools"].add(library_experience.TOOL)
            declaration["tools"].add(REFIT_TOOL)
            declaration["recoverable_tools"].add(REFIT_TOOL)
            declaration["required_tools"].add(library_experience.TOOL)
            context += "\nCall read_model_library_experience. Treat any donor implementation as a precedent to adapt, refit and independently validate for this task. The admitted control condition may provide no prior implementations."
            context += " If a suitable source is admitted, evaluate_library_model_code(experience_id, rationale) refits it unchanged on current training data without retyping code or using donor weights. For an adaptation or new implementation use evaluate_model_code. Both routes share two training-only attempts and require submit_model_code followed by independent target acceptance."
        action = (
            "In this episode, discover the accepted model and use it to predict every supplied unseen input. Do not redevelop the model."
            if lc
            else self.development_action()
        )
        task = Task(
            work,
            capability,
            folder / "execution",
            prediction_batches.append_prompt(context + "\n\nNext goal stage: " + action, self.protocol, phase),
            SYSTEM,
            frozenset(declaration["tools"]),
            phase=phase,
        )
        if memory is not None:
            write_json(folder / "attempt_prompt.json", task.prompt)
        run = await execute(profile, task, default_runtimes(), credential=self.credential)
        audit = audit_events(
            task.output,
            phase,
            run,
            declaration["required_tools"],
            declaration["recoverable_tools"],
            declaration.get("retryable_read_tools", set()),
        )
        write_json(folder / "lifecycle_execution_audit.json", audit)
        versions.verify(version)
        progress = dict(state["progress"])
        outcome = {
            "status": "failed",
            "summary": phase + " episode did not pass its execution/evidence gate",
            "phase": phase,
            "run_record": str((task.output / "run.json").relative_to(self.store.root)),
            "execution_audit_sha256": digest(audit),
            "usage": {
                "model_calls": run["audit"]["requests"]["model_calls"],
                "tool_calls": run["audit"]["events"]["tool_calls"],
                "wall_seconds": run["host_wall_seconds"],
            },
            "progress": progress,
        }
        if memory is not None:
            outcome["attempt_memory_sha256"] = digest(memory)
            if phase == "develop":
                attempts = attempt_memory.capture(folder, self.protocol)
                write_json(folder / "attempt_evidence.json", attempts)
                outcome["attempt_evidence_sha256"] = digest(attempts)
        if run["status"] == "completed" and audit["passed"]:
            if phase == "develop":
                policy = prediction_batches.apply_policy(self.acceptance_policy(state), self.protocol)
                lc = ModelLifecycle(folder / "controller", policy)
                mid = lc.submit(work / "candidate.json", session_id=run["session_id"], mode=profile.mode)
                accepted = lc.assess(mid, training=self.data["training"], validation=self.data["validation"])
                write_json(folder / "acceptance.json", accepted)
                if accepted["passed"]:
                    progress["accepted_model"] = {
                        "controller": str(lc.root.relative_to(self.store.root)),
                        "model_id": mid,
                    }
                    outcome.update(
                        status="completed",
                        summary="Candidate independently accepted; goal still requires a fresh-session reuse episode",
                    )
                else:
                    outcome.update(
                        status="candidate_rejected",
                        rejection_reasons=accepted.get("rejection_reasons", []),
                        summary="Candidate failed independent development acceptance; goal remains unmet",
                    )
            else:
                receipt = lc.record_reuse(
                    mid,
                    session_id=run["session_id"],
                    mode=profile.mode,
                    inputs=inputs["inputs.json"],
                    predictions=read(work / "predictions.json"),
                    scope=self.scope,
                )
                path = folder / "reuse_receipt.json"
                write_json(path, receipt)
                progress["reuse_receipt"] = str(path.relative_to(self.store.root))
                outcome.update(
                    status="completed", summary="Accepted model independently replayed in a fresh execution session"
                )
        if outcome["status"] != "completed":
            outcome.update(access_wait(task.output, self.store.root))
        write_json(folder / "outcome.json", outcome)
        return outcome


def prepare(args):
    if not args.profile or not args.objective:
        raise ValueError("Create requires a profile and explicit objective")
    memory_policy = getattr(args, "attempt_memory", attempt_memory.POLICY)
    attempt_memory.policy_for({"training_attempt_memory": memory_policy})
    profile = Profile.from_dict(read(args.profile))
    task_definition = None
    if getattr(args, "task_definition", None):
        task_definition, task_data = model_tasks.read_task(args.task_definition)
        scope = task_definition["scope"]
        train, val, future = (task_data[p] for p in model_tasks.PHASES)
        source, split = model_tasks.source_and_split(task_definition)
    else:
        scope, train, val, future, source, split = prepare_case(args.case, args.source_cache)
    feedback_policy = context_feedback.policy_for(
        {context_feedback.FIELD: getattr(args, "training_baseline_feedback", "none")},
        plugin_id=task_definition["plugin_id"] if task_definition else "generated_python.v1",
    )
    transfer = None
    if getattr(args, "experience_selection", None) is not None:
        from agent.rsi.engine.scientist_model_library import ScientistModelLibrary

        if task_definition is None:
            raise ValueError("Library transfer requires an admitted model task")
        request = library_experience.selection(read(args.experience_selection))
        ScientistModelLibrary(request["library_root"]).goal(args.output)
        transfer = library_experience.snapshot(request, task_definition)
    goal = GoalSpec(
        args.objective,
        ("accepted_model", "fresh_session_reuse"),
        max_cycles=args.max_cycles,
        model_calls=args.model_calls,
        tool_calls=args.tool_calls,
        wall_seconds=args.wall_seconds,
    )
    extra = {"admission": model_tasks.admission(task_definition)} if task_definition else {}
    store = GoalStore.create(args.output, goal, profile, condition=getattr(args, "condition", "science_goal"), **extra)
    review = prediction_batches.freeze_goal(store)
    if memory_policy == attempt_memory.CUMULATIVE_POLICY:

        def bind_memory(state):
            state["definition"][attempt_memory.FIELD] = memory_policy
            state["definition_sha256"] = digest(state["definition"])

        store.mutate(bind_memory)
    if feedback_policy != "none":

        def bind_feedback(state):
            state["definition"][context_feedback.FIELD] = feedback_policy
            state["definition_sha256"] = digest(state["definition"])

        store.mutate(bind_feedback)
    if task_definition:
        write_json(store.root / "model_task.json", task_definition)
    (store.root / "data").mkdir()
    for name, rows in [("training", train), ("validation", val), ("future", future)]:
        write_json(store.root / "data" / (name + ".json"), rows)
    paths = [
        Path(__file__),
        ROOT / "src/agent/science_goal.py",
        ROOT / "configs/rsi/model_access.json",
        ROOT / "src/agent/research_modes.py",
        ROOT / "scripts/rsi/run_generated_lifecycle.py",
        ROOT / "scripts/rsi/run_paired_generated.py",
        *sorted((ROOT / "src/agent/harness").glob("*.py")),
        *sorted((ROOT / "src/agent/rsi").rglob("*.py")),
        *sorted((ROOT / "src/agent/graph").rglob("*.py")),
    ]
    hashes = {str(p.relative_to(ROOT)): file_hash(p) for p in paths}
    for path in paths:
        dst = store.root / "implementation" / path.relative_to(ROOT)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(path.read_bytes())
    transfer_binding = library_experience.freeze(store.root, transfer) if transfer is not None else None
    if transfer_binding is not None:

        def bind_transfer(state):
            state["definition"]["library_experience"] = transfer_binding
            state["definition_sha256"] = digest(state["definition"])

        store.mutate(bind_transfer)
    write_json(
        store.root / "protocol.json",
        {
            "scope": scope,
            prediction_batches.FIELD: review,
            "source": source,
            "split": split,
            "case": task_definition["id"] if task_definition else args.case,
            **(
                {
                    "model_task": model_tasks.admission(task_definition),
                    "model_plugin_id": task_definition["plugin_id"],
                    "model_task_minimum_skill": task_definition["minimum_skill"],
                }
                if task_definition
                else {}
            ),
            "data_hashes": {
                n: digest(rows) for n, rows in [("training", train), ("validation", val), ("future", future)]
            },
            "implementation": hashes,
            "training_attempt_memory": memory_policy,
            **({context_feedback.FIELD: feedback_policy} if feedback_policy != "none" else {}),
            **({"library_experience": transfer_binding} if transfer_binding is not None else {}),
            "scope_note": "Goal-continuation engineering test on prior partitions; no confirmatory efficacy inference",
            "future_policy": "Only score after the complete goal is verified; never use as episode feedback",
        },
    )
    return store


async def main(args):
    from agent.rsi.engine.scientist_model_library import ScientistModelLibrary

    library_root = getattr(args, "model_library", None)
    publication_library = ScientistModelLibrary(library_root) if library_root is not None else None
    if publication_library is not None:
        publication_library.goal(args.output)
    store = prepare(args) if args.action == "create" else GoalStore(args.output)
    if args.action == "pause":
        state = store.pause()
    elif args.action in {"create", "status"}:
        state = store.read()
    else:
        if args.action == "resume":
            store.resume()
        profile = Profile.from_dict(store.read()["definition"]["profile"])
        import os

        key = (
            credential_from_toml(args.credential_toml, profile.model)
            if args.credential_toml
            else os.environ.get(profile.model.credential_env)
        )
        if not key:
            raise ValueError("Goal model credential is unresolved")
        plugin = GeneratedModelGoal(store, key)
        state = await run_goal(store, plugin.cycle, plugin.evaluate)
        if state["status"] == "completed" and not (store.root / "future_metrics.json").exists():
            lc, mid = plugin.lifecycle(state)
            metrics = get_model_plugin(plugin.model_plugin_id).evaluate(
                lc._model(mid), plugin.data["future"], plugin.scope
            )
            write_json(store.root / "future_metrics.json", metrics)
    if publication_library is not None and args.action in {"run", "resume"} and state["status"] == "completed":
        publication = publication_library.publish_goal(store.root)
        if not publication["all_published"]:
            raise ValueError("Model publication failed; retained report " + publication["report_id"])
        print(json.dumps({"model_publication": publication}), flush=True)
    print(
        json.dumps(
            {
                "status": state["status"],
                "episodes": len(state["cycles"]),
                "usage": state["usage"],
                "criteria": state["assessment"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["create", "run", "pause", "resume", "status"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--objective")
    parser.add_argument("--task-definition", type=Path, help="Host-admitted source-bound protein modelling task")
    parser.add_argument("--model-library", type=Path, help="Publish verified models after a completed run")
    parser.add_argument(
        "--experience-selection",
        type=Path,
        help="Frozen owner-library donor selection for none/summary/source transfer",
    )
    parser.add_argument("--case", choices=["alde_parpgb", "mcba_itopride"], default="alde_parpgb")
    parser.add_argument("--attempt-memory", choices=attempt_memory.POLICIES, default=attempt_memory.POLICY)
    parser.add_argument("--training-baseline-feedback", choices=context_feedback.POLICIES, default="none")
    parser.add_argument("--max-cycles", type=int, default=4)
    parser.add_argument("--model-calls", type=int, default=72)
    parser.add_argument("--tool-calls", type=int, default=36)
    parser.add_argument("--wall-seconds", type=float, default=2400)
    parser.add_argument("--credential-toml", type=Path)
    parser.add_argument("--source-cache", type=Path, default=ROOT / "research_runs/rsi/20261011_paper_case_audit")
    asyncio.run(main(parser.parse_args()))
