"""Persistent native Scientist goals for independently qualified shared programs.

One expert-team execution can develop several target-specific fitted models.
Every admitted task must pass its own acceptance and fresh reuse criteria.
Publication requires the explicit model-library option and native provenance audit.
Final labels are not scored by this entry.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import math
import os
import time
from dataclasses import replace
from pathlib import Path

from run_generated_lifecycle import SYSTEM, spec
from run_harness_probe import ROOT, credential_from_toml, default_runtimes
from run_paired_generated import make_bundle
from run_protein_lifecycle import audit_events

from agent.harness import Profile, Task
from agent.harness.contracts import digest, file_hash
from agent.harness.host import execute
from agent.rsi import model_tasks, shared_library_experience
from agent.rsi.engine.context_feedback_dispatch import policy_for as feedback_policy
from agent.rsi.engine.model_access import authorize_profile
from agent.rsi.engine.model_lifecycle import ModelLifecycle
from agent.rsi.engine.model_plugins import get_model_plugin
from agent.rsi.engine.provider_access import access_wait
from agent.rsi.engine.shared_goal_history import POLICY as HISTORY_POLICY
from agent.rsi.engine.shared_goal_history import SUPPORTED_POLICIES
from agent.rsi.engine.shared_goal_history import make_prompt as make_shared_prompt
from agent.rsi.engine.shared_goal_history import project as project_shared_history
from agent.rsi.engine.shared_model_bundle import admit, prepare_bundle
from agent.rsi.engine.source_snapshot import research_implementation_paths
from agent.rsi.engine.storage import atomic_json, safe_child
from agent.rsi.engine.tool_contracts import validate_generated_result
from agent.science_goal import GoalSpec, GoalStore, run_goal

SCHEMA = "shared_scientist_goal.v1"
CRITERIA = ("all_targets_accepted", "all_targets_freshly_reused")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    require(
        path.is_file() and not any(p.is_symlink() for p in [path, *path.parents]),
        "Missing or symlinked shared-goal evidence",
    )
    value = json.loads(path.read_text())
    digest(value)
    return value


def evidence(root, path):
    return {"path": path.relative_to(root).as_posix(), "sha256": file_hash(path)}


def execution_profile(profile, remaining_seconds):
    """Keep evidence finalization inside the existing outer cycle reservation."""
    if type(remaining_seconds) not in (int, float) or not math.isfinite(remaining_seconds):
        raise ValueError("Invalid remaining shared-cycle time")
    if remaining_seconds <= 0:
        raise TimeoutError("Shared preparation exhausted its cycle reservation")
    available = min(profile.budget.wall_seconds, remaining_seconds)
    finalization = min(30.0, 0.1 * available)
    execution_seconds = available - finalization
    return replace(
        profile,
        budget=replace(
            profile.budget,
            wall_seconds=execution_seconds,
            request_seconds=min(profile.budget.request_seconds, execution_seconds),
        ),
    )


def prepare(
    output,
    tasks,
    profile,
    goal,
    *,
    training_baseline_feedback="none",
    shared_training_history=HISTORY_POLICY,
    library_selection=None,
):
    require(shared_training_history in SUPPORTED_POLICIES, "Unsupported shared training history")
    require(tuple(goal.criteria) == CRITERIA, "Shared goal must qualify and freshly reuse every target")
    require(profile.mode == "science_expert", "Shared goals require Scientist mode")
    data = admit(tasks)
    for task in tasks:
        feedback_policy(
            {"training_baseline_feedback": training_baseline_feedback, "scope": task["scope"]},
            plugin_id=task["plugin_id"],
        )
    historical = shared_library_experience.snapshot(library_selection, tasks) if library_selection is not None else None
    implementation = {}
    paths = research_implementation_paths(ROOT)
    for path in paths:
        require(not path.is_symlink(), "Symlinked goal implementation")
        implementation[path.relative_to(ROOT).as_posix()] = file_hash(path)
    admission = {
        "schema": SCHEMA,
        "tasks": [model_tasks.admission(task) for task in tasks],
        "training_baseline_feedback": training_baseline_feedback,
        "implementation_sha256": digest(implementation),
        "shared_training_history": shared_training_history,
    }
    if historical is not None:
        admission[shared_library_experience.FIELD] = shared_library_experience.binding(historical)
    store = GoalStore.create(output, goal, profile, admission=admission)
    if historical is not None:
        require(
            shared_library_experience.freeze(store.root, historical) == admission[shared_library_experience.FIELD],
            "Shared experience freeze changed",
        )
    declarations = []
    for task in tasks:
        task_id = task["id"]
        folder = store.root / "targets" / task_id
        atomic_json(folder / "model_task.json", task)
        for phase, rows in data[task_id].items():
            atomic_json(folder / "data" / (phase + ".json"), rows)
        source, split = model_tasks.source_and_split(task)
        declarations.append(
            {
                "id": task_id,
                "task_sha256": digest(task),
                "data_hashes": {p: digest(rows) for p, rows in data[task_id].items()},
                "source": source,
                "split": split,
            }
        )
    for path in paths:
        name = path.relative_to(ROOT).as_posix()
        target = store.root / "implementation" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
        require(
            file_hash(target) == implementation[name] == file_hash(path), "Goal implementation changed while preparing"
        )
    atomic_json(
        store.root / "protocol.json",
        {
            "schema": SCHEMA,
            **(
                {shared_library_experience.FIELD: admission[shared_library_experience.FIELD]}
                if historical is not None
                else {}
            ),
            "admission": admission,
            "targets": declarations,
            "implementation": implementation,
            "target_training_baseline_feedback": training_baseline_feedback,
            "training_attempt_memory": "none",
            "counting": "One shared execution ledger; independently accepted and freshly reused target scopes. Repeated attempts and versions do not add scopes.",
            "future_policy": "No final-label evaluation in goal execution or publication; a separately frozen study performs it after terminal allocation status.",
        },
    )
    return store


class SharedScientistGoal:
    def __init__(self, store, credential):
        self.store, self.credential = store, credential
        state = store.read()
        self.protocol = read(store.root / "protocol.json")
        require(self.protocol["schema"] == SCHEMA, "Unsupported shared goal")
        require(
            tuple(state["definition"]["goal"]["criteria"]) == CRITERIA
            and state["definition"]["condition"] == "science_goal",
            "Shared goal criteria or mode changed",
        )
        self.tasks = [
            read(safe_child(store.root, "targets/" + t["id"] + "/model_task.json")) for t in self.protocol["targets"]
        ]
        self.by_id = {t["id"]: t for t in self.tasks}
        self.data = admit(self.tasks)
        history_policy = self.protocol["admission"]["shared_training_history"]
        require(history_policy in SUPPORTED_POLICIES, "Unsupported shared training history")
        expected = {
            "schema": SCHEMA,
            "tasks": [model_tasks.admission(t) for t in self.tasks],
            "training_baseline_feedback": self.protocol["target_training_baseline_feedback"],
            "implementation_sha256": digest(self.protocol["implementation"]),
            "shared_training_history": history_policy,
        }
        if shared_library_experience.FIELD in self.protocol:
            expected[shared_library_experience.FIELD] = self.protocol[shared_library_experience.FIELD]
        require(
            expected == self.protocol["admission"] == state["definition"]["admission"], "Shared task admission changed"
        )
        self.library_experience = shared_library_experience.verify_goal(store.root, state, self.protocol, self.tasks)
        for binding, task in zip(self.protocol["targets"], self.tasks, strict=True):
            require(
                binding["id"] == task["id"] and binding["task_sha256"] == digest(task),
                "Shared target order or definition changed",
            )
            for phase, rows in self.data[task["id"]].items():
                frozen = read(safe_child(store.root, f"targets/{task['id']}/data/{phase}.json"))
                require(frozen == rows and digest(rows) == binding["data_hashes"][phase], "Shared cohort changed")
            source, split = model_tasks.source_and_split(task)
            require(
                (source, split) == (binding["source"], binding["split"]), "Shared source or partition evidence changed"
            )
            feedback_policy(
                {
                    "training_baseline_feedback": self.protocol["target_training_baseline_feedback"],
                    "scope": task["scope"],
                },
                plugin_id=task["plugin_id"],
            )
        require(bool(self.protocol["implementation"]), "Missing shared goal implementation")
        for name, expected_hash in self.protocol["implementation"].items():
            require(
                file_hash(safe_child(ROOT, name)) == expected_hash
                and file_hash(safe_child(store.root, "implementation/" + name)) == expected_hash,
                "Shared goal implementation changed; explicit migration required",
            )

    def records(self, state):
        progress = state["progress"]
        require(not progress or set(progress) == {"targets"}, "Unknown shared progress fields")
        records = progress.get("targets", {})
        require(isinstance(records, dict) and set(records) <= set(self.by_id), "Unallocated target in shared progress")
        for record in records.values():
            fields = {"controller", "model_id", "development_episode"}
            require(
                set(record) in [fields, fields | {"reuse_episode", "reuse_receipt"}], "Invalid shared model progress"
            )
        return records

    def lifecycle(self, task_id, record):
        task = self.by_id[task_id]
        folder = safe_child(self.store.root, record["development_episode"])
        controller = safe_child(self.store.root, record["controller"])
        require(controller == folder / "targets" / task_id / "controller", "Shared model controller changed")
        policy = model_tasks.policy_for_task(task, self.data[task_id])
        require(read(controller / "policy.json") == policy, "Shared target acceptance policy changed")
        lifecycle = ModelLifecycle(controller, policy)
        mid = record["model_id"]
        model = lifecycle._model(mid)
        local = lifecycle._state()
        report = lifecycle._verified_assessment(mid, local, model)
        require(
            mid in local["published"] and report["passed"] and report["training_replayed"],
            "Shared target is not independently accepted",
        )
        run = read(folder / "execution/run.json")
        require(
            run["phase"] == "develop"
            and run["status"] == "completed"
            and local["candidates"][mid]["session_id"] == run["session_id"]
            and local["candidates"][mid]["mode"] == "science_expert",
            "Shared development session changed",
        )
        require(
            read(folder / "workspace/targets" / task_id / "candidate.json") == model,
            "Shared accepted candidate changed",
        )
        return lifecycle, mid

    def evaluate(self, state):
        records = self.records(state)
        accepted, reused = [], []
        for task_id, record in records.items():
            lifecycle, mid = self.lifecycle(task_id, record)
            accepted.append(evidence(self.store.root, lifecycle.root / f"assessment_{mid}.json"))
            if "reuse_receipt" not in record:
                continue
            folder = safe_child(self.store.root, record["reuse_episode"])
            path = safe_child(self.store.root, record["reuse_receipt"])
            require(path == folder / "receipts" / (task_id + ".json"), "Shared reuse receipt location changed")
            run = read(folder / "execution/run.json")
            local = lifecycle._state()
            require(
                run["phase"] == "reuse"
                and run["status"] == "completed"
                and run["session_id"] != local["candidates"][mid]["session_id"],
                "Shared reuse session was not fresh",
            )
            task = self.by_id[task_id]
            inputs = model_tasks.inference_inputs(self.data[task_id]["future"], task["plugin_id"])
            predictions = read(folder / "workspace/targets" / task_id / "predictions.json")
            actual = get_model_plugin(task["plugin_id"]).predict(lifecycle._model(mid), inputs, task["scope"])
            require(predictions == actual, "Shared reuse predictions changed")
            expected = {
                "event": "reused",
                "model_id": mid,
                "session_id": run["session_id"],
                "mode": "science_expert",
                "input_sha256": digest(inputs),
                "prediction_sha256": digest(actual),
                "rows": len(inputs),
            }
            require(read(path) == expected and expected in local["events"], "Shared reuse event changed")
            reused.append(evidence(self.store.root, path))
        return {
            "criteria": {
                CRITERIA[0]: {"passed": len(accepted) == len(self.tasks), "evidence": accepted},
                CRITERIA[1]: {"passed": len(reused) == len(self.tasks), "evidence": reused},
            }
        }

    def training_history(self, state, selected):
        return project_shared_history(
            self.store.root, state, selected, self.protocol["admission"]["shared_training_history"]
        )

    async def cycle(self, index, profile, state, context):
        started = time.monotonic()
        records = copy.deepcopy(self.records(state))
        pending = [t for t in self.tasks if t["id"] in records and "reuse_receipt" not in records[t["id"]]]
        phase = "reuse" if pending else "develop"
        selected = pending if pending else [t for t in self.tasks if t["id"] not in records]
        require(bool(selected), "No unfinished shared target remains")
        folder = self.store.root / "episodes" / f"cycle_{index:03d}"
        work = folder / "workspace"
        registrations = None
        if phase == "reuse":
            registrations = {}
            for definition in selected:
                lifecycle, mid = self.lifecycle(definition["id"], records[definition["id"]])
                registrations[definition["id"]] = {"lifecycle": lifecycle, "model_id": mid}
        history = self.training_history(state, selected) if phase == "develop" else None
        versions, version, capability = prepare_bundle(
            folder / "capability",
            work,
            selected,
            profile.public()["model"],
            template_factory=make_bundle,
            phase=phase,
            registrations=registrations,
            training_baseline_feedback=self.protocol["target_training_baseline_feedback"]
            if phase == "develop"
            else "none",
        )
        declaration = spec(phase)
        if phase == "develop":
            declaration["tools"].discard("discover_generated_models")
            declaration["retryable_read_tools"].discard("discover_generated_models")
        task = Task(
            work,
            capability,
            folder / "execution",
            shared_library_experience.append_prompt(
                make_shared_prompt(context, phase, history),
                phase,
                self.library_experience if phase == "develop" else None,
            ),
            SYSTEM,
            frozenset(declaration["tools"]),
            phase=phase,
        )
        if history is not None:
            atomic_json(folder / "shared_training_history.json", history)
        atomic_json(folder / "shared_prompt.json", task.prompt)
        reserved = (state.get("inflight") or {}).get("reserved", {})
        allocated_wall = reserved.get("wall_seconds", profile.budget.wall_seconds)
        bounded = execution_profile(profile, allocated_wall - (time.monotonic() - started))
        execution_cancelled = False
        try:
            run = await execute(bounded, task, default_runtimes(), credential=self.credential)
        except asyncio.CancelledError:
            # Only the host's finalized, bound cancellation record permits a
            # failed checkpoint. Missing/modified records still stop the goal.
            run = read(task.output / "run.json")
            lock = read(task.output / "profile.lock.json")
            require(
                run.get("cancellation_finalized") is True
                and run["status"] == "interrupted"
                and run["error_type"] == "CancelledError"
                and run["phase"] == phase
                and run["binding_intact"] is True
                and run["profile_sha256"] == lock["sha256"]
                and lock["sha256"] == digest({k: v for k, v in lock.items() if k != "sha256"})
                and digest(lock["profile"]) == digest(bounded.public())
                and lock["task"]["prompt_sha256"] == digest(task.prompt),
                "Cancelled execution lacks a finalized host binding",
            )
            execution_cancelled = True
        audit = audit_events(
            task.output,
            phase,
            run,
            declaration["required_tools"],
            declaration["recoverable_tools"],
            declaration["retryable_read_tools"],
        )
        atomic_json(folder / "lifecycle_execution_audit.json", audit)
        versions.verify(version)
        outcome = {
            "status": "failed",
            "summary": phase + " shared episode did not pass its execution gate",
            "phase": phase,
            "allocated_targets": [t["id"] for t in selected],
            "run_record": (task.output / "run.json").relative_to(self.store.root).as_posix(),
            "execution_audit_sha256": digest(audit),
            "usage": {
                "model_calls": run["audit"]["requests"]["model_calls"],
                "tool_calls": run["audit"]["events"]["tool_calls"],
                "wall_seconds": run["host_wall_seconds"],
            },
            "progress": {"targets": records},
            "target_outcomes": {},
        }
        if execution_cancelled:
            outcome.update(
                execution_cancelled=True,
                error_type="CancelledError",
                summary=phase + " execution cancelled; observed usage saved and no new models accepted",
            )
        if history is not None:
            outcome["training_history_sha256"] = digest(history)
        if phase == "develop" and self.library_experience is not None:
            outcome[shared_library_experience.FIELD + "_sha256"] = digest(self.library_experience)
        if run["status"] == "completed" and audit["passed"]:
            if phase == "develop":
                ledger = read(work / ".shared_model_development.json")
                validate_generated_result(
                    "submit_model_code", ledger["submission"]["receipt"], workspace=work, capability=capability
                )
                candidates = read(work / "candidate.json")["targets"]
                require(
                    [c["task_id"] for c in candidates] == [t["id"] for t in selected], "Shared submission lost targets"
                )
                for candidate in candidates:
                    task_id = candidate["task_id"]
                    definition = self.by_id[task_id]
                    if candidate["status"] != "candidate":
                        outcome["target_outcomes"][task_id] = {"status": "training_failed"}
                        continue
                    controller = folder / "targets" / task_id / "controller"
                    lifecycle = ModelLifecycle(controller, model_tasks.policy_for_task(definition, self.data[task_id]))
                    mid = lifecycle.submit(
                        work / "targets" / task_id / "candidate.json", session_id=run["session_id"], mode=profile.mode
                    )
                    accepted = lifecycle.assess(
                        mid, training=self.data[task_id]["training"], validation=self.data[task_id]["validation"]
                    )
                    atomic_json(folder / "targets" / task_id / "acceptance.json", accepted)
                    outcome["target_outcomes"][task_id] = {
                        "status": "accepted" if accepted["passed"] else "candidate_rejected",
                        "model_id": mid,
                        **({"rejection_reasons": accepted.get("rejection_reasons", [])} if not accepted["passed"] else {}),
                    }
                    if accepted["passed"]:
                        records[task_id] = {
                            "controller": controller.relative_to(self.store.root).as_posix(),
                            "model_id": mid,
                            "development_episode": folder.relative_to(self.store.root).as_posix(),
                        }
                outcome.update(
                    status="completed",
                    summary="Shared development finished; each target retains its own acceptance outcome",
                )
            else:
                for definition in selected:
                    task_id = definition["id"]
                    entry = registrations[task_id]
                    inputs = model_tasks.inference_inputs(self.data[task_id]["future"], definition["plugin_id"])
                    receipt = entry["lifecycle"].record_reuse(
                        entry["model_id"],
                        session_id=run["session_id"],
                        mode=profile.mode,
                        inputs=inputs,
                        predictions=read(work / "targets" / task_id / "predictions.json"),
                        scope=definition["scope"],
                    )
                    path = folder / "receipts" / (task_id + ".json")
                    atomic_json(path, receipt)
                    records[task_id].update(
                        reuse_episode=folder.relative_to(self.store.root).as_posix(),
                        reuse_receipt=path.relative_to(self.store.root).as_posix(),
                    )
                    outcome["target_outcomes"][task_id] = {"status": "freshly_reused", "model_id": entry["model_id"]}
                outcome.update(
                    status="completed",
                    summary="Every selected target model replayed in a fresh shared execution session",
                )
        if outcome["status"] != "completed":
            outcome.update(access_wait(task.output, self.store.root))
        atomic_json(folder / "outcome.json", outcome)
        return outcome


async def main(args):
    if args.model_library is not None:
        from agent.rsi.engine.scientist_model_library import ScientistModelLibrary

        ScientistModelLibrary(args.model_library).goal(args.output)
    if args.action == "create":
        require(
            args.tasks is not None and args.profile is not None and args.objective,
            "Create requires tasks, profile and objective",
        )
        tasks = read(args.tasks)
        profile = Profile.from_dict(read(args.profile))
        authorize_profile(profile)
        goal = GoalSpec(
            args.objective,
            CRITERIA,
            max_cycles=args.max_cycles,
            model_calls=args.model_calls,
            tool_calls=args.tool_calls,
            wall_seconds=args.wall_seconds,
        )
        store = prepare(
            args.output,
            tasks,
            profile,
            goal,
            training_baseline_feedback=args.training_baseline_feedback,
            shared_training_history=args.shared_training_history,
            library_selection=read(args.library_selection) if args.library_selection else None,
        )
    else:
        store = GoalStore(args.output)
    if args.action == "pause":
        state = store.pause()
    elif args.action in {"create", "status"}:
        state = store.read()
    else:
        profile = Profile.from_dict(store.read()["definition"]["profile"])
        authorize_profile(profile)
        key = (
            credential_from_toml(args.credential_toml, profile.model)
            if args.credential_toml
            else os.environ.get(profile.model.credential_env)
        )
        require(bool(key), "Shared goal model credential is unresolved")
        driver = SharedScientistGoal(store, key)
        if args.action == "resume":
            store.resume()
        state = await run_goal(store, driver.cycle, driver.evaluate, stop_on_uncheckpointed_failure=True)
    publication = None
    if (
        args.model_library is not None
        and args.action in {"run", "resume"}
        and state["status"] in {"completed", "budget_exhausted"}
        and any("reuse_receipt" in row for row in state["progress"].get("targets", {}).values())
    ):
        from agent.rsi.engine.scientist_model_library import ScientistModelLibrary

        publication = ScientistModelLibrary(args.model_library).publish_goal(store.root)
        require(publication["all_published"], "Shared model publication did not pass")
    print(
        json.dumps(
            {
                "status": state["status"],
                "episodes": len(state["cycles"]),
                "usage": state["usage"],
                "criteria": state["assessment"],
                "published_models": len(publication["outcomes"]) if publication else 0,
                "publication": publication,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["create", "run", "pause", "resume", "status"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tasks", type=Path)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--objective")
    parser.add_argument("--credential-toml", type=Path)
    parser.add_argument("--model-library", type=Path)
    parser.add_argument(
        "--library-selection", type=Path, help="Frozen verified source selection for every allocated target"
    )
    parser.add_argument("--max-cycles", type=int, default=8)
    parser.add_argument("--model-calls", type=int, default=192)
    parser.add_argument("--tool-calls", type=int, default=96)
    parser.add_argument("--wall-seconds", type=float, default=7200)
    parser.add_argument("--training-baseline-feedback", default="none")
    parser.add_argument("--shared-training-history", choices=SUPPORTED_POLICIES, default=HISTORY_POLICY)
    asyncio.run(main(parser.parse_args()))
