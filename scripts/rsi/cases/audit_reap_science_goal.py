"""Offline audit of terminal REAP persistent feature-model goals."""

from __future__ import annotations

# Direct-script entry point; implementation imports remain package-qualified.
import sys as _entry_sys
from pathlib import Path as _EntryPath
_entry_root = _EntryPath(__file__).resolve().parents[3]
_entry_sys.path[:0] = [str(_entry_root), str(_entry_root / 'src'), str(_entry_root / 'scripts/rsi')]


import argparse
import copy
import json
import sys
from pathlib import Path

from scripts.rsi.evaluation.audit_goal_reservations import retained_timeout_reservation
from scripts.rsi.cases.audit_reap_generated import compare_metrics, independent_metrics, read, require, verify_code_origin

from agent.harness.contracts import digest, file_hash
from agent.harness.events import audit_requests, audit_trace
from agent.rsi import generated_features as feature_model
from agent.rsi import training_attempt_memory as attempt_memory
from agent.rsi.engine.numerical_evidence import verify_assessment
from agent.rsi.cases.reap_model_case import prepare_reap_model_case
from agent.rsi.engine.versions import VersionStore
from agent.science_goal import GoalStore, goal_cycle_prompt, validate_assessment


from agent.rsi.layout import recorded_relative, recorded_path

def audit_completed_goal(root, state, plugin, *, measure, compare, predict, defer_future_scoring):
    """Verify reuse without scoring labels until an explicitly published score exists."""
    if defer_future_scoring:
        require(
            not any((root / name).exists() for name in ["future_metrics.json", "selected_future_metrics.json"]),
            "Deferred audit found a published future score",
        )
        published_metrics = None
    else:
        # Admission precedes metric calculation, including in ordinary audits.
        published_metrics = read(root / "future_metrics.json")
    lc, mid = plugin.lifecycle(state)
    model = lc._model(mid)
    inputs = plugin.inference_inputs()
    require(
        all(set(row) <= {"id", "sequence", "features"} for row in inputs),
        "Reuse input contains a label or undeclared field",
    )
    result = {}
    if defer_future_scoring:
        predictions = predict(model, inputs, plugin.scope)
        result["future_scoring_deferred"] = True
    else:
        predictions, metrics = measure(model, plugin.data["future"], plugin.scope)
        compare(published_metrics, metrics)
        result["future_metrics"] = metrics
    receipt_path = root / state["progress"]["reuse_receipt"]
    receipt = read(receipt_path)
    require(
        read(receipt_path.parent / "workspace/predictions.json") == predictions
        and receipt["input_sha256"] == digest(inputs)
        and receipt["prediction_sha256"] == digest(predictions),
        "Fresh-session replay differs",
    )
    publication = read(root / "registered_experience.json")
    entry = read(root / "experience/entries" / (publication["experience_id"] + ".json"))
    require(
        digest(entry) == publication["experience_id"]
        and entry["model_id"] == mid
        and entry["source_code"] == model["recipe"]["source_code"]
        and entry["source_sha256"] == publication["source_sha256"]
        and entry["donor_scope"] == plugin.scope
        and entry["parents"] == [],
        "Registered source experience differs",
    )
    result.update(reuse_replayed=True, source_experience_verified=True)
    return result


def audit(root, implementation_root, *, domain=None, defer_future_scoring=False):
    root, implementation_root = Path(root).resolve(), Path(implementation_root).resolve()
    sys.path.insert(0, str(implementation_root / "scripts/rsi"))
    from run_generated_lifecycle import SYSTEM, spec
    from run_protein_lifecycle import audit_events
    from scripts.rsi.cases.run_reap_science_goal import ReapModelGoal

    store = GoalStore(root)
    state, protocol = store.read(), read(root / "protocol.json")
    require(
        state["status"] in {"completed", "budget_exhausted"} and state["inflight"] is None, "Terminal goal required"
    )
    for name, sha in protocol["implementation"].items():
        require(
            file_hash(root / "implementation" / name) == sha == file_hash(implementation_root / name),
            "Frozen implementation changed: " + name,
        )
    if domain is not None:
        plugin = domain.goal_factory(store, "offline-audit-no-credential")
    elif "improver" in protocol:
        from scripts.rsi.cases.run_reap_improver_goal import PolicyReapGoal

        plugin = PolicyReapGoal(store, "offline-audit-no-credential")
    else:
        plugin = ReapModelGoal(store, "offline-audit-no-credential")
    source = protocol["source"]
    rebuilt = (
        domain.rebuild(protocol)
        if domain is not None
        else prepare_reap_model_case(
            source["admission"],
            source["features"],
            source["baseline"],
            protocol["split"]["family"],
            protocol["split"]["seed"],
        )
    )
    fit_candidate = domain.fit_candidate if domain is not None else feature_model.fit_candidate
    measure = domain.independent_metrics if domain is not None else independent_metrics
    compare = domain.compare_metrics if domain is not None else compare_metrics
    acceptance_metric = domain.acceptance_metric if domain is not None else "mse_skill_vs_training_mean"
    require(
        (rebuilt[0], rebuilt[4], rebuilt[5]) == (protocol["scope"], source, protocol["split"]),
        "Public-source reconstruction differs",
    )
    for name, rows in zip(["training", "validation", "future"], rebuilt[1:4], strict=True):
        require(plugin.data[name] == rows, "Frozen data reconstruction differs")
    require(digest(SYSTEM) == protocol["instructions_sha256"], "System instruction binding differs")
    records, sessions, counts = [], [], {"model_calls": 0, "tool_calls": 0}
    charged_wall = 0.0
    prefix = copy.deepcopy(state)
    prefix["cycles"], prefix["progress"] = [], {}
    for index, cycle in enumerate(state["cycles"]):
        require(cycle["index"] == index, "Cycle order differs")
        folder = root / "episodes" / f"cycle_{index:03d}"
        require(read(root / f"cycle_{index:03d}.json") == cycle, "Cycle checkpoint differs")
        reservation = retained_timeout_reservation(root, state, cycle)
        if reservation is None:
            outcome = read(folder / "outcome.json")
            require(
                {k: v for k, v in cycle.items() if k not in {"index", "usage"}}
                == {k: v for k, v in outcome.items() if k != "usage"},
                "Episode outcome differs",
            )
        expected_feedback = [] if prefix["progress"].get("accepted_model") else plugin.development_feedback(prefix)
        feedback_path = folder / "development_feedback.json"
        if reservation is None:
            require(
                read(feedback_path) == expected_feedback
                and cycle["development_feedback_sha256"] == digest(expected_feedback),
                "Adaptive feedback differs",
            )
        elif feedback_path.exists():
            require(read(feedback_path) == expected_feedback, "Partial adaptive feedback differs")
        context = goal_cycle_prompt(prefix)
        if expected_feedback:
            context += (
                domain.feedback_context(expected_feedback)
                if domain is not None
                else (
                    "\nPrevious independently replayed source and aggregate DEVELOPMENT acceptance feedback. "
                    "Discuss why the model failed and revise it. These are adaptive development data, "
                    "not confirmatory outcomes; higher-order future scores and labels remain unavailable.\n"
                    + json.dumps(expected_feedback)
                )
            )
        phase = "reuse" if prefix["progress"].get("accepted_model") else "develop"
        if reservation is None:
            packet = attempt_memory.verify_episode(root, prefix, protocol, cycle)
            if packet is not None:
                context += attempt_memory.context(packet)
        if reservation is None:
            require(cycle["phase"] == phase, "Goal skipped verified progress")
        action = (
            "In this episode, discover the accepted model and use it to predict every supplied unseen input. Do not redevelop the model."
            if phase == "reuse"
            else plugin.development_action()
        )
        run, lock = read(folder / "execution/run.json"), read(folder / "execution/profile.lock.json")
        require(
            digest({k: v for k, v in lock.items() if k != "sha256"}) == lock["sha256"]
            and (reservation is not None or lock["sha256"] == run["profile_sha256"]),
            "Profile lock differs",
        )
        require(
            lock["task"]["prompt_sha256"] == digest(context + "\n\nNext goal stage: " + action),
            "Observed goal prompt differs from reconstructed development-only feedback",
        )
        require(lock["task"]["instructions_sha256"] == protocol["instructions_sha256"], "Episode instructions differ")
        expected_profile = copy.deepcopy(state["definition"]["profile"])
        for name in counts:
            expected_profile["budget"][name] = min(
                expected_profile["budget"][name], state["definition"]["goal"][name] - counts[name]
            )
        # Assessment time is separately charged but not separately archived.
        # Bind the actual phase wall ceiling and verify its common upper bound.
        wall = lock["profile"]["budget"]["wall_seconds"]
        require(0 < wall <= expected_profile["budget"]["wall_seconds"], "Invalid episode wall ceiling")
        expected_profile["budget"]["wall_seconds"] = wall
        require(expected_profile == lock["profile"], "Model/harness/call budget drift")
        inputs = (
            {"training.json": plugin.data["training"]}
            if phase == "develop"
            else {"inputs.json": plugin.inference_inputs()}
        )
        require(set(lock["task"]["inputs"]) == set(inputs), "Unexpected runtime input")
        for name, values in inputs.items():
            require(
                read(folder / "workspace" / name) == values
                and file_hash(folder / "workspace" / name) == lock["task"]["inputs"][name],
                "Runtime data changed",
            )
        bundle = folder / "capability/bundles"
        caps = list((bundle / "versions").iterdir())
        require(len(caps) == 1, "Ambiguous capability")
        cap = caps[0]
        manifest = VersionStore(bundle).verify(cap.name)
        require(
            {str(p.relative_to(cap)): file_hash(p) for p in cap.rglob("*") if p.is_file()}
            == lock["task"]["capabilities"],
            "Capability changed",
        )
        task = read(cap / "task.json")
        require(
            task["scope"] == plugin.scope
            and task["plugin_id"] == plugin.model_plugin_id
            and task["phase"] == phase
            and task["inputs"] == {n: digest(v) for n, v in inputs.items()},
            "Scientific contract changed",
        )
        require(
            manifest["metadata"] == {"phase": phase, "model": lock["profile"]["model"]}, "Capability metadata differs"
        )
        for name, sha in manifest["files"].items():
            if name.endswith(".py"):
                require(protocol["implementation"].get(recorded_relative(protocol["implementation"], name)) == sha, "Bundle source changed")
        for section, prefix_path in [("adapter_source", "src/agent/harness/"), ("supporting_source", "src/agent/")]:
            for name, sha in lock[section].items():
                require(protocol["implementation"].get(recorded_path(protocol["implementation"], prefix_path + name)) == sha, "Runtime source differs")
        if phase == "develop":
            empty = plugin.development_experience(prefix)
            require(
                read(cap / "experience.json") == empty and task["experience_sha256"] == digest(empty),
                "Unplanned external experience",
            )
        else:
            packet = read(cap / "registered/capability.json")
            lc, mid = plugin.lifecycle(prefix)
            require(
                packet["model_id"] == mid
                and packet["model"] == lc._model(mid)
                and packet["model_sha256"] == digest(packet["model"])
                and digest(packet) == task["capability_sha256"],
                "Reuse model differs",
            )
        requests = audit_requests(folder / "execution/model_requests.jsonl")
        events = audit_trace(
            folder / "execution/events.jsonl", lock["task"]["tools"], lock["profile"]["budget"]["tool_calls"]
        )
        if reservation is None:
            require(
                requests == run["audit"]["requests"] and events == run["audit"]["events"],
                "Terminal audit snapshot differs",
            )
            declaration = spec(phase)
            if phase == "develop":
                declaration["required_tools"].add("read_model_experience")
            scientific = audit_events(
                folder / "execution", phase, run, declaration["required_tools"], declaration["recoverable_tools"]
            )
            debit = cycle["usage"]
        else:
            # Partial evidence remains an unsuccessful stage. Missing final audit
            # snapshots must not be synthesized or used to claim scientific success.
            scientific = {"passed": False}
            require(
                set(events["problems"]) <= {"missing_event_trace", "unpaired_tool_calls"},
                "Invalid partial tool trace",
            )
            debit = reservation
        for name, value in [("model_calls", requests["model_calls"]), ("tool_calls", events["tool_calls"])]:
            if reservation is None:
                require(debit[name] == outcome["usage"][name] == value, "Episode accounting differs")
            else:
                require(value <= debit[name], "Partial observed calls exceed retained reservation")
            counts[name] += debit[name]
        charged_wall += debit["wall_seconds"]
        record = {
            "cycle": index,
            "phase": phase,
            "status": cycle["status"],
            "model_calls": requests["model_calls"],
            "tool_calls": events["tool_calls"],
            "request_passed": requests["passed"],
            "scientific_passed": scientific["passed"],
            "development_feedback_entries": len(expected_feedback),
            "reservation_retained": reservation is not None,
            "charged_usage": debit,
        }
        if cycle["status"] in {"completed", "candidate_rejected"}:
            require(
                run["status"] == "completed" and requests["passed"] and events["passed"] and scientific["passed"],
                "Successful scientific stage has failed execution evidence",
            )
            if run["mode"] == "science_expert":
                require(run["four_role_path_completed"] and run["finalizer_succeeded"], "Expert-team path incomplete")
        if (folder / "workspace/candidate.json").exists():
            candidate = read(folder / "workspace/candidate.json")
            raw_events = [json.loads(x) for x in (folder / "execution/events.jsonl").read_text().splitlines()]
            record["origin"] = verify_code_origin(folder / "workspace", raw_events, candidate)
        if (folder / "acceptance.json").exists():
            assessment = read(folder / "acceptance.json")
            mid = assessment["model_id"]
            controller = folder / "controller"
            VersionStore(controller / "artifacts").verify(mid)
            model = read(controller / "artifacts/versions" / mid / "model.json")
            require(model == candidate, "Assessed model differs from submitted source")
            require(
                digest(fit_candidate(plugin.data["training"], plugin.scope, model["recipe"])) == digest(model),
                "Independent training replay differs",
            )
            _, metrics = measure(model, plugin.data["validation"], plugin.scope)
            compare(assessment["metrics"], metrics)
            published = read(controller / "state.json")
            policy = plugin.acceptance_policy(prefix)
            witness = verify_assessment(controller, mid, model, policy, assessment)
            record["numerical_evidence_status"] = witness["status"]
            require(
                read(controller / "policy.json") == policy and assessment["policy_sha256"] == digest(policy),
                "Acceptance policy changed",
            )
            require(
                digest(assessment) == published["candidates"][mid]["assessment_sha256"]
                and assessment == read(controller / ("assessment_" + mid + ".json")),
                "Assessment binding changed",
            )
            passed = metrics[acceptance_metric] > 0
            require(assessment["passed"] == passed == (mid in published["published"]), "Publication gate differs")
            record.update(
                training_replayed=True,
                acceptance_metrics=metrics,
                accepted=passed,
                source_sha256=model["source_sha256"],
            )
        sessions.append(run["session_id"])
        prefix["cycles"].append(cycle)
        if "progress" in cycle:
            prefix["progress"] = cycle["progress"]
        records.append(record)
    require(len(set(sessions)) == len(sessions), "Goal episodes did not use fresh sessions")
    for name, total in counts.items():
        require(total == state["usage"][name] <= state["definition"]["goal"][name], "Cumulative calls differ")
    expected_assessment = plugin.evaluate(state)
    require(expected_assessment == state["assessment"], "Goal completion criteria changed")
    passed = validate_assessment(store, state, expected_assessment)
    require((state["status"] == "completed") == passed, "Goal status contradicts evidence")
    result = {
        "status": "passed",
        "goal_status": state["status"],
        "goal_sha256": file_hash(root / "goal.json"),
        "protocol_sha256": file_hash(root / "protocol.json"),
        "episodes": records,
        "usage": state["usage"],
        "observed_usage": {key: sum(record[key] for record in records) for key in counts},
        "retained_timeout_episodes": sum(record["reservation_retained"] for record in records),
        "scope": "Independent terminal source/input/feedback binding, numeric refit and metric replay; public retrospective engineering evidence, not an efficacy replicate or improver-evolution test.",
    }
    if passed:
        result.update(
            audit_completed_goal(
                root,
                state,
                plugin,
                measure=measure,
                compare=compare,
                predict=domain.predict if domain is not None else feature_model.predict,
                defer_future_scoring=defer_future_scoring,
            )
        )
    else:
        require(not (root / "future_metrics.json").exists(), "Incomplete goal has future score")
    result["wall_accounting_residual_seconds"] = state["usage"]["wall_seconds"] - charged_wall
    require(result["wall_accounting_residual_seconds"] >= -1e-8, "Negative assessment overhead")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--goal", type=Path, required=True)
    p.add_argument("--implementation-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--defer-future-scoring", action="store_true")
    args = p.parse_args()
    result = audit(args.goal, args.implementation_root, defer_future_scoring=args.defer_future_scoring)
    args.output.mkdir(parents=True, exist_ok=False)
    result["auditor_sha256"] = file_hash(Path(__file__))
    helper = Path(__file__).with_name("audit_reap_generated.py")
    result["helper_sha256"] = file_hash(helper)
    (args.output / "audit.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    (args.output / "auditor.py").write_bytes(Path(__file__).read_bytes())
    (args.output / helper.name).write_bytes(helper.read_bytes())
    reservation_helper = Path(__file__).with_name("audit_goal_reservations.py")
    result["reservation_helper_sha256"] = file_hash(reservation_helper)
    (args.output / reservation_helper.name).write_bytes(reservation_helper.read_bytes())
    (args.output / "audit.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2))
