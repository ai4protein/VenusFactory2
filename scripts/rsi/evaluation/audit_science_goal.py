"""Read-only integrity audit and numerical replay of terminal model goals."""

from __future__ import annotations

# Direct-script entry point; implementation imports remain package-qualified.
import sys as _entry_sys
from pathlib import Path as _EntryPath
_entry_root = _EntryPath(__file__).resolve().parents[3]
_entry_sys.path[:0] = [str(_entry_root), str(_entry_root / 'src'), str(_entry_root / 'scripts/rsi')]


import argparse
import copy
import json
from pathlib import Path

from scripts.rsi.evaluation.audit_generated_campaign import read, require_equal, verify_snapshot
from run_generated_lifecycle import policy_for
from run_protein_lifecycle import prepare_case
from run_science_goal import evidence

from agent.harness.contracts import digest, file_hash
from agent.harness.events import audit_requests, audit_trace
from agent.harness.host import write_json
from agent.rsi import model_tasks
from agent.rsi import training_attempt_memory as attempt_memory
from agent.rsi.engine.model_lifecycle import ModelLifecycle
from agent.rsi.engine.model_plugins import get_model_plugin
from agent.science_goal import GoalStore, goal_cycle_prompt, validate_assessment

ROOT = Path(__file__).resolve().parents[3]


def audit(args):
    store = GoalStore(args.goal)
    state = store.read()
    if state["status"] != "completed" or state["inflight"]:
        raise ValueError("This replay requires a verified, terminal model goal")
    protocol = read(store.root / "protocol.json")
    for name, sha in protocol["implementation"].items():
        require_equal(file_hash(store.root / "implementation" / name), sha, "Archived source changed")
    task_definition = None
    if "model_task" in protocol:
        task_definition = read(store.root / "model_task.json")
        if (
            model_tasks.admission(task_definition) != protocol["model_task"]
            or state["definition"].get("admission") != protocol["model_task"]
            or task_definition["plugin_id"] != protocol["model_plugin_id"]
            or task_definition["minimum_skill"] != protocol["model_task_minimum_skill"]
        ):
            raise ValueError("Model task admission changed")
        task_data = model_tasks.load_task(task_definition)
        scope = task_definition["scope"]
        train, val, future = (task_data[p] for p in model_tasks.PHASES)
        source, split = model_tasks.source_and_split(task_definition)
    else:
        scope, train, val, future, source, split = prepare_case(protocol["case"], args.source_cache)
    for key, value in [("scope", scope), ("source", source), ("split", split)]:
        require_equal(value, protocol[key], "Source reconstruction changed")
    for name, rows in [("training", train), ("validation", val), ("future", future)]:
        require_equal(read(store.root / "data" / (name + ".json")), rows, "Frozen data changed")
    episodes, sessions = [], []
    counts = {"model_calls": 0, "tool_calls": 0}
    prefix = copy.deepcopy(state)
    prefix["cycles"], prefix["progress"] = [], {}
    for cycle in state["cycles"]:
        folder = store.root / "episodes" / f"cycle_{cycle['index']:03d}"
        require_equal(read(store.root / f"cycle_{cycle['index']:03d}.json"), cycle, "Episode checkpoint changed")
        outcome = read(folder / "outcome.json")
        # Accounting includes host assessment beyond the runtime's own wall time.
        require_equal(
            {k: v for k, v in cycle.items() if k not in {"index", "usage"}},
            {k: v for k, v in outcome.items() if k != "usage"},
            "Episode outcome changed",
        )
        run = read(folder / "execution/run.json")
        lock = read(folder / "execution/profile.lock.json")
        packet = attempt_memory.verify_episode(store.root, prefix, protocol, cycle)
        if packet is not None:
            action = (
                "In this episode, discover the accepted model and use it to predict every supplied unseen input. Do not redevelop the model."
                if cycle["phase"] == "reuse"
                else "In this episode, inspect the supplied protein task, author deterministic numerical fit/predict source code, evaluate using training-only feedback, and submit one actually tested candidate. Independent controller acceptance follows this episode."
            )
            expected_prompt = (
                goal_cycle_prompt(prefix) + attempt_memory.context(packet) + "\n\nNext goal stage: " + action
            )
            require_equal(
                lock["task"]["prompt_sha256"], digest(expected_prompt), "Training-memory prompt binding differs"
            )
        require_equal(lock["sha256"], digest({k: v for k, v in lock.items() if k != "sha256"}), "Lock changed")
        require_equal(run["profile_sha256"], lock["sha256"], "Run binding changed")
        caps_dirs = list((folder / "capability/bundles/versions").iterdir())
        if len(caps_dirs) != 1:
            raise ValueError("Ambiguous capability version")
        caps = caps_dirs[0]
        verify_snapshot(caps)
        for name, sha in lock["task"]["inputs"].items():
            require_equal(file_hash(folder / "workspace" / name), sha, "Input changed")
        for name, sha in lock["task"]["capabilities"].items():
            require_equal(file_hash(caps / name), sha, "Capability changed")
        events = audit_trace(
            folder / "execution/events.jsonl", lock["task"]["tools"], lock["profile"]["budget"]["tool_calls"]
        )
        requests = audit_requests(folder / "execution/model_requests.jsonl")
        for name, check in [("events", events), ("requests", requests)]:
            require_equal(check["passed"], run["audit"][name]["passed"], "Audit outcome changed")
        condition = state["definition"].get("condition", "science_goal")
        expected_mode = "science_expert" if condition == "science_goal" else "science_agent"
        if condition not in {"science_goal", "persistent_agent_ablation", "scripted_agent_turns"}:
            raise ValueError("Unknown research condition")
        require_equal(run["mode"], expected_mode, "Executor mode changed")
        require_equal(lock["profile"]["mode"], expected_mode, "Profile mode changed")
        architecture_passed = (
            run.get("four_role_path_completed") and run.get("finalizer_succeeded")
            if expected_mode == "science_expert"
            else not run.get("four_role_path_completed", False)
        )
        if cycle["status"] == "completed" and not (events["passed"] and requests["passed"] and architecture_passed):
            raise ValueError("Completed episode has incomplete execution evidence")
        for name, value in [("model_calls", requests["model_calls"]), ("tool_calls", events["tool_calls"])]:
            require_equal(value, cycle["usage"][name], "Accounting changed")
            counts[name] += value
        sessions.append(run["session_id"])
        episodes.append(
            {
                "index": cycle["index"],
                "phase": cycle["phase"],
                "original_status": cycle["status"],
                "evidence_verified": True,
            }
        )
        prefix["cycles"].append(cycle)
        if cycle.get("progress") is not None:
            prefix["progress"] = copy.deepcopy(cycle["progress"])
    for key, value in counts.items():
        require_equal(value, state["usage"][key], "Cumulative accounting changed")
    if len(set(sessions)) != len(sessions):
        raise ValueError("Sessions were not fresh")
    # Reconstruct criteria from archived evidence; do not instantiate a live
    # executor or weaken its strict guard against changing an active goal.
    record = state["progress"]["accepted_model"]
    controller = (store.root / record["controller"]).resolve()
    controller.relative_to(store.root)
    policy = (
        model_tasks.policy_for_task(task_definition, task_data) if task_definition else policy_for(scope, train, val)
    )
    lc = ModelLifecycle(controller, policy)
    mid = record["model_id"]
    model = lc._model(mid)
    publication = lc._state()
    acceptance_path = lc.root / ("assessment_" + mid + ".json")
    accepted = read(acceptance_path)
    require_equal(digest(accepted), publication["candidates"][mid]["assessment_sha256"], "Acceptance evidence changed")
    require_equal(
        mid in publication["published"] and accepted["passed"] and accepted["training_replayed"],
        True,
        "Model not independently published",
    )
    receipt_path = (store.root / state["progress"]["reuse_receipt"]).resolve()
    receipt_path.relative_to(store.root)
    receipt = read(receipt_path)
    require_equal(receipt in publication["events"] and receipt["event"] == "reused", True, "Reuse event not recorded")
    if receipt["session_id"] == publication["candidates"][mid]["session_id"]:
        raise ValueError("Reuse session is not fresh")
    assessment = {
        "criteria": {
            "accepted_model": {"passed": True, "evidence": [evidence(store, acceptance_path)]},
            "fresh_session_reuse": {"passed": True, "evidence": [evidence(store, receipt_path)]},
        }
    }
    require_equal(assessment, state["assessment"], "Goal criteria changed")
    if not validate_assessment(store, state, assessment):
        raise ValueError("Goal criteria do not pass")
    plugin = get_model_plugin(protocol.get("model_plugin_id", "generated_python.v1"))
    require_equal(plugin.fit(train, scope, model["recipe"]), model, "Independent training replay changed")
    require_equal(
        plugin.evaluate(model, val, scope),
        read(lc.root / ("assessment_" + mid + ".json"))["metrics"],
        "Acceptance replay changed",
    )
    future_metrics = plugin.evaluate(model, future, scope)
    require_equal(future_metrics, read(store.root / "future_metrics.json"), "Future replay changed")
    reuse_folder = (store.root / state["progress"]["reuse_receipt"]).parent
    inputs = model_tasks.inference_inputs(future, plugin.id)
    predictions = plugin.predict(model, inputs, scope)
    require_equal(predictions, read(reuse_folder / "workspace/predictions.json"), "Reuse replay changed")
    receipt = read(reuse_folder / "reuse_receipt.json")
    require_equal(digest(predictions), receipt["prediction_sha256"], "Receipt changed")
    require_equal(digest(inputs), receipt["input_sha256"], "Reuse inputs changed")
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(
        args.output / "report.json",
        {
            "status": "audit_completed",
            "goal": str(store.root),
            "goal_sha256": file_hash(store.root / "goal.json"),
            "protocol_sha256": file_hash(store.root / "protocol.json"),
            "auditor_sha256": file_hash(Path(__file__)),
            "replay_sources": {
                name: file_hash(ROOT / name) for name in protocol["implementation"] if (ROOT / name).exists()
            },
            "episodes": episodes,
            "accounting_verified": True,
            "training_replayed": True,
            "acceptance_replayed": True,
            "fresh_session_reuse_replayed": True,
            "future_metrics": future_metrics,
            "scope": "Integrity and deterministic replay; not a new scientific replicate",
        },
    )
    print(json.dumps({"status": "audit_completed", "episodes": len(episodes), "training_and_reuse_replayed": True}))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--goal", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--source-cache", type=Path, default=ROOT / "research_runs/rsi/20261011_paper_case_audit")
    audit(p.parse_args())
