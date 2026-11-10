"""Independently replay a terminal Science model lineage, without rerunning an agent."""

from __future__ import annotations

# Direct-script entry point; implementation imports remain package-qualified.
import sys as _entry_sys
from pathlib import Path as _EntryPath

_entry_root = _EntryPath(__file__).resolve().parents[3]
_entry_sys.path[:0] = [str(_entry_root), str(_entry_root / 'src'), str(_entry_root / 'scripts/rsi')]


import argparse
import json
from pathlib import Path

from agent.harness.contracts import digest, file_hash
from agent.rsi import model_tasks
from agent.rsi import training_attempt_memory as attempt_memory
from agent.rsi.datasets.source_lineage import program_hash
from agent.rsi.engine import prediction_batches
from agent.rsi.engine.experience import ModelExperienceStore
from agent.rsi.engine.model_lifecycle import ModelLifecycle
from agent.rsi.engine.model_plugins import get_model_plugin
from agent.rsi.engine.storage import atomic_json


def read(path):
    return json.loads(path.read_text())


def audit(root, *, score_future=True):
    root = root.resolve()
    state, protocol = read(root / "goal.json"), read(root / "protocol.json")
    prediction_batches.verify_goal(state, protocol)
    if state["status"] not in {"completed", "budget_exhausted", "cancelled"} or state["inflight"]:
        raise ValueError("Audit requires a terminal goal")
    if digest(state["definition"]) != state["definition_sha256"]:
        raise ValueError("Goal definition changed")
    for name, sha in protocol["implementation"].items():
        if file_hash(root / "implementation" / name) != sha:
            raise ValueError("Archived implementation changed")
    data = {name: read(root / "data" / (name + ".json")) for name in ["training", "validation", "future"]}
    assert all(digest(rows) == protocol["data_hashes"][name] for name, rows in data.items())
    if "model_task" in protocol:
        definition = read(root / "model_task.json")
        if (
            model_tasks.admission(definition) != protocol["model_task"]
            or state["definition"].get("admission") != protocol["model_task"]
            or definition["plugin_id"] != protocol["model_plugin_id"]
            or definition["scope"] != protocol["scope"]
            or definition["minimum_skill"] != protocol["model_task_minimum_skill"]
            or model_tasks.load_task(definition) != data
        ):
            raise ValueError("Model task admission changed")
    bank = ModelExperienceStore(root / "experience")
    plugin = get_model_plugin(protocol.get("model_plugin_id", "generated_python.v1"))
    seen_programs = set()
    history = protocol.get("history_policy", "source_and_metrics")
    if history not in {"source_and_metrics", "metrics_only", "independent_restart"}:
        raise ValueError("Unknown history intervention")
    prefix = {"cycles": []}
    for cycle in state["cycles"]:
        attempt_memory.verify_episode(root, prefix, protocol, cycle)
        prefix["cycles"].append(cycle)
    previous, previous_skill = [], protocol.get("model_task_minimum_skill", 0.0)
    results = []
    for record in state["progress"].get("generations", []):
        folder = (root / record["controller"]).resolve()
        folder.relative_to(root)
        lc = ModelLifecycle(folder, read(folder / "policy.json"))
        prediction_batches.require_bound_policy(lc.policy, protocol)
        mid = record["model_id"]
        model = lc._model(mid)
        controller = lc._state()
        report = lc._verified_assessment(mid, controller, model)
        assert mid in controller["published"] and report["passed"] and report["training_replayed"]
        assert digest(report) == controller["candidates"][mid]["assessment_sha256"]
        assert lc.policy["training_sha256"] == digest(data["training"])
        assert lc.policy["validation_sha256"] == digest(data["validation"])
        rebuilt = plugin.fit(data["training"], protocol["scope"], model["recipe"])
        assert digest(rebuilt) == digest(model)
        metrics = plugin.evaluate(rebuilt, data["validation"], protocol["scope"])
        assert digest(metrics) == digest(report["metrics"])
        skill = metrics[plugin.acceptance_metric]
        assert skill > previous_skill and lc.policy["minimum_skill"] == previous_skill
        if protocol["evolution"].get("schema", 1) >= 2:
            program = program_hash(model["recipe"]["source_code"])
            assert program not in seen_programs
            assert record["program_sha256"] == program and record["source_sha256"] == model["source_sha256"]
            assert record["generation"] == len(results) + 1
            seen_programs.add(program)
        entry = bank._entry(record["experience_id"])
        assert entry["parents"] == previous and entry["model_id"] == mid
        assert entry["source_sha256"] == model["source_sha256"]
        reused = False
        if record.get("reuse_receipt"):
            receipt = read(root / record["reuse_receipt"])
            assert receipt in controller["events"] and receipt["event"] == "reused"
            assert receipt["model_id"] == mid
            inputs = model_tasks.inference_inputs(data["future"], plugin.id)
            assert receipt["input_sha256"] == digest(inputs)
            assert receipt["session_id"] != controller["candidates"][mid]["session_id"]
            predictions = plugin.predict(rebuilt, data["future"], protocol["scope"])
            saved = read((root / record["reuse_receipt"]).parent / "workspace/predictions.json")
            assert digest(saved) == digest(predictions) == receipt["prediction_sha256"]
            reused = True
        result = {
            "generation": record["generation"],
            "model_id": mid,
            "source_sha256": model["source_sha256"],
            "development_skill": skill,
            "training_replayed": True,
            "reuse_replayed": reused,
            "parents": previous,
        }
        if state["status"] == "completed" and score_future:
            result["future_metrics"] = plugin.evaluate(rebuilt, data["future"], protocol["scope"])
        results.append(result)
        previous, previous_skill = ([record["experience_id"]] if history == "source_and_metrics" else []), skill
    completed = len(results) >= protocol["evolution"]["generations"] and all(x["reuse_replayed"] for x in results)
    assert (state["status"] == "completed") == completed
    if completed and score_future:
        saved = read(root / "future_metrics.json")
        assert len(saved) == len(results)
        assert all(digest(a["metrics"]) == digest(b["future_metrics"]) for a, b in zip(saved, results))
    return {
        "status": "passed",
        "goal_status": state["status"],
        "future_scored": bool(score_future and completed),
        "history_policy": history,
        "generations": results,
        "episodes": len(state["cycles"]),
        "usage": state["usage"],
        "scope": "Functional evidence of goal-driven iterative code development and capability reuse. Adaptive development selection and one protein case do not establish general RSI efficacy.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--no-future", action="store_true", help="Replay lineage and predictions without evaluating future labels"
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    atomic_json(args.output / "audit.json", audit(args.run, score_future=not args.no_future))
    print("Independent lineage replay passed")
