"""Classification goal preserves accepted work across a failed reuse episode."""

import json
import sys
from pathlib import Path

import pytest
from test_generated_classifier import CODE, ROWS, SCOPE

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/rsi"))
import run_science_goal as base
from scripts.rsi.cases.run_nucb_science_goal import PLUGIN, NucbModelGoal

from agent.harness import Budget, ModelBinding, Profile
from agent.harness.contracts import digest
from agent.harness.host import write_json
from agent.rsi.engine.model_plugins import get_model_plugin
from agent.science_goal import GoalSpec, GoalStore, run_goal


def setup_goal(tmp_path):
    profile = Profile(
        "science_expert",
        "langgraph",
        ModelBinding("fixture", "fixture", "https://example.org/v1"),
        Budget(wall_seconds=120, model_calls=12, tool_calls=12),
    )
    store = GoalStore.create(
        tmp_path / "goal",
        GoalSpec(
            "Develop and reuse a classifier",
            ("accepted_model", "fresh_session_reuse"),
            max_cycles=4,
            model_calls=36,
            tool_calls=36,
            wall_seconds=600,
        ),
        profile,
    )
    data = {"training": ROWS[:720], "validation": ROWS[720:840], "future": ROWS[840:900]}
    (store.root / "data").mkdir()
    for name, rows in data.items():
        write_json(store.root / "data" / (name + ".json"), rows)
    write_json(
        store.root / "protocol.json",
        {
            "scope": SCOPE,
            "data_hashes": {k: digest(v) for k, v in data.items()},
            "implementation": {},
            "model_plugin_id": PLUGIN,
            "model_plugin_sha256": get_model_plugin(PLUGIN).source_sha256,
        },
    )
    return store


@pytest.mark.asyncio
async def test_real_classifier_survives_failed_reuse_without_retraining(tmp_path, monkeypatch):
    store = setup_goal(tmp_path)
    plugin = get_model_plugin(PLUGIN)
    phases, bundles, sessions = [], [], []

    async def execute(profile, task, runtimes, credential):
        phases.append(task.phase)
        packet = json.loads((task.capability / "task.json").read_text())
        bundles.append(packet)
        assert packet["plugin_id"] == PLUGIN
        assert packet["scope"]["class_labels"] == SCOPE["class_labels"]
        session = "fixture-session-" + str(len(phases))
        sessions.append(session)
        status = "completed"
        if task.phase == "develop":
            train = json.loads((task.workspace / "training.json").read_text())
            assert train == ROWS[:720]
            assert "probabilit" in task.prompt and "read_model_experience" in task.tools
            model = plugin.fit(train, SCOPE, {"source_code": CODE, "seed": 19})
            write_json(task.workspace / "candidate.json", model)
        else:
            inputs = json.loads((task.workspace / "inputs.json").read_text())
            assert inputs and all(set(row) == {"id", "sequence"} for row in inputs)
            assert not (task.workspace / "training.json").exists()
            if len(phases) == 2:
                status = "runtime_error"
            else:
                lc, mid = goal.lifecycle(store.read())
                write_json(task.workspace / "predictions.json", plugin.predict(lc._model(mid), inputs, SCOPE))
        return {
            "status": status,
            "session_id": session,
            "host_wall_seconds": 0.01,
            "audit": {
                "passed": status == "completed",
                "requests": {"model_calls": 2 if status != "completed" else 1},
                "events": {"tool_calls": 1},
            },
        }

    monkeypatch.setattr(base, "execute", execute)
    monkeypatch.setattr(
        base, "audit_events", lambda output, phase, run, *args: {"passed": run["status"] == "completed"}
    )
    goal = NucbModelGoal(store, "fixture")
    state = await run_goal(store, goal.cycle, goal.evaluate)
    assert state["status"] == "completed"
    assert phases == ["develop", "reuse", "reuse"]
    assert state["usage"]["model_calls"] == 4
    assert all(v["passed"] for v in state["assessment"]["criteria"].values())
    assert (store.root / "episodes/cycle_001/outcome.json").exists()
    assert json.loads((store.root / "episodes/cycle_001/outcome.json").read_text())["status"] == "failed"
    lc, mid = goal.lifecycle(state)
    assert len(lc._state()["published"]) == 1
    receipt = json.loads((store.root / state["progress"]["reuse_receipt"]).read_text())
    assert receipt["session_id"] == sessions[2] and receipt["session_id"] != sessions[0]
    assert not (store.root / "future_metrics.json").exists()


def test_changed_classifier_binding_is_rejected(tmp_path):
    store = setup_goal(tmp_path)
    p = store.root / "protocol.json"
    record = json.loads(p.read_text())
    record["model_plugin_sha256"] = "wrong"
    write_json(p, record)
    with pytest.raises(ValueError, match="binding"):
        NucbModelGoal(store, "fixture")
