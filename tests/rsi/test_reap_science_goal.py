"""Feature goals preserve an accepted model, labels and encoder identity across failed reuse."""

import json
import sys
from pathlib import Path

import pytest
from test_generated_features import CODE, ROWS, SCOPE

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/rsi"))
import run_science_goal as base
from scripts.rsi.cases.run_reap_science_goal import PLUGIN, ReapModelGoal, publish_completed_source

from agent.harness import Budget, ModelBinding, Profile
from agent.harness.contracts import digest
from agent.harness.host import write_json
from agent.rsi.engine.model_plugins import get_model_plugin
from agent.science_goal import GoalSpec, GoalStore, run_goal


def setup_goal(tmp_path, mode="science_expert"):
    profile = Profile(
        "science_agent" if mode == "scripted_agent_turns" else mode,
        "langgraph",
        ModelBinding("fixture", "fixture", "https://example.org/v1"),
        Budget(wall_seconds=120, model_calls=12, tool_calls=12),
    )
    store = GoalStore.create(
        tmp_path / "goal",
        GoalSpec(
            "Develop and reuse a frozen-feature predictor",
            ("accepted_model", "fresh_session_reuse"),
            max_cycles=4,
            model_calls=36,
            tool_calls=36,
            wall_seconds=600,
        ),
        profile,
        condition=(
            "science_goal"
            if mode == "science_expert"
            else "scripted_agent_turns"
            if mode == "scripted_agent_turns"
            else "persistent_agent_ablation"
        ),
    )
    data = {"training": ROWS[:60], "validation": ROWS[60:80], "future": ROWS[80:]}
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


@pytest.mark.parametrize("mode", ["science_expert", "science_agent", "scripted_agent_turns"])
@pytest.mark.parametrize("reject_first", [False, True])
@pytest.mark.asyncio
async def test_feature_goal_survives_failed_reuse_without_retraining(tmp_path, monkeypatch, mode, reject_first):
    store = setup_goal(tmp_path, mode)
    plugin = get_model_plugin(PLUGIN)
    phases, bundles, sessions = [], [], []

    async def execute(profile, task, runtimes, credential):
        phases.append(task.phase)
        packet = json.loads((task.capability / "task.json").read_text())
        bundles.append(packet)
        assert packet["plugin_id"] == PLUGIN
        assert packet["scope"]["representation"] == SCOPE["representation"]
        session = "fixture-session-" + str(len(phases))
        sessions.append(session)
        status = "completed"
        if task.phase == "develop":
            train = json.loads((task.workspace / "training.json").read_text())
            assert train == ROWS[:60]
            assert "fit(features, targets, seed)" in task.prompt and "read_model_experience" in task.tools
            code = CODE
            if reject_first and len(phases) == 1:
                code = CODE.replace(
                    "return (np.asarray(features) @ np.asarray(model['weights'])).tolist()",
                    "return [1000.0 for row in features]",
                )
            if reject_first and len(phases) == 2:
                assert "DEVELOPMENT acceptance feedback" in task.prompt
                assert "mse_skill_vs_training_mean" in task.prompt and "1000.0" in task.prompt
            model = plugin.fit(train, SCOPE, {"source_code": code, "seed": 19})
            write_json(task.workspace / "candidate.json", model)
        else:
            inputs = json.loads((task.workspace / "inputs.json").read_text())
            assert inputs and all(set(row) == {"id", "sequence", "features"} for row in inputs)
            assert not (task.workspace / "training.json").exists()
            if phases.count("reuse") == 1:
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
    goal = ReapModelGoal(store, "fixture")
    state = await run_goal(store, goal.cycle, goal.evaluate)
    assert state["status"] == "completed"
    assert phases == (["develop"] if reject_first else []) + ["develop", "reuse", "reuse"]
    assert state["usage"]["model_calls"] == 4 + int(reject_first)
    assert all(v["passed"] for v in state["assessment"]["criteria"].values())
    failed = store.root / "episodes" / f"cycle_{1 + int(reject_first):03d}" / "outcome.json"
    assert json.loads(failed.read_text())["status"] == "failed"
    if reject_first:
        feedback = goal.development_feedback(state)
        assert len(feedback) == 2 and not feedback[0]["acceptance_passed"]
        assert all("future" not in key and "target" not in key for entry in feedback for key in entry)
        assert "predictions_sha256" not in feedback[0]["development_acceptance_metrics"]
    lc, mid = goal.lifecycle(state)
    assert len(lc._state()["published"]) == 1
    receipt = json.loads((store.root / state["progress"]["reuse_receipt"]).read_text())
    assert receipt["session_id"] == sessions[-1] and receipt["session_id"] != sessions[0]
    assert not (store.root / "future_metrics.json").exists()
    packet = publish_completed_source(store, goal, state)
    assert publish_completed_source(store, goal, state) == packet
    from agent.rsi.engine.experience import ModelExperienceStore

    projection = ModelExperienceStore(store.root / "experience").project("source", [packet["experience_id"]])
    entry = projection["entries"][0]
    assert entry["source_code"] == CODE and "model_state" not in entry
    assert entry["donor_scope"]["representation"] == SCOPE["representation"]


def test_changed_feature_binding_is_rejected(tmp_path):
    store = setup_goal(tmp_path)
    p = store.root / "protocol.json"
    record = json.loads(p.read_text())
    record["model_plugin_sha256"] = "wrong"
    write_json(p, record)
    with pytest.raises(ValueError, match="binding"):
        ReapModelGoal(store, "fixture")


def test_unmet_goal_cannot_publish_experience(tmp_path):
    store = setup_goal(tmp_path)
    goal = ReapModelGoal(store, "fixture")
    with pytest.raises(ValueError, match="complete goal"):
        publish_completed_source(store, goal, store.read())
    assert not (store.root / "experience").exists()
