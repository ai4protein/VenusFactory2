"""History interventions separate source inheritance from feedback and extra search."""

import asyncio
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/rsi"))
from run_classifier_evolution import HISTORY_POLICIES, EvolvingClassifierGoal
from run_classifier_evolution_comparison import comparison_profiles, contrasts, schedule
from run_science_evolution import EvolvingModelGoal
from test_classifier_evolution import classifier_goal

from agent.harness.contracts import digest
from agent.harness.host import write_json


@pytest.mark.parametrize("policy", HISTORY_POLICIES)
def test_history_context_cannot_leak_prior_checkpoint_source(tmp_path, monkeypatch, policy):
    store = classifier_goal(tmp_path)
    path = store.root / "protocol.json"
    protocol = json.loads(path.read_text())
    protocol.update(schema=3, history_policy=policy)
    write_json(path, protocol)
    goal = EvolvingClassifierGoal(store, "fixture")
    state = store.read()
    state["cycles"] = [{"index": 0, "status": "failed", "summary": "SENTINEL_PRIVATE_PRIOR_SOURCE"}]
    observed = {}

    async def cycle(self, index, profile, state, context):
        observed["context"] = context
        return {"status": "failed", "progress": state["progress"]}

    monkeypatch.setattr(EvolvingModelGoal, "cycle", cycle)
    (store.root / "episodes/cycle_001").mkdir(parents=True)
    asyncio.run(goal.cycle(1, None, state, "SENTINEL_PRIVATE_PRIOR_SOURCE"))
    assert "SENTINEL_PRIVATE_PRIOR_SOURCE" not in observed["context"]
    assert "Persistent Science goal" in observed["context"]
    assert json.loads((store.root / "episodes/cycle_001/goal_context.json").read_text())["sha256"] == digest(
        observed["context"]
    )


@pytest.mark.parametrize("repeats", [1, 2, 3])
def test_comparison_allocation_is_complete_and_ordered(repeats):
    rows = schedule(repeats)
    assert len(rows) == 18 * repeats == len({r["id"] for r in rows})
    for repeat in range(repeats):
        cells = {(r["case"], r["condition"], r["history_policy"]) for r in rows if r["repeat"] == repeat}
        assert len(cells) == 18
    if repeats > 1:
        assert [r["case"] + r["condition"] + r["history_policy"] for r in rows[:18]] != [
            r["case"] + r["condition"] + r["history_policy"] for r in rows[18:36]
        ]


def test_contrasts_preserve_negative_scores_and_separate_interventions():
    rows = [
        {
            **r,
            "selected_future_skill": -0.2 if r["history_policy"] == "source_and_metrics" else 0.1,
            "complete_objective": False,
            "observed_model_calls": 8,
        }
        for r in schedule(1)
    ]
    values = contrasts(rows)
    assert len(values) == 36
    assert {r["kind"] for r in values} == {"history", "mode"}
    assert any(r["selected_future_skill_difference"] == pytest.approx(-0.3) for r in values)


def test_unknown_history_is_rejected():
    goal = object.__new__(EvolvingClassifierGoal)
    goal.protocol = {"history_policy": "uncontrolled"}
    with pytest.raises(ValueError, match="history"):
        goal.history_policy()


def test_frozen_runtime_imports_without_original_checkout(tmp_path):
    import os
    import subprocess

    from run_classifier_evolution import ROOT, implementation_paths

    paths = implementation_paths()
    assert ROOT / "src/logger.py" in paths
    for source in paths:
        destination = tmp_path / source.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
    result = subprocess.run(
        [sys.executable, str(tmp_path / "scripts/rsi/run_classifier_evolution_comparison.py"), "--help"],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(tmp_path / "src")},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "--nucb-assets" in result.stdout


@pytest.mark.parametrize("thinking", ["enabled", "disabled"])
def test_matched_profiles_preserve_declared_thinking(tmp_path, thinking):
    from dataclasses import replace

    from agent.harness import Profile

    store = classifier_goal(tmp_path)
    base = Profile.from_dict(store.read()["definition"]["profile"])
    base = replace(
        base,
        harness="langgraph",
        model=replace(base.model, model="glm-5.2"),
        budget=replace(base.budget, thinking=thinking),
    )
    profiles = comparison_profiles(base)
    assert set(profiles) == {"science_goal", "persistent_agent_ablation", "scripted_agent_turns"}
    assert profiles["science_goal"].mode == "science_expert"
    assert profiles["scripted_agent_turns"].mode == profiles["persistent_agent_ablation"].mode == "science_agent"
    for profile in profiles.values():
        assert profile.budget.thinking == thinking
        assert profile.budget.model_calls == 24
        assert profile.budget.tool_calls == 12
        assert profile.budget.output_tokens == 12288
        assert profile.budget.wall_seconds == 600
    assert base.budget.thinking == thinking
