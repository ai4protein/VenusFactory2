"""Responses reasoning controls remain fixed across scientific episodes."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from agent.harness import Budget, ModelBinding, Profile
from agent.rsi import generated_step_library, paper_step_goal, scientist_model_library
from agent.rsi.storage import atomic_json


def profiles():
    base = Profile(
        "science_expert",
        "expert-codex",
        ModelBinding("synthetic-gate-fixture", "model", "https://fixture.invalid/v1", protocol="responses"),
        Budget(thinking=None, temperature=None, reasoning_effort="low"),
    )
    return base, replace(base, budget=replace(base.budget, reasoning_effort="high"))


@pytest.mark.parametrize("target", ["goal", "library"])
def test_paper_step_rejects_reasoning_drift_before_acceptance(tmp_path, monkeypatch, target):
    base, changed = profiles()
    state = {"definition": {"profile": base.public()}}
    atomic_json(tmp_path / "runtime/profile.lock.json", {"profile": changed.public()})
    module = paper_step_goal if target == "goal" else generated_step_library
    monkeypatch.setattr(module, "verify_phase", lambda folder: {"synthetic": True})
    with pytest.raises(ValueError, match="inference settings changed"):
        if target == "goal":
            paper_step_goal.PaperStepGoal.phase_proof(SimpleNamespace(protocol={}), state, tmp_path)
        else:
            generated_step_library.session_proof(state, tmp_path)


def test_model_publication_rejects_reasoning_drift_before_audit(tmp_path, monkeypatch):
    base, changed = profiles()
    cycle = {"index": 0, "status": "completed", "usage": {"model_calls": 0, "tool_calls": 0}}
    state = {"definition": {"profile": base.public()}, "cycles": [cycle], "progress": {}}
    episode = tmp_path / "episodes/cycle_000"
    atomic_json(tmp_path / "cycle_000.json", cycle)
    atomic_json(episode / "outcome.json", {"status": "completed"})
    atomic_json(episode / "execution/run.json", {})
    atomic_json(episode / "execution/profile.lock.json", {"profile": changed.public()})
    monkeypatch.setattr(scientist_model_library.training_attempt_memory, "verify_episode", lambda *args: None)
    with pytest.raises(ValueError, match="Episode sampling changed"):
        scientist_model_library._episodes(tmp_path, state, {})
