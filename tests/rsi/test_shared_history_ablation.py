import pytest

from agent.rsi import shared_goal_history as history


def test_no_history_does_not_read_or_project_previous_programs(monkeypatch, tmp_path):
    def forbidden(*args):
        raise AssertionError("Control must not read prior program/feedback evidence")

    monkeypatch.setattr(history, "_observations", forbidden)
    state = {"cycles": [{"source_code": "SENTINEL_PRIOR_PROGRAM", "feedback": "SENTINEL_METRIC"}]}
    packet = history.project(tmp_path, state, [{"id": "target"}], history.NO_HISTORY_POLICY)
    assert packet is None
    prompt = history.make_prompt("same verified host progress", "develop", packet)
    assert "SENTINEL" not in prompt and "Prior paired" not in prompt
    assert history.DEVELOP_ACTION in prompt
    assert "same verified host progress" in prompt


def test_default_history_retains_verified_projection(monkeypatch, tmp_path):
    calls = []

    def observe(*args):
        calls.append(args)
        return []

    monkeypatch.setattr(history, "_observations", observe)
    packet = history.project(tmp_path, {"cycles": []}, [])
    assert calls and packet["policy"] == history.POLICY
    assert packet["available_attempts"] == 0
    assert history.POLICY == "bounded_training_program_history.v2"


def test_no_history_cannot_smuggle_a_prompt_packet():
    with pytest.raises(ValueError, match="Unsupported"):
        history.make_prompt("context", "develop", {"policy": history.NO_HISTORY_POLICY, "attempts": ["hidden"]})


def test_reuse_and_within_episode_instructions_are_identical():
    assert history.make_prompt("context", "reuse", None) == "context\nShared-program stage: " + history.REUSE_ACTION
    assert "within two attempts" in history.make_prompt("context", "develop", None)
