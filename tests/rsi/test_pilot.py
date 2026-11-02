import json

import pytest

from agent.rsi import pilot
from agent.rsi import pilot_tools as tools


def call(fn, *args, **kwargs):
    return getattr(fn, "fn", fn)(*args, **kwargs)


def test_fixture_split_is_explicit_and_separate():
    tasks = pilot.build_tasks(5)
    assert len(tasks) == 30 and len({t["task_id"] for t in tasks}) == 30
    assert {s: sum(t["split"] == s for t in tasks) for s in ("development", "selection", "audit")} == {
        "development": 12,
        "selection": 6,
        "audit": 12,
    }


def test_model_must_validate_before_reuse_and_reject_wrong_task(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "ROOT", tmp_path)
    task = next(t for t in pilot.build_tasks() if t["kind"] == "reuse")
    for name, content in task["files"].items():
        (tmp_path / name).write_text(content)
    result = call(tools.fit_linear_fixture, "train.csv", "fixture")
    assert not result["registered"]
    assert call(tools.predict_linear_fixture, "fixture", "test_inputs.csv")["error"] == "model_not_registered"
    with pytest.raises(ValueError):
        call(tools.validate_linear_fixture, "fixture", "train.csv")
    assert call(tools.validate_linear_fixture, "fixture", "validation.csv")["registered"]
    result = call(tools.predict_linear_fixture, "fixture", "test_inputs.csv")
    assert pilot.matches(result["predictions"], task["expected"]["predictions"])
    assert call(tools.predict_linear_fixture, "fixture", "test_inputs.csv", "other")["valid"] is False


def test_failed_validation_does_not_register(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "ROOT", tmp_path)
    task = next(t for t in pilot.build_tasks() if t["kind"] == "reuse")
    for name, content in task["files"].items():
        (tmp_path / name).write_text(content)
    call(tools.fit_linear_fixture, "train.csv", "fixture")
    (tmp_path / "bad.csv").write_text("x1,x2,y\n4,3,100\n")
    assert not call(tools.validate_linear_fixture, "fixture", "bad.csv")["registered"]
    assert not (tmp_path / "registry/fixture.json").exists()


def test_wrong_id_and_traversal_detected(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "ROOT", tmp_path)
    task = next(t for t in pilot.build_tasks() if t["kind"] == "ids")
    for name, content in task["files"].items():
        (tmp_path / name).write_text(content)
    assert call(tools.audit_prediction_table, "input.csv", "prediction.csv") == task["expected"]
    with pytest.raises(ValueError):
        tools.child("../hidden.json")


def test_nonfinite_or_boolean_predictions_are_not_correct():
    assert not pilot.matches(float("nan"), 1.0)
    assert not pilot.matches(True, 1)
    assert not pilot.matches(1, True)


def test_fake_output_without_execution_is_not_success(tmp_path):
    task = pilot.build_tasks()[2]
    (tmp_path / "output.json").write_text(json.dumps(task["expected"]))
    (tmp_path / "events.jsonl").write_text("")
    result = pilot.evaluate(task, tmp_path, {"status": "completed"}, tmp_path / "events.jsonl")
    assert result["reason"] == "required_tool_not_executed"
