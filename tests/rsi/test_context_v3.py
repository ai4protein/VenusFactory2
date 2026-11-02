"""Condition groups through the real numeric worker and native Scientist tools."""

import copy
import json
from pathlib import Path

import pytest
from test_context_v2 import make_task, save

from agent.rsi import context_feedback_dispatch as feedback
from agent.rsi import generated_context_v2, model_tasks
from agent.rsi import generated_context_v3 as api
from agent.rsi.engine.context_baselines import run_controls
from agent.rsi.engine.model_plugins import get_model_plugin


@pytest.fixture
def contextual(tmp_path):
    task, data, _ = make_task(tmp_path / "assets")
    task["plugin_id"] = api.PLUGIN_ID
    task["scope"]["interface_version"] = api.PLUGIN_ID
    task["scope"]["grouping"]["level"] = "treatment_condition"
    source_path = Path(task["references"][0]["path"])
    source = save(source_path, {"kind": "synthetic treatment-condition measurements"})
    assignments = {}
    for phase, rows in data.items():
        for row in rows:
            row["group"] = row["group"].replace("deletion", "condition")
            assignments[row["id"]] = {"phase": phase, "group": row["group"]}
        task["data"][phase] = save(Path(task["data"][phase]["path"]), rows)
    path = Path(task["references"][1]["path"])
    evidence = json.loads(path.read_text())
    evidence.update(level="treatment_condition", sources=[source], observations=assignments)
    bound = save(path, evidence)
    task["references"] = [source, bound]
    task["scope"]["grouping"]["evidence_sha256"] = bound["sha256"]
    return task, data, assignments


def test_condition_groups_require_explicit_v3_and_source_evidence(contextual):
    task, data, _ = contextual
    assert model_tasks.load_task(task) == data
    old = copy.deepcopy(task["scope"])
    old["interface_version"] = generated_context_v2.PLUGIN_ID
    with pytest.raises(ValueError, match="Independent-group"):
        generated_context_v2.validate_scope(old)
    assert (
        get_model_plugin(api.PLUGIN_ID).source_sha256 != get_model_plugin(generated_context_v2.PLUGIN_ID).source_sha256
    )
    assert feedback.policy_for({}, plugin_id=api.PLUGIN_ID) == "none"
    assert (
        feedback.policy_for({feedback.FIELD: feedback.CONDITION_POLICY}, plugin_id=api.PLUGIN_ID)
        == feedback.CONDITION_POLICY
    )
    with pytest.raises(ValueError):
        feedback.policy_for({feedback.FIELD: feedback.WIDE_POLICY}, plugin_id=api.PLUGIN_ID)
    task["scope"]["grouping"]["evidence_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="group"):
        model_tasks.load_task(task)


def test_fixed_controls_fit_training_and_ignore_changed_outer_labels(contextual):
    task, data, _ = contextual
    first = run_controls(task, score_partition="future")
    assert len(first["outcomes"]) == 4 and all(row["status"] == "completed" for row in first["outcomes"])
    for phase in ("validation", "future"):
        for row in data[phase]:
            row["target"] += 1000
        task["data"][phase] = save(Path(task["data"][phase]["path"]), data[phase])
    second = run_controls(task, score_partition="future")
    assert [row["predictions"] for row in first["outcomes"]] == [row["predictions"] for row in second["outcomes"]]


@pytest.mark.asyncio
async def test_native_mcp_fit_feedback_registration_reuse_and_replay(contextual, tmp_path, monkeypatch):
    import test_context_integration as integration

    factory = integration.make_bundle

    def condition_bundle(*args, **kwargs):
        if args[1] == "develop":
            kwargs["training_baseline_feedback"] = feedback.CONDITION_POLICY
        result = factory(*args, **kwargs)
        assert (result[2] / "generated_context_v3.py").is_file()
        if args[1] == "develop":
            assert (result[2] / "context_v3_feedback.py").is_file()
        return result

    monkeypatch.setattr(integration, "api", api)
    monkeypatch.setattr(integration, "make_bundle", condition_bundle)
    await integration.test_real_context_mcp_fit_register_reuse_and_publication_replay(contextual, tmp_path)


@pytest.mark.asyncio
@pytest.mark.parametrize("evolving", [False, True])
async def test_scientist_goal_uses_condition_bundle(contextual, tmp_path, monkeypatch, evolving):
    import test_context_integration as integration

    monkeypatch.setattr(integration, "api", api)
    await integration.test_context_goal_reaches_host_with_production_bundle(contextual, tmp_path, monkeypatch, evolving)


def test_shared_final_scorer_accepts_condition_tasks_with_missing_model(contextual):
    from agent.rsi.engine.shared_context_evaluation import evaluate_task

    task, _, _ = contextual
    result = evaluate_task(task, None)
    assert result["scientist"]["status"] != "scored"
    assert len(result["controls"]) == 4
    assert all(item["status"] == "completed" for item in result["controls"])
