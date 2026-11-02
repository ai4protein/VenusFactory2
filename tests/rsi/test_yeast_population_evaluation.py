"""Synthetic terminal boundaries and real six-method numerical comparisons."""

import copy
from contextlib import nullcontext
from pathlib import Path

import pytest
from test_yeast_population_controls import example

from agent.rsi import yeast_population_evaluation as evaluation
from agent.rsi.prediction_metrics import measure_context
from agent.rsi.protein_model import digest
from agent.rsi.scientist_model_campaign import ALL_ALLOCATIONS, ModelCampaignStore
from agent.rsi.storage import atomic_json, sha256_file
from agent.rsi.yeast_population_controls import controls


def terminal_state():
    return {
        "definition_sha256": "frozen",
        "plans": ["original"],
        "definition": {"execution_policy": ALL_ALLOCATIONS},
        "owner": None,
        "inflight": None,
        "allocations": {"task": {"status": "budget_exhausted", "attempts": [{"status": "budget_exhausted"}]}},
    }


def definition():
    return {"campaign_definition_sha256": "frozen", "plans": ["original"], "execution_policy": ALL_ALLOCATIONS}


@pytest.mark.parametrize("status", ["queued", "running", "paused", "waiting_for_model"])
def test_live_allocations_fail_before_data_or_output(tmp_path, monkeypatch, status):
    state = terminal_state()
    state["allocations"]["task"]["status"] = status

    class Store:
        root = tmp_path / "campaign"

        def lease(self):
            return nullcontext()

        def read(self):
            return state

    monkeypatch.setattr(evaluation, "load_binding", lambda pilot: (definition(), Store(), {}))
    monkeypatch.setattr(evaluation.model_tasks, "load_task", lambda task: pytest.fail("Opened task labels"))
    monkeypatch.setattr(evaluation.core, "evaluate", lambda *args: pytest.fail("Scored a live campaign"))
    output = tmp_path / "scores"
    with pytest.raises(ValueError, match="terminal execution"):
        evaluation.evaluate(tmp_path / "pilot", output)
    assert not output.exists()


@pytest.mark.parametrize("attack", ["design", "plans", "owner", "inflight", "unattempted"])
def test_terminal_gate_rejects_changed_or_unfinished_state(attack):
    state = terminal_state()
    if attack == "design":
        state["definition_sha256"] = "changed"
    elif attack == "plans":
        state["plans"] = ["changed"]
    elif attack in {"owner", "inflight"}:
        state[attack] = {"pid": 123}
    else:
        state["allocations"]["task"]["attempts"] = []
    with pytest.raises(ValueError):
        evaluation.require_terminal(definition(), state)


def test_budget_exhaustion_is_a_terminal_outcome():
    evaluation.require_terminal(definition(), terminal_state())


def test_runner_lease_blocks_evaluation_before_reading_state(tmp_path, monkeypatch):
    store = ModelCampaignStore(tmp_path / "campaign")
    store.root.mkdir()
    monkeypatch.setattr(evaluation, "load_binding", lambda pilot: (definition(), store, {}))
    monkeypatch.setattr(evaluation.model_tasks, "load_task", lambda task: pytest.fail("Opened task labels"))
    output = tmp_path / "scores"
    with store.lease(), pytest.raises(RuntimeError, match="runner already owns"):
        evaluation.evaluate(tmp_path / "pilot", output)
    assert not output.exists()


@pytest.fixture
def numeric_case(tmp_path, monkeypatch):
    scope, training, scoring = example()
    task = {"id": "synthetic_expression", "scope": scope}
    data = {"training": training, "future": scoring}
    monkeypatch.setattr(evaluation.model_tasks, "load_task", lambda requested: copy.deepcopy(data))
    core_output = tmp_path / "core"
    archived = []
    for outcome in controls(training, scoring, scope)[:4]:
        predictions = [
            {
                "id": row["id"],
                "protein": row["protein"],
                "group": row["group"],
                "values_sha256": digest(row["values"]),
                "prediction": item["prediction"],
            }
            for row, item in zip(scoring, outcome["predictions"], strict=True)
        ]
        target = core_output / (outcome["method"] + ".json")
        atomic_json(target, predictions)
        archived.append(
            {
                "method": outcome["method"],
                "status": "completed",
                "metrics": measure_context(training, scoring, predictions, scope=scope),
                "predictions": {"path": target.name, "sha256": sha256_file(target)},
            }
        )
    row = {
        "allocation_id": "task",
        "original_status": "budget_exhausted",
        "publication_and_reuse_success": False,
        "final_metrics": None,
        "context_controls": {"outcomes": archived},
    }
    return task, row, core_output


def test_failed_model_retains_six_controls_and_missing_score(tmp_path, numeric_case):
    result = evaluation.score_task(*numeric_case, tmp_path / "scores")
    assert result["original_status"] == "budget_exhausted"
    assert result["publication_and_reuse_success"] is False
    assert result["final_metrics"] is result["final_model_minus_best_rmse"] is None
    assert result["final_model_better_than_all_controls"] is None
    assert len(result["controls"]) == 6
    transcript = next(x for x in result["controls"] if x["method"] == "transcript_only_ols")
    assert transcript["metrics"]["rmse"] < 1e-10


@pytest.mark.parametrize("attack", ["rehashed_predictions", "metrics", "missing_control"])
def test_core_results_cannot_be_changed_before_extension(tmp_path, numeric_case, attack):
    task, row, core_output = numeric_case
    if attack == "rehashed_predictions":
        reference = row["context_controls"]["outcomes"][0]["predictions"]
        path = core_output / reference["path"]
        predictions = evaluation.core.read(path)
        predictions[0]["prediction"] += 1
        atomic_json(path, predictions)
        reference["sha256"] = sha256_file(path)
    elif attack == "metrics":
        row["context_controls"]["outcomes"][0]["metrics"]["rmse"] += 1
    else:
        row["context_controls"]["outcomes"].pop()
    with pytest.raises(ValueError, match="Core"):
        evaluation.score_task(task, row, core_output, tmp_path / "scores")


@pytest.mark.parametrize("attack", ["methods", "numeric_source", "protocol_hash"])
def test_predeclared_control_binding_is_enforced(tmp_path, monkeypatch, attack):
    plan = tmp_path / "plan.json"
    atomic_json(plan, {})
    protocol = {
        "final_controls": {
            "methods": list(evaluation.METHODS),
            "implementation": "src/agent/rsi/yeast_population_controls.py",
            "sha256": sha256_file(Path(evaluation.yeast_population_controls.__file__)),
        }
    }
    if attack == "methods":
        protocol["final_controls"]["methods"].pop()
    elif attack == "numeric_source":
        protocol["final_controls"]["sha256"] = "0" * 64
    atomic_json(tmp_path / "protocol.json", protocol)
    atomic_json(
        tmp_path / "preparation.json",
        {
            "plan_sha256": sha256_file(plan),
            "protocol_sha256": "0" * 64 if attack == "protocol_hash" else sha256_file(tmp_path / "protocol.json"),
        },
    )
    monkeypatch.setattr(evaluation.model_tasks, "load_task", lambda task: pytest.fail("Opened task labels"))
    with pytest.raises(ValueError):
        evaluation.load_binding(tmp_path)
