"""Synthetic curation checks; no experiment or model-count claims."""

import copy
from types import SimpleNamespace

import pytest

from agent.harness.contracts import digest
from agent.rsi.engine import model_retention as retention
from agent.rsi.engine.model_plugins import get_model_plugin
from agent.rsi.engine.numerical_evidence import archive_assessment
from agent.rsi.engine.storage import atomic_json

POLICY = {
    "schema": "model_retention_policy.v1",
    "metric_directions": {"rmse": "min", "mae": "min", "spearman": "max"},
    "archive_validation_dominated": True,
}


def record(identity, error=1.0, size=10, predictions=None, **changes):
    value = {
        "entry_id": identity,
        "scope_id": "same",
        "protein": "P1",
        "property": "binding",
        "scenario": "assay",
        "function": "binding",
        "interface": "generated.v1",
        "comparison": {
            "status": "comparable",
            "cohort": "same-fixed-data",
            "objectives": {"rmse": error, "mae": error, "spearman": -0.8},
            "payload_bytes": size,
            "predictions_sha256": predictions or identity,
        },
    }
    value.update(changes)
    return value


def test_same_function_on_different_proteins_is_not_redundant():
    rows = [record("a"), record("b", 10, 100, protein="P2", scope_id="other")]
    assert retention.select_retention(rows, POLICY)["archive"] == []


@pytest.mark.parametrize(
    "field,value",
    [("property", "stability"), ("scenario", "new assay"), ("function", "secretion"), ("interface", "other.v1")],
)
def test_coverage_dimensions_cannot_be_removed(field, value):
    rows = [record("a"), record("b", 10, 100, **{field: value})]
    assert not retention.select_retention(rows, POLICY)["archive"]


@pytest.mark.parametrize("change", ["cohort", "status", "objective_set"])
def test_unmatched_or_missing_evidence_is_retained(change):
    a, b = record("a"), record("b", 10, 100)
    if change == "cohort":
        b["comparison"]["cohort"] = "other validation rows"
    elif change == "status":
        b["comparison"] = {"status": "comparison_evidence_unavailable"}
    else:
        b["comparison"]["objectives"].pop("mae")
    assert not retention.select_retention([a, b], POLICY)["archive"]


def test_equal_aggregate_scores_do_not_prove_equal_predictions():
    assert not retention.select_retention([record("a"), record("b", size=100)], POLICY)["archive"]


def test_identical_validation_predictions_keep_smaller_artifact():
    a, b = record("a", size=100, predictions="same"), record("b", size=10, predictions="same")
    result = retention.select_retention([a, b], POLICY)
    assert result["archive"] == [{"entry_id": "a", "replacement": "b", "reason": "identical_validation_predictions"}]
    assert all(v["lost"] == 0 for v in result["coverage"].values())


def test_metric_or_artifact_tradeoffs_remain_active():
    small = record("a", error=2, size=10)
    accurate = record("b", error=1, size=100)
    assert not retention.select_retention([small, accurate], POLICY)["archive"]
    accurate["comparison"]["objectives"]["spearman"] = -0.1
    small["comparison"]["payload_bytes"] = 1000
    assert not retention.select_retention([small, accurate], POLICY)["archive"]


def test_dominance_replacements_are_retained_and_order_independent():
    rows = [record("a", 1, 10), record("b", 2, 20), record("c", 3, 30)]
    a = retention.select_retention(rows, POLICY)
    assert a == retention.select_retention(list(reversed(rows)), POLICY)
    assert a["active_ids"] == ["a"]
    assert {x["replacement"] for x in a["archive"]} == {"a"}


def test_subgroup_tradeoff_is_not_hidden_by_overall_metrics():
    a, b = record("a", 1), record("b", 2)
    a["comparison"]["objectives"]["non_reference.rmse"] = 3
    b["comparison"]["objectives"]["non_reference.rmse"] = 1
    assert not retention.select_retention([a, b], POLICY)["archive"]


def test_policy_can_limit_archiving_to_identical_predictions():
    policy = {**POLICY, "archive_validation_dominated": False}
    assert not retention.select_retention([record("a"), record("b", 2, 100)], policy)["archive"]


def test_invalid_policy_and_nonfinite_metrics_rejected():
    with pytest.raises(ValueError):
        retention.validate_policy({**POLICY, "archive_validation_dominated": 1})
    with pytest.raises(ValueError):
        retention._objectives({"rmse": float("nan")}, POLICY["metric_directions"])


@pytest.fixture
def numerical_reader(tmp_path):
    # Hand-authored storage fixture exercises evidence bindings only.
    folder = tmp_path / "goal"
    controller = folder / "episodes/cycle_000/controller"
    controller.mkdir(parents=True)
    plugin = get_model_plugin("generated_python.v1")
    scope = {"protein": "fixture", "target": "value", "assay": "fixture", "unit": "unit"}
    model = {"plugin_id": plugin.id, "scope": scope, "model_state": {"mean": 0}, "recipe": {"source_code": "# fixture"}}
    policy = {
        "scope": scope,
        "training_sha256": "a" * 64,
        "validation_sha256": digest([{"id": "x", "sequence": "AA", "target": 0.0}]),
        "plugin_id": plugin.id,
        "plugin_sha256": plugin.source_sha256,
    }
    predictions = [{"id": "x", "sequence": "AA", "prediction": 0.0}]
    metrics = {"rows": 1, "rmse": 0.0, "mae": 0.0, "spearman": 1.0, "predictions_sha256": digest(predictions)}
    evidence = archive_assessment(
        controller,
        "b" * 64,
        model,
        policy,
        metrics,
        predictions,
        [{"id": "x", "sequence": "AA", "target": 0.0}],
        [],
        {},
    )
    assessment = {
        "passed": True,
        "training_replayed": True,
        "metrics": metrics,
        "policy_sha256": digest(policy),
        "plugin_sha256": plugin.source_sha256,
        "numerical_evidence": evidence,
    }
    atomic_json(controller / "policy.json", policy)
    atomic_json(controller / ("assessment_" + "b" * 64 + ".json"), assessment)
    publication = {
        "key": {"scope": scope, "plugin_id": plugin.id, "training_sha256": "a" * 64},
        "packet": {"model": model, "model_id": "b" * 64, "assessment": assessment},
        "origin": {"goal_sha256": "c" * 64},
        "origin_goal": str(folder),
    }
    library = SimpleNamespace(owner=tmp_path, inspect_publication=lambda _: copy.deepcopy(publication))
    return library, controller, publication


def test_comparison_binds_actual_numeric_archive(numerical_reader):
    library, _, publication = numerical_reader
    result = retention._comparison(library, "entry", POLICY)
    assert result["status"] == "comparable"
    assert result["predictions_sha256"] == publication["packet"]["assessment"]["metrics"]["predictions_sha256"]


def test_modified_prediction_file_is_not_used_for_curation(numerical_reader):
    library, controller, _ = numerical_reader
    file = next(controller.glob("numerical_evidence/versions/*/predictions.json"))
    file.chmod(0o600)
    file.write_text("[]")
    with pytest.raises(ValueError):
        retention._comparison(library, "entry", POLICY)


def test_incompatible_runtime_is_retained(numerical_reader):
    library, _, publication = numerical_reader
    publication["packet"]["assessment"]["plugin_sha256"] = "0" * 64
    assert retention._comparison(library, "entry", POLICY)["status"] == "runtime_implementation_requires_review"
