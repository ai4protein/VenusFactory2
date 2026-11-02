"""Independent classifier audit detects numerical, allocation and request drift."""

import copy
import json
import sys
from pathlib import Path

import pytest
from test_generated_classifier import CODE, ROWS, SCOPE
from test_nucb_goal_comparison import candidate
from test_nucb_science_goal import setup_goal

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/rsi"))
import audit_nucb_goal_comparison as audit
import run_nucb_goal_comparison as runner
from audit_nucb_science_goal import compare_metrics, independent_metrics

from agent.harness.contracts import digest
from agent.rsi.model_plugins import get_model_plugin


@pytest.fixture(scope="module")
def real_scores():
    plugin = get_model_plugin("generated_classifier.v1")
    model = plugin.fit(ROWS[:720], SCOPE, {"source_code": CODE, "seed": 314159})
    actual = plugin.evaluate(model, ROWS[720:840], SCOPE)
    predictions, independent = independent_metrics(model, ROWS[720:840], SCOPE)
    return actual, independent, predictions


def test_all_classification_metrics_independently_agree(real_scores):
    actual, independent, predictions = real_scores
    compare_metrics(actual, independent)
    assert independent["rows"] == 120 and independent["predictions_sha256"] == digest(predictions)
    assert {"average_precision", "roc_auc", "class_recall", "balanced_accuracy"} <= set(independent)


@pytest.mark.parametrize(
    "field", ["brier", "average_precision", "roc_auc", "class_recall", "predictions_sha256", "class_counts"]
)
def test_any_corrupted_reported_metric_is_rejected(real_scores, field):
    actual, independent, _ = real_scores
    actual = copy.deepcopy(actual)
    if isinstance(actual[field], list):
        actual[field][0] += 1
    elif isinstance(actual[field], str):
        actual[field] = "incorrect"
    else:
        actual[field] += 0.1
    with pytest.raises(ValueError, match="score differs"):
        compare_metrics(actual, independent)


def design():
    profile = {
        "mode": "science_agent",
        "harness": "langgraph",
        "model": {"model": "glm-5.2"},
        "budget": {
            "wall_seconds": 900,
            "request_seconds": 300,
            "model_calls": 24,
            "tool_calls": 12,
            "output_tokens": 12288,
            "temperature": 0.1,
            "thinking": "enabled",
        },
    }
    profiles = {
        c: dict(copy.deepcopy(profile), mode="science_expert" if c == "science_goal" else "science_agent")
        for c in audit.CONDITIONS
    }
    value = {
        "schedule": runner.schedule(),
        "profiles": profiles,
        "goal_budget": runner.BUDGET,
        "runtime_concurrency": 1,
        "start_after": None,
    }
    return {"design": value, "design_sha256": digest(value)}, {"design_sha256": digest(value)}


@pytest.mark.parametrize("mutation", ["missing_arm", "feedback", "mode_model", "total_budget"])
def test_factorial_design_rejects_drift_even_with_recomputed_digest(mutation):
    protocol, report = design()
    assert len(audit.verify_design(protocol, report)) == 18
    d = protocol["design"]
    if mutation == "missing_arm":
        d["schedule"].pop()
    elif mutation == "feedback":
        d["schedule"][0]["feedback"] = "source_and_metrics"
    elif mutation == "mode_model":
        d["profiles"]["science_goal"]["model"]["model"] = "other-model"
    else:
        d["goal_budget"] = dict(d["goal_budget"], model_calls=65)
    protocol["design_sha256"] = report["design_sha256"] = digest(d)
    with pytest.raises(ValueError):
        audit.verify_design(protocol, report)


def test_backend_request_alias_is_checked_not_only_profile(tmp_path):
    protocol, _ = design()
    profile = protocol["design"]["profiles"]["science_goal"]
    path = tmp_path / "episodes/cycle_000/execution/model_requests.jsonl"
    path.parent.mkdir(parents=True)
    request = {
        "event": "request",
        "model": "glm-5.2",
        "temperature": 0.1,
        "max_tokens": 12288,
        "thinking_sent": {"type": "enabled", "clear_thinking": True},
    }
    path.write_text(json.dumps(request) + "\n")
    assert audit.verify_request_parameters(tmp_path, profile) == 1
    request["model"] = "silent-fallback"
    path.write_text(json.dumps(request) + "\n")
    with pytest.raises(ValueError, match="parameters differ"):
        audit.verify_request_parameters(tmp_path, profile)


def test_independent_development_selection_preserves_rejected_model(tmp_path):
    store = setup_goal(tmp_path)
    cycle, lc, mid = candidate(store, 0)
    state = store.read()
    state["cycles"] = [cycle]
    model = lc._model(mid)
    _, metrics = independent_metrics(model, ROWS[720:840], SCOPE)
    replay = {
        "episodes": [
            {
                "cycle": 0,
                "training_replayed": True,
                "accepted": False,
                "source_sha256": model["source_sha256"],
                "acceptance_metrics": metrics,
            }
        ]
    }
    selected = audit.selection_from_replayed_assessments(
        store.root, state, replay, metric=audit.METRIC, compare=compare_metrics
    )
    assert selected == runner.select_model(store, state) and not selected["accepted"]


def test_audit_refuses_running_campaign_without_creating_outputs(tmp_path):
    run = tmp_path / "running"
    run.mkdir()
    (run / "report.json").write_text(json.dumps({"status": "running"}))
    output = tmp_path / "audit"
    with pytest.raises(ValueError, match="Complete 18-goal"):
        audit.audit(run, tmp_path / "implementation", output, assets=tmp_path / "assets", rebuilt=tmp_path / "rebuilt")
    assert not output.exists()


def test_independent_contrasts_keep_negative_observations():
    rows = [
        dict(
            row,
            complete_objective=False,
            all_allocation_brier_skill=-0.2 if row["feedback"] == "source_and_metrics" else 0.0,
        )
        for row in runner.schedule()
    ]
    result = audit.independent_contrasts(rows)
    assert result == runner.contrasts(rows) and len(result) == 21
    assert all(
        r["brier_skill_difference"] == -0.2 for r in result if r["comparison"] == "source_and_metrics_minus_none"
    )
