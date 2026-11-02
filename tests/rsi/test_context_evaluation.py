"""Independent condition scoring and terminal acquisition evaluation on synthetic data."""

import copy
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import mean_absolute_error, mean_squared_error
from test_campaign_evaluation import campaign as campaign
from test_campaign_evaluation import finish, install_publications
from test_context_integration import make_task, save
from test_generated_context import CODE, ROWS, SCOPE

from agent.harness.contracts import digest
from agent.rsi import campaign_evaluation as evaluation
from agent.rsi import generated_context, model_tasks
from agent.rsi.context_baselines import run_controls
from agent.rsi.prediction_metrics import measure_context


@pytest.fixture
def task(tmp_path):
    return make_task(tmp_path / "context_assets")[0]


def fixtures():
    train, future = copy.deepcopy(ROWS[:40]), copy.deepcopy(ROWS[50:])
    predictions = [
        {
            "id": r["id"],
            "protein": r["protein"],
            "group": r["group"],
            "values_sha256": digest(r["values"]),
            "prediction": r["target"] + i / 100,
        }
        for i, r in enumerate(future)
    ]
    return train, future, predictions


def test_grouped_metrics_match_independent_libraries_with_repeated_conditions():
    train, future, predictions = fixtures()
    got = measure_context(train, future, predictions, scope=SCOPE)
    truth = [r["target"] for r in future]
    values = [r["prediction"] for r in predictions]
    assert got["rmse"] == pytest.approx(mean_squared_error(truth, values) ** 0.5)
    assert got["mae"] == pytest.approx(mean_absolute_error(truth, values))
    assert got["mse_skill_vs_training_mean"] == pytest.approx(
        1 - mean_squared_error(truth, values) / mean_squared_error(truth, [np.mean([r["target"] for r in train])] * 10)
    )
    assert got["independent_groups"] == 1 and got["training_groups"] == 4


@pytest.mark.parametrize(
    "attack",
    [
        "id",
        "protein",
        "group",
        "values_sha256",
        "order",
        "missing",
        "duplicate",
        "training_id",
        "training_group",
        "wrong_protein",
        "domain",
        "bool",
        "nan",
    ],
)
def test_context_identity_and_independent_group_boundaries(attack):
    train, future, predictions = fixtures()
    if attack in {"id", "protein", "group", "values_sha256"}:
        predictions[0][attack] = "changed"
    elif attack == "order":
        predictions.reverse()
    elif attack == "missing":
        predictions.pop()
    elif attack == "duplicate":
        future[1] = copy.deepcopy(future[0])
        predictions[1] = copy.deepcopy(predictions[0])
    elif attack == "training_id":
        train[0]["id"] = future[0]["id"]
    elif attack == "training_group":
        train[0]["group"] = future[0]["group"]
    elif attack == "wrong_protein":
        future[0]["protein"] = predictions[0]["protein"] = "another protein"
    elif attack == "domain":
        future[0]["values"] = [100.0]
        predictions[0]["values_sha256"] = digest([100.0])
    else:
        predictions[0]["prediction"] = True if attack == "bool" else float("nan")
    with pytest.raises(ValueError):
        measure_context(train, future, predictions, scope=SCOPE)


def test_reference_anchor_is_excluded_without_refitting_training_mean():
    train, future, predictions = fixtures()
    scope = copy.deepcopy(SCOPE)
    scope["unit"] = "ratio_to_within_replicate_reference"
    scope["covariates"].append({"name": "reference_temperature", "unit": "degC", "minimum": 20.0, "maximum": 90.0})
    for row in train + future:
        row["values"].append(30.0)
    for row, prediction in zip(future, predictions, strict=True):
        prediction["values_sha256"] = digest(row["values"])
    got = measure_context(train, future, predictions, scope=scope)
    non_reference = got["non_reference_metrics"]
    assert non_reference["rows"] == 9
    assert non_reference["training_mean"] == got["training_mean"]
    assert non_reference["mse"] == pytest.approx(np.mean([(i / 100) ** 2 for i in range(1, 10)]))


def test_future_controls_never_fit_validation_or_future_labels(task):
    first = run_controls(task, score_partition="future")
    assert first["future_scored"] and first["groups"] == {"training": 4, "future": 1}
    assert first["fit_partition"] == "training"
    data = model_tasks.load_task(task)
    for partition in ("validation", "future"):
        for row in data[partition]:
            row["target"] += 10
        task["data"][partition] = save(Path(task["data"][partition]["path"]), data[partition])
    second = run_controls(task, score_partition="future")
    assert len(first["outcomes"]) == 6
    for old, new in zip(first["outcomes"], second["outcomes"], strict=True):
        assert old["predictions"] == new["predictions"]
        assert old["metrics"] != new["metrics"]
    with pytest.raises(ValueError, match="training only"):
        run_controls(task, score_partition="training")


def fitted_model(campaign):
    task = campaign.store.specifications()["source"]["task"]
    data = model_tasks.load_task(task)
    return generated_context.fit_candidate(data["training"], task["scope"], {"source_code": CODE, "seed": 0})


def test_prospective_context_prediction_uses_real_sandbox_and_independent_metrics(campaign, monkeypatch):
    evaluation.admit(campaign.store.root, campaign.study, pairs=campaign.pairs)
    finish(campaign, successful=["source"])
    install_publications(monkeypatch, fitted_model(campaign))
    result = evaluation.evaluate(campaign.study, campaign.tmp / "scores")
    assert result["status"] == "completed" and result["future_scored"] == 1
    row = result["rows"][1]
    assert row["final_metrics"]["rmse"] < 1e-12 and row["final_metrics"]["independent_groups"] == 1
    assert evaluation.audit(campaign.study, campaign.tmp / "scores", campaign.tmp / "audit")["status"] == "passed"


@pytest.mark.parametrize("state", ["queued", "running", "paused", "waiting_for_model", "cancelled", "missing_attempt"])
def test_descriptive_admission_cannot_open_future_before_actual_terminal_attempts(campaign, state):
    finish(campaign)
    campaign.store.mutate(
        lambda s: s["allocations"]["none"].update(
            **({"attempts": []} if state == "missing_attempt" else {"status": state})
        )
    )
    with pytest.raises(ValueError, match="every allocated"):
        evaluation.admit_descriptive(campaign.store.root, campaign.study)
    assert not campaign.study.exists()


def test_terminal_descriptive_evaluation_keeps_failed_scope_and_six_controls(campaign, monkeypatch):
    finish(campaign, successful=["source"])
    install_publications(monkeypatch, fitted_model(campaign))
    definition = evaluation.admit_descriptive(campaign.store.root, campaign.study)
    assert definition["pairs"] == [] and len(definition["allocations"]) == 2
    result = evaluation.evaluate(campaign.study, campaign.tmp / "scores")
    assert result["status"] == "completed" and result["admission_protocol"] == "terminal_descriptive.v1"
    assert result["publication_and_reuse_successes"] == result["future_scored"] == 1
    assert result["rows"][0]["original_status"] == "failed" and result["rows"][0]["final_metrics"] is None
    for row in result["rows"]:
        assert row["context_controls"]["status"] == "completed"
        assert len(row["context_controls"]["outcomes"]) == 6
        assert row["context_controls"]["new_autonomous_models"] == 0
    assert evaluation.audit(campaign.study, campaign.tmp / "scores", campaign.tmp / "audit")["status"] == "passed"


@pytest.mark.parametrize("attack", ["campaign", "goal"])
def test_descriptive_acquisition_snapshot_cannot_change_before_scoring(campaign, monkeypatch, attack):
    finish(campaign)
    evaluation.admit_descriptive(campaign.store.root, campaign.study)
    if attack == "campaign":
        campaign.store.mutate(lambda s: s["events"].append({"event": "changed"}))
    else:
        root = campaign.store.goal("none")
        root.mkdir(parents=True, exist_ok=True)
        (root / "new.txt").write_text("changed")
    monkeypatch.setattr(evaluation, "published_models", lambda *a: pytest.fail("Opened future after change"))
    with pytest.raises(ValueError, match="changed after descriptive"):
        evaluation.evaluate(campaign.study, campaign.tmp / "scores")
    assert not (campaign.tmp / "scores").exists()


def test_saved_control_predictions_are_checked_during_audit(campaign, monkeypatch):
    finish(campaign)
    monkeypatch.setattr(evaluation, "published_models", lambda *a: [])
    evaluation.admit_descriptive(campaign.store.root, campaign.study)
    result = evaluation.evaluate(campaign.study, campaign.tmp / "scores")
    asset = result["rows"][0]["context_controls"]["outcomes"][0]["predictions"]
    (campaign.tmp / "scores" / asset["path"]).write_text("[]")
    with pytest.raises(ValueError, match="Archived predictions changed"):
        evaluation.audit(campaign.study, campaign.tmp / "scores", campaign.tmp / "audit")


@pytest.mark.parametrize("campaign", [2], indirect=True)
def test_published_partial_lineage_is_scored_without_claiming_full_goal_success(campaign, monkeypatch):
    finish(campaign)
    model = fitted_model(campaign)

    def partial(store, identity, allocation, record):
        return (
            [
                {
                    "generation": 1,
                    "model_id": digest("generation1"),
                    "entry_id": digest("published1"),
                    "model": copy.deepcopy(model),
                }
            ]
            if identity == "source"
            else []
        )

    monkeypatch.setattr(evaluation, "published_models", partial)
    evaluation.admit_descriptive(campaign.store.root, campaign.study)
    result = evaluation.evaluate(campaign.study, campaign.tmp / "scores")
    assert result["status"] == "completed" and result["future_scored"] == 1
    assert result["publication_and_reuse_successes"] == 0
    assert len(result["rows"][1]["generations"]) == 1
    assert result["rows"][1]["future_lineage_skill_change"] is None
