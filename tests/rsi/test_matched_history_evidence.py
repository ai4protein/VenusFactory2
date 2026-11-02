"""Terminal barriers, file binding and independent scalar metric replay."""

import hashlib
import json

import numpy as np
import pytest

from agent.rsi import matched_history_evidence as module
from agent.rsi.prediction_metrics import regression_scores
from agent.rsi.shared_sequence_evaluation import score_predictions


def write(path, value):
    path.write_text(json.dumps(value))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_bindings_reject_modified_files(tmp_path):
    p = tmp_path / "evidence.json"
    sha = write(p, {"a": 1})
    reader = module.Evidence(tmp_path)
    assert reader.read(p, sha) == {"a": 1}
    write(p, {"a": 2})
    with pytest.raises(ValueError, match="changed"):
        reader.unchanged()


@pytest.mark.parametrize(
    "relative",
    [
        "../outside.json",
        ".env",
        "roles/MLS/private_home/a.json",
        "research_runs/rsi/20261014_improver_revision_glm52_v1/a.json",
    ],
)
def test_excluded_paths_are_rejected_before_read(tmp_path, relative):
    with pytest.raises(ValueError):
        module.Evidence(tmp_path).read(tmp_path / relative)


def test_symlink_evidence_is_rejected(tmp_path):
    p = tmp_path / "data.json"
    write(p, {})
    link = tmp_path / "link.json"
    link.symlink_to(p)
    with pytest.raises(ValueError, match="Symlink"):
        module.Evidence(tmp_path).read(link)


@pytest.mark.parametrize(
    "live,status,completed",
    [
        (True, "terminal_pending_independent_reuse", []),
        (False, "active", []),
        (False, "failed", []),
        (False, "terminal_pending_independent_reuse", []),
    ],
)
def test_global_barrier_precedes_every_label_or_result_read(tmp_path, monkeypatch, live, status, completed):
    study = tmp_path / "study"
    study.mkdir()
    design = {"units": [{"id": "u"}], "final_protocol_sha256": "unused"}
    dsha = write(study / "study_design.json", design)
    lsha = write(study / "launch_basis.json", {"study_design_sha256": dsha, "final_protocol_sha256": "unused"})
    write(study / "runner_identity.json", {"study": str(study)})
    write(study / "runner_state.json", {"status": status, "completed_units": completed})
    monkeypatch.setattr(module, "process_live", lambda _: live)
    reader = module.Evidence(tmp_path)
    with pytest.raises(ValueError, match="Global terminal barrier"):
        module.terminal_study(reader, study, lsha)
    assert {p.rsplit("/", 1)[-1] for p in reader.bindings} == {
        "launch_basis.json",
        "study_design.json",
        "runner_identity.json",
        "runner_state.json",
    }


@pytest.mark.parametrize(
    "truth,predicted",
    [
        ([1, 2, 2, 4], [2, 3, 3, 1]),
        ([1, 1, 1, 1], [1, 2, 3, 4]),
        ([1, 2, 3, 4], [2, 2, 2, 2]),
        ([1, 2, 3, 4], [1, 1 + 1e-13, 1 + 2e-13, 1 + 3e-13]),
    ],
)
def test_independent_metrics_match_frozen_scoring_including_ties(truth, predicted):
    train = [{"id": "tr" + str(i), "target": v} for i, v in enumerate([0.0, 2.0, 4.0])]
    future = [{"id": str(i), "target": v} for i, v in enumerate(truth)]
    predictions = [{"id": str(i), "prediction": v} for i, v in enumerate(predicted)]
    observed = module.replay_metrics(train, future, predictions)
    expected = regression_scores(np.array([0.0, 2.0, 4.0]), np.array(truth), np.array(predicted))
    module.equal_metrics(expected, observed)
    control = module.replay_metrics(train, future, predictions, controls=True)
    module.equal_metrics(score_predictions(future, predictions, 2.0), control)


def test_degenerate_baseline_is_missing_not_zero():
    value = module.replay_metrics(
        [{"target": 1}],
        [{"id": "a", "target": 1}, {"id": "b", "target": 1}],
        [{"id": "a", "prediction": 1}, {"id": "b", "prediction": 2}],
    )
    assert value["mse_skill_vs_training_mean"] is None
    assert value["spearman"] is None
    assert value["rmse"] > 0


def test_prediction_order_and_metric_tampering_rejected():
    with pytest.raises(ValueError, match="identity"):
        module.replay_metrics([{"target": 0}], [{"id": "a", "target": 1}], [{"id": "b", "prediction": 0}])
    good = {"rmse": 1.0, "mae": 1.0, "spearman": None, "mse_skill_vs_training_mean": 0.0}
    with pytest.raises(ValueError, match="replay"):
        module.equal_metrics({**good, "spearman": 0.0}, good)


def test_unpublished_score_is_rejected_before_control_refit(tmp_path, monkeypatch):
    task = {"scope": {}, "data": {}, "references": []}
    row = {
        "task_sha256": module.digest(task),
        "scope": {},
        "data_bindings": {},
        "score_partition": "future",
        "fit_partition": "training",
        "future_scored": True,
    }
    with pytest.raises(ValueError, match="Three partitions"):
        module.verify_task(module.Evidence(tmp_path), task, row, None)


@pytest.fixture
def unit_fixture(tmp_path, monkeypatch):
    folder = tmp_path / "unit"
    reuse, evaluation = folder / "reuse", folder / "evaluation"
    reuse.mkdir(parents=True)
    evaluation.mkdir()
    task = {"id": "synthetic_task", "scope": {}, "data": {}, "references": []}
    prediction = [{"id": "q", "prediction": 1.0}]
    campaign = {
        "batches": {"b": {"attempts": [{"publication": {"outcomes": [{"task_id": task["id"], "entry_id": "entry"}]}}]}}
    }
    rp = {
        "source_bindings": {},
        "task_bindings": [{"task_id": task["id"], "entry_id": "entry", "prediction_sha256": module.digest(prediction)}],
    }
    result = {"results": [{"receipt": {"entry_id": "entry"}, "predictions": prediction}]}
    ep = {
        "source_bindings": {},
        "implementation": {},
        "tasks": [task],
        "task_bindings": {task["id"]: module.digest(task)},
    }
    row = {"task_id": task["id"]}
    row_sha = write(evaluation / (task["id"] + ".json"), row)
    rsha = write(
        reuse / "report.json",
        {
            "status": "independent_api_reuse_passed",
            "protocol_sha256": write(reuse / "protocol.json", rp),
            "results_sha256": write(reuse / "inference_results.json", result),
        },
    )
    write(reuse / "exit.json", {"returncode": 0})
    write(reuse / "independent_receipt_audit.json", {"status": "passed", "report_sha256": rsha, "checked_files": {}})
    esha = write(
        evaluation / "report.json",
        {
            "status": "scored_pending_independent_replay",
            "final_labels_scored": True,
            "protocol_sha256": write(evaluation / "protocol.json", ep),
            "task_reports": {task["id"]: row_sha},
        },
    )
    paths = {
        "reuse": {"path": str(reuse / "report.json"), "sha256": rsha},
        "evaluation": {"path": str(evaluation / "report.json"), "sha256": esha},
    }
    # Numeric replay is checked on real completed data separately; here isolate
    # receipt identity, exact coverage and immutable binding checks.
    calls = []

    def replay(reader, supplied_task, supplied_row, supplied_prediction):
        assert supplied_task == task and supplied_row == row and supplied_prediction == prediction
        calls.append(task["id"])
        return {"status": "fixture_verified"}

    monkeypatch.setattr(module, "verify_task", replay)
    return module.Evidence(tmp_path), {"batches": [{"tasks": [task]}]}, (folder, campaign, {}), paths, calls


def test_unit_reader_accepts_exact_published_reuse_coverage(unit_fixture):
    reader, plan, context, paths, calls = unit_fixture
    rows, checks = module.unit_evidence(reader, {}, plan, context, paths)
    assert len(rows) == len(checks) == len(calls) == 1


@pytest.mark.parametrize(
    "mutation",
    [
        "duplicate_publication",
        "wrong_entry",
        "removed_publication",
        "stale_reuse",
        "changed_row",
        "failed_receipt",
        "failed_exit",
    ],
)
def test_unit_reader_rejects_missing_or_modified_evidence(unit_fixture, mutation):
    reader, plan, context, paths, calls = unit_fixture
    folder, campaign, _ = context
    publication = campaign["batches"]["b"]["attempts"][0]["publication"]["outcomes"]
    if mutation == "duplicate_publication":
        publication.append(dict(publication[0]))
    elif mutation == "wrong_entry":
        publication[0]["entry_id"] = "wrong"
    elif mutation == "removed_publication":
        publication.clear()
    elif mutation == "stale_reuse":
        write(folder / "reuse/report.json", {})
    elif mutation == "changed_row":
        write(folder / "evaluation/synthetic_task.json", {"task_id": "other"})
    elif mutation == "failed_receipt":
        write(folder / "reuse/independent_receipt_audit.json", {"status": "failed"})
    else:
        write(folder / "reuse/exit.json", {"returncode": 1})
    with pytest.raises(ValueError):
        module.unit_evidence(reader, {}, plan, context, paths)
    assert not calls
