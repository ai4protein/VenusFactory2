import json
from copy import deepcopy

import pytest

from agent.rsi import matched_failure_evaluation as module
from agent.rsi.engine.matched_history_evidence import Evidence
from agent.rsi.campaigns.matched_history_operations import duration_totals, resource_accounting
from agent.rsi.engine.protein_model import digest


def test_unknown_failure_reservation_is_not_physical_usage():
    campaign = {
        "usage": {"model_calls": 144, "tool_calls": 72, "wall_seconds": 5400},
        "batches": {"b": {"status": "failed", "attempts": [{"publication": None}]}},
    }
    original = deepcopy(campaign)
    terminal = {"unpublished_failure": True, "classification": "unpublished_failure", "usage": campaign["usage"]}
    state = {"usage": {"model_calls": 63, "tool_calls": 30, "wall_seconds": 900}}
    result = resource_accounting(campaign, terminal, state, 63, 30)
    assert result["campaign_wall_seconds"] is None
    assert result["charged_usage"] == original["usage"] and campaign == original
    with pytest.raises(ValueError, match="observed usage"):
        resource_accounting(campaign, terminal, state, 144, 72)
    with pytest.raises(ValueError, match="Unit request"):
        resource_accounting(campaign, {"usage": campaign["usage"]}, state, 63, 30)


def test_unknown_wall_cost_prevents_total_but_retains_known_sum():
    assert duration_totals([{"campaign_wall_seconds": 20}, {"campaign_wall_seconds": None}]) == {
        "campaign_wall_seconds": None,
        "known_campaign_wall_seconds": 20,
        "unknown_campaign_wall_units": 1,
    }
    assert duration_totals([{"campaign_wall_seconds": 20}, {"campaign_wall_seconds": 5}])["campaign_wall_seconds"] == 25


@pytest.fixture
def failed_report(tmp_path, monkeypatch):
    folder = tmp_path / "unit"
    output = folder / "evaluation"
    output.mkdir(parents=True)
    task = {"id": "task_a", "scope": {"protein": "fixture"}}
    plan = {"batches": [{"tasks": [task]}]}
    unit = {"plan": str(folder / "plan.json")}
    closeout = {"path": str(tmp_path / "closeout.json"), "sha256": "pinned"}
    row = {
        "task_id": "task_a",
        "comparison": None,
        "scientist": {"status": "not_registered", "metrics": None, "predictions": None},
        "controls": [{"status": "completed"}],
    }

    def save(path, value):
        path.write_text(json.dumps(value))
        return module.sha(path)

    protocol = {
        "terminal_closeout": closeout,
        "fit_partition": "training",
        "score_partition": "future",
        "hyperparameter_search": False,
        "source_bindings": {},
        "implementation": {},
        "task_bindings": {"task_a": digest(task)},
        "tasks": [task],
    }
    report = {
        "schema": module.shared_context_evaluation.SCHEMA,
        "status": "unpublished_failure_controls_scored_pending_replay",
        "final_labels_scored": True,
        "protocol_sha256": save(output / "protocol.json", protocol),
        "task_reports": {"task_a": save(output / "task_a.json", row)},
        "totals": module.failure_totals([row]),
    }
    paths = {"evaluation": {"path": str(output / "report.json"), "sha256": save(output / "report.json", report)}}
    calls = []

    def verify(evidence, assigned, observed, prediction):
        assert assigned == task and prediction is None
        calls.append(observed)
        return {"verified": True}

    monkeypatch.setattr(module, "verify_task", verify)
    return Evidence(tmp_path), unit, plan, paths, closeout, calls, save, output


def test_failed_task_has_controls_but_no_scientist_quality(failed_report):
    evidence, unit, plan, paths, closeout, calls, _, _ = failed_report
    rows, refits = module.failed_unit_evidence(evidence, unit, plan, paths, closeout)
    assert len(rows) == len(calls) == len(refits) == 1
    assert rows[0]["scientist"]["metrics"] is None


@pytest.mark.parametrize(
    "mutation",
    ["fake_reuse", "omitted_task", "changed_closeout", "quality_imputation", "wrong_partition", "wrong_count"],
)
def test_failed_evidence_cannot_drop_tasks_or_claim_success(failed_report, mutation):
    evidence, unit, plan, paths, closeout, calls, save, output = failed_report
    report = json.loads((output / "report.json").read_text())
    if mutation == "fake_reuse":
        paths["reuse"] = {"path": "unpublished"}
    elif mutation == "omitted_task":
        report["task_reports"] = {}
    elif mutation == "changed_closeout":
        closeout = {**closeout, "sha256": "different"}
    elif mutation == "quality_imputation":
        row = json.loads((output / "task_a.json").read_text())
        row["scientist"]["metrics"] = {"rmse": 0}
        report["task_reports"]["task_a"] = save(output / "task_a.json", row)
    elif mutation == "wrong_partition":
        protocol = json.loads((output / "protocol.json").read_text())
        protocol["fit_partition"] = "future"
        report["protocol_sha256"] = save(output / "protocol.json", protocol)
    else:
        report["totals"]["allocated"] = 0
    paths["evaluation"]["sha256"] = save(output / "report.json", report)
    with pytest.raises(ValueError):
        module.failed_unit_evidence(evidence, unit, plan, paths, closeout)


def test_global_barrier_precedes_any_failed_unit_scoring(tmp_path, monkeypatch):
    def block(*args, **kwargs):
        raise ValueError("live acquisition")

    def forbidden(*args, **kwargs):
        pytest.fail("Scored before the barrier")

    monkeypatch.setattr(module, "verify", block)
    monkeypatch.setattr(module.shared_context_evaluation, "evaluate_task", forbidden)
    with pytest.raises(ValueError, match="live acquisition"):
        module.score_unpublished(tmp_path, {}, "unit", tmp_path / "score", repository=tmp_path)
    assert not (tmp_path / "score").exists()
