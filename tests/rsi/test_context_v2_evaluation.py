"""Terminal scoring with real wide-context predictions and synthetic publication boundaries."""

import pytest
from test_campaign_evaluation import campaign as campaign
from test_campaign_evaluation import finish, install_publications
from test_context_v2 import CODE, make_task

from agent.rsi import campaign_evaluation as evaluation
from agent.rsi import generated_context_v2 as api
from agent.rsi import model_tasks


@pytest.fixture
def task(tmp_path):
    return make_task(tmp_path / "wide_task")[0]


def test_wide_terminal_evaluation_preserves_failures_and_independent_scores(campaign, monkeypatch):
    definition = evaluation.admit(campaign.store.root, campaign.study, pairs=campaign.pairs)
    assert definition["include_context_controls"] is True
    finish(campaign, successful=["source"])
    task = campaign.store.specifications()["source"]["task"]
    data = model_tasks.load_task(task)
    model = api.fit_candidate(data["training"], task["scope"], {"source_code": CODE, "seed": 0})
    install_publications(monkeypatch, model)
    result = evaluation.evaluate(campaign.study, campaign.tmp / "scores")
    assert result["status"] == "completed" and result["allocated"] == 2 and result["future_scored"] == 1
    success = next(r for r in result["rows"] if r["allocation_id"] == "source")
    failed = next(r for r in result["rows"] if r["allocation_id"] == "none")
    assert success["final_metrics"]["rmse"] < 1e-11
    assert success["final_metrics"]["grouping_level"] == "deleted_gene"
    assert failed["final_metrics"] is None and failed["original_status"] == "failed"
    assert all(len(r["context_controls"]["outcomes"]) == 4 for r in result["rows"])
    assert evaluation.audit(campaign.study, campaign.tmp / "scores", campaign.tmp / "audit")["status"] == "passed"
