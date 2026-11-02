import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

scripts = Path(__file__).resolve().parents[2] / "scripts/rsi"
sys.path.insert(0, str(scripts))
spec = importlib.util.spec_from_file_location("htfunclib_generated", scripts / "run_htfunclib_generated.py")
campaign = importlib.util.module_from_spec(spec)
spec.loader.exec_module(campaign)


def terminal_fixture(tmp_path, candidates=False):
    arms = [
        {"id": 0, "mode": "science_agent", "status": "candidate_rejected"},
        {"id": 1, "mode": "science_expert", "status": "development_failed"},
    ]
    for arm in arms:
        folder = tmp_path / ("arm_{:03d}".format(arm["id"]))
        folder.mkdir()
        (folder / "terminal.json").write_text(json.dumps(arm))
        if candidates:
            work = folder / "develop/workspace"
            work.mkdir(parents=True)
            (work / "candidate.json").write_text(json.dumps({"recipe": {"skill": -0.4}, "source_sha256": "fixture"}))
    return arms


def test_barrier_requires_every_preallocated_arm_to_terminate(tmp_path):
    arms = terminal_fixture(tmp_path)
    arms[1]["status"] = "running"
    with pytest.raises(ValueError, match="terminate"):
        campaign.seal_barrier(tmp_path, arms)
    assert not (tmp_path / "future_barrier.json").exists()
    with pytest.raises(ValueError, match="terminate"):
        campaign.seal_barrier(tmp_path, arms[:1])


def test_missing_candidates_remain_in_allocation_denominator(tmp_path, monkeypatch):
    arms = terminal_fixture(tmp_path)
    barrier = campaign.seal_barrier(tmp_path, arms)
    monkeypatch.setattr(campaign, "get_model_plugin", lambda _: SimpleNamespace())
    campaign.score_all(tmp_path, arms, barrier, [], [], {})
    assert len(arms) == 2
    for arm in arms:
        assert arm["future_scoring"]["all_allocation_brier_skill"] == 0
        assert arm["future_scoring"]["raw_metrics"] is None
        assert arm["future_scoring"]["status"] == "missing_submitted_candidate"


def test_rejected_and_failed_but_replayable_models_keep_negative_results(tmp_path, monkeypatch):
    arms = terminal_fixture(tmp_path, candidates=True)
    barrier = campaign.seal_barrier(tmp_path, arms)
    model = {"recipe": {"skill": -0.4}, "source_sha256": "fixture"}
    plugin = SimpleNamespace(
        acceptance_metric="brier_skill_vs_training_prior",
        fit=lambda *_: model,
        evaluate=lambda *_: {"brier_skill_vs_training_prior": -0.4},
    )
    monkeypatch.setattr(campaign, "get_model_plugin", lambda _: plugin)
    campaign.score_all(tmp_path, arms, barrier, [], [], {})
    assert [x["future_scoring"]["all_allocation_brier_skill"] for x in arms] == [-0.4, -0.4]
    assert all(x["future_scoring"]["independent_training_replay"] for x in arms)


def test_post_barrier_candidate_or_terminal_changes_are_detected(tmp_path, monkeypatch):
    arms = terminal_fixture(tmp_path, candidates=True)
    barrier = campaign.seal_barrier(tmp_path, arms)
    monkeypatch.setattr(campaign, "get_model_plugin", lambda _: SimpleNamespace())
    for arm in arms:
        (tmp_path / ("arm_{:03d}".format(arm["id"])) / "develop/workspace/candidate.json").write_text("{}")
    campaign.score_all(tmp_path, arms, barrier, [], [], {})
    assert all(x["future_scoring"]["status"] == "unverifiable_or_unscorable_candidate" for x in arms)
    (tmp_path / "arm_000/terminal.json").write_text("{}")
    with pytest.raises(ValueError, match="Frozen allocation changed"):
        campaign.score_all(tmp_path, arms, barrier, [], [], {})
