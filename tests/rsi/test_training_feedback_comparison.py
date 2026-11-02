"""A declared feedback intervention must keep tasks and resource ceilings matched."""

from copy import deepcopy

import pytest
from test_context_feedback import prepare_goal

from agent.harness.contracts import file_hash
from agent.rsi import context_feedback as feedback
from agent.rsi.engine.campaign_evaluation import admit, load_study, paired_design
from agent.rsi.engine.model_coverage import PROGRAM_ENTRIES
from agent.rsi.campaigns.scientist_model_campaign import ALL_ALLOCATIONS, ModelCampaignStore


def design(tmp_path):
    _, right, _ = prepare_goal(tmp_path, feedback.POLICY)
    left = deepcopy(right)
    left["id"], right["id"] = "feedback_off", "feedback_on"
    left.pop(feedback.FIELD)
    pair = {"id": "synthetic_pair", "left": left["id"], "right": right["id"], "factor": feedback.FIELD}
    return {left["id"]: left, right["id"]: right}, [pair]


def test_admission_and_reload_freeze_feedback_factor_before_any_execution(tmp_path):
    specs, pairs = design(tmp_path)
    assert paired_design(specs, pairs) == pairs
    repo = tmp_path / "repository"
    repo.mkdir()
    (repo / "fixture.py").write_text("# Synthetic admission fixture\n")
    resources = {
        key: sum(a["goal_budget"][key] for a in specs.values()) for key in ("model_calls", "tool_calls", "wall_seconds")
    }
    campaign = ModelCampaignStore.create(
        tmp_path / "campaign",
        objective="Synthetic matched comparison",
        target_models=2,
        library_root=tmp_path / "library",
        repository=repo,
        implementation={"fixture.py": file_hash(repo / "fixture.py")},
        allocations=list(specs.values()),
        resources=resources,
        counting=PROGRAM_ENTRIES,
        execution=ALL_ALLOCATIONS,
    )
    study = admit(campaign.root, tmp_path / "study", pairs=pairs)
    assert load_study(tmp_path / "study") == study
    assert study["pairs"] == pairs
    assert study["include_context_controls"] is True
    assert all(not x["attempts"] for x in campaign.read()["allocations"].values())


@pytest.mark.parametrize(
    "change",
    ["same_policy", "unknown_policy", "model", "tokens", "calls", "cycles", "generations", "task", "experience"],
)
def test_mismatched_feedback_pairs_are_rejected(tmp_path, change):
    specs, pairs = design(tmp_path)
    right = specs["feedback_on"]
    if change == "same_policy":
        right[feedback.FIELD] = "none"
    elif change == "unknown_policy":
        right[feedback.FIELD] = "future_labels"
    elif change == "model":
        right["profile"]["model"]["model"] = "another-model"
    elif change == "tokens":
        right["profile"]["budget"]["output_tokens"] += 1
    elif change == "calls":
        right["goal_budget"]["model_calls"] += 1
    elif change == "cycles":
        right["goal_budget"]["max_cycles"] += 1
    elif change == "generations":
        right["generations"] += 1
    elif change == "task":
        right["task"]["minimum_skill"] += 0.1
    else:
        right["experience"] = {"condition": "source", "selection": "explicit", "entry_ids": []}
    with pytest.raises(ValueError):
        paired_design(specs, pairs)
