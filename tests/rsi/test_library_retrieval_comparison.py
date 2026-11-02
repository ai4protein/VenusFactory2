"""Source selection can vary while task evidence and recipient budgets stay fixed."""

from copy import deepcopy

import pytest
from test_campaign_evaluation import campaign as campaign  # noqa: F401
from test_scientist_model_campaign import task as task  # noqa: F401

from agent.rsi import campaign_evaluation as evaluation
from agent.rsi.campaigns.scientist_model_campaign import ALL_ALLOCATIONS, ModelCampaignStore


def three_arms(campaign):
    original = campaign.store.specifications()
    specifications = {
        "reset": deepcopy(original["none"]),
        "all_sources": deepcopy(original["source"]),
        "screened": deepcopy(original["source"]),
    }
    for name, row in specifications.items():
        row["id"] = name
        row["experience"]["entry_ids"] = ["a" * 64, "b" * 64, "c" * 64, "d" * 64]
    specifications["screened"]["experience"]["entry_ids"] = ["a" * 64, "c" * 64]
    return specifications


def comparisons():
    return [
        {"id": left + "_minus_" + right, "left": left, "right": right, "factor": evaluation.LIBRARY_RETRIEVAL}
        for left, right in [("screened", "reset"), ("screened", "all_sources"), ("all_sources", "reset")]
    ]


def test_three_arm_retrieval_is_admitted_without_mutating_inputs(campaign, tmp_path):
    specs = three_arms(campaign)
    before = deepcopy(specs)
    pairs = comparisons()
    assert evaluation.paired_design(specs, pairs) == pairs
    assert specs == before
    definition = campaign.store.read()["definition"]
    store = ModelCampaignStore.create(
        tmp_path / "retrieval_campaign",
        objective="Synthetic retrieval ablation",
        target_models=3,
        library_root=tmp_path / "retrieval_library",
        repository=definition["repository"],
        implementation=definition["implementation"],
        allocations=list(specs.values()),
        resources={"model_calls": 36, "tool_calls": 24, "wall_seconds": 180},
        counting="program_entry.v1",
        execution=ALL_ALLOCATIONS,
    )
    study = evaluation.admit(store.root, tmp_path / "retrieval_study", pairs=pairs)
    assert len(study["allocations"]) == 3 and study["pairs"] == pairs
    assert not any(item["attempts"] for item in store.read()["allocations"].values())


@pytest.mark.parametrize(
    "change", ["budget", "profile", "task", "memory", "feedback", "generations", "coverage", "policy"]
)
def test_retrieval_does_not_allow_other_intervention_changes(campaign, change):
    specs = three_arms(campaign)
    row = specs["screened"]
    if change == "budget":
        row["goal_budget"]["model_calls"] += 1
    elif change == "profile":
        row["profile"]["budget"]["temperature"] = 0.5
    elif change == "task":
        row["task"]["minimum_skill"] += 0.1
    elif change == "memory":
        row["training_attempt_memory"] = "cumulative_training_attempts.v2"
    elif change == "feedback":
        row["training_baseline_feedback"] = "none"
    elif change == "generations":
        row["generations"] += 1
    elif change == "coverage":
        row["coverage"] = {"candidate_id": "changed"}
    else:
        row["experience"]["overlap_policy"] = "context_study_disjoint.v1"
    with pytest.raises(ValueError):
        evaluation.paired_design(specs, comparisons())


@pytest.mark.parametrize(
    "change", ["same", "reordered", "both_hidden", "empty_source", "summary", "dynamic", "invalid_id"]
)
def test_invalid_or_identical_exposure_is_rejected(campaign, change):
    specs = three_arms(campaign)
    selected = specs["screened"]["experience"]
    if change == "same":
        selected["entry_ids"] = specs["all_sources"]["experience"]["entry_ids"][:]
    elif change == "reordered":
        selected["entry_ids"] = specs["all_sources"]["experience"]["entry_ids"][::-1]
    elif change == "both_hidden":
        selected["condition"] = "none"
        specs["all_sources"]["experience"]["condition"] = "none"
    elif change == "empty_source":
        selected["entry_ids"] = []
        specs["all_sources"]["experience"]["condition"] = "none"
    elif change == "summary":
        selected["condition"] = "summary"
    elif change == "dynamic":
        selected.update(selection="prior_campaign", max_entries=2)
    else:
        selected["entry_ids"] = ["invalid"]
    with pytest.raises(ValueError):
        evaluation.paired_design(specs, [comparisons()[1]])


def test_legacy_retention_factor_still_requires_identical_donors(campaign):
    specs = three_arms(campaign)
    pair = {"id": "legacy", "left": "all_sources", "right": "reset", "factor": "library_experience"}
    assert evaluation.paired_design(specs, [pair]) == [pair]
    pair["left"] = "screened"
    with pytest.raises(ValueError, match="donors"):
        evaluation.paired_design(specs, [pair])
