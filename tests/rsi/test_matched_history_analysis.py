"""Allocation, missingness and cluster-aware inference in matched studies."""

from copy import deepcopy
from itertools import permutations

import pytest

from agent.rsi.matched_history_analysis import ARMS, interval, summarize, validate_design
from agent.rsi.protein_model import digest


def fixture():
    design = {
        "units": [],
        "condition_policies": {a: a + ".v1" for a in ARMS},
        "free_model_only": {"model": "free-fixture"},
        "budget_per_unit": {"max_cycles": 8, "model_calls": 144, "tool_calls": 72, "wall_seconds": 5400},
        "tasks_per_block": 2,
        "matched_blocks": 6,
        "scientific_scopes": 12,
        "allocated_model_attempts": 36,
        "source_projects": ["source"],
    }
    plans, rows = {}, {}
    for b, order in enumerate(permutations(ARMS)):
        tasks = [
            {"id": f"t{b}_{i}", "plugin_id": "generated_context.v2", "scope": {"protein": f"synthetic{b}_{i}"}}
            for i in range(2)
        ]
        for arm in order:
            uid = f"b{b}_{arm}"
            design["units"].append({"id": uid, "arm": arm, "project": "source", "source_block": str(b)})
            plans[uid] = {
                "target_scopes": 2,
                "resources": {k: v for k, v in design["budget_per_unit"].items() if k != "max_cycles"},
                "batches": [
                    {
                        "tasks": deepcopy(tasks),
                        "shared_training_history": arm + ".v1",
                        "goal_budget": dict(design["budget_per_unit"]),
                        "profile": {"model": design["free_model_only"], "harness": "fixture"},
                    }
                ],
            }
            rows[uid] = []
            for task in tasks:
                acquired = arm != "no_history" or task["id"].endswith("0")
                scientist = {"status": "not_registered", "metrics": None, "predictions": None}
                if acquired:
                    scientist = {
                        "status": "scored",
                        "metrics": {
                            "rmse": 1.0,
                            "mae": 0.8,
                            "spearman": None,
                            "mse_skill_vs_training_mean": 0.5 if arm == "selective" else 0.2,
                        },
                        "predictions": [],
                    }
                rows[uid].append({"task_id": task["id"], "task_sha256": digest(task), "scientist": scientist})
    return design, plans, rows


def test_denominator_scope_and_missing_quality():
    d, p, r = fixture()
    report = summarize(d, p, r)
    assert report["scientific_scopes"] == report["distinct_registered_scopes"] == 12
    assert report["allocated_model_attempts"] == 36
    assert report["arms"]["no_history"] == {
        "allocated": 12,
        "registered_and_independently_reused": 6,
        "not_registered": 6,
    }
    contrast = report["contrasts"]["selective_minus_no_history"]
    assert contrast["acquisition"]["estimate"] == 0.5
    quality = contrast["conditional_paired_quality"]
    assert quality["mse_skill_vs_training_mean"]["pooled"]["paired_tasks"] == 6
    assert quality["spearman"]["pooled"]["estimate"] is None
    assert quality["spearman"]["pooled"]["paired_tasks"] == 0
    assert "pooled" not in quality["rmse"]


def test_zero_acquisition_preserves_denominators():
    d, p, r = fixture()
    for rows in r.values():
        for row in rows:
            row["scientist"] = {"status": "not_registered", "metrics": None, "predictions": None}
    report = summarize(d, p, r)
    assert report["distinct_registered_scopes"] == 0
    assert report["arms"]["selective"]["allocated"] == 12
    assert report["contrasts"]["selective_minus_recent"]["acquisition"]["degenerate"] is True


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_arm",
        "duplicate_unit",
        "changed_task",
        "changed_budget",
        "changed_harness",
        "wrong_policy",
        "repeat_scope",
        "unbalanced_order",
    ],
)
def test_reject_incomparable_design(mutation):
    d, p, _ = fixture()
    uid = d["units"][0]["id"]
    if mutation == "missing_arm":
        d["units"].pop(0)
        p.pop(uid)
    elif mutation == "duplicate_unit":
        d["units"].append(d["units"][0])
    elif mutation == "changed_task":
        p[uid]["batches"][0]["tasks"][0]["scope"]["protein"] = "different"
    elif mutation == "changed_budget":
        p[uid]["batches"][0]["goal_budget"]["model_calls"] = 145
    elif mutation == "changed_harness":
        p[uid]["batches"][0]["profile"]["harness"] = "different"
    elif mutation == "wrong_policy":
        p[uid]["batches"][0]["shared_training_history"] = "different"
    elif mutation == "repeat_scope":
        for u in d["units"][3:6]:
            p[u["id"]]["batches"][0]["tasks"][0]["scope"] = deepcopy(p[uid]["batches"][0]["tasks"][0]["scope"])
    else:
        d["units"][3:6] = sorted(d["units"][3:6], key=lambda u: ARMS.index(u["arm"]))
    with pytest.raises(ValueError):
        validate_design(d, p)


@pytest.mark.parametrize(
    "mutation",
    ["missing_unit", "missing_task", "wrong_task_hash", "nan", "unknown_status", "missing_rmse", "score_without_model"],
)
def test_reject_missing_or_invalid_results(mutation):
    d, p, r = fixture()
    uid = "b0_selective"
    row = r[uid][0]
    if mutation == "missing_unit":
        r.pop(uid)
    elif mutation == "missing_task":
        r[uid].pop()
    elif mutation == "wrong_task_hash":
        row["task_sha256"] = "0" * 64
    elif mutation == "nan":
        row["scientist"]["metrics"]["rmse"] = float("nan")
    elif mutation == "unknown_status":
        row["scientist"]["status"] = "failed"
    elif mutation == "missing_rmse":
        row["scientist"]["metrics"]["rmse"] = None
    else:
        row["scientist"]["status"] = "not_registered"
    with pytest.raises(ValueError):
        summarize(d, p, r)


def test_bootstrap_is_block_paired_stratified_and_order_invariant():
    values = [("a", "1", -1.0), ("a", "2", 1.0), ("b", "1", 10.0), ("b", "2", 10.0)]
    answer = interval(values)
    assert answer == interval(list(reversed(values)))
    assert answer["estimate"] == 5.0
    assert answer["source_blocks"] == {"a": 2, "b": 2}
    assert answer["interval_95"] == [4.5, 5.5]
    assert answer["blocks"] == 4


def test_single_block_does_not_fabricate_uncertainty():
    assert interval([("a", "1", 0.1)])["interval_95"] is None
    assert interval([])["estimate"] is None
