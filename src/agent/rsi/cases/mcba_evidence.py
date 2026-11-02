"""Complete released-notebook scope, separate from agreement and new evidence."""

from __future__ import annotations

PRODUCTS = (
    "moclobemide",
    "metoclopramide",
    "itopride",
    "cinchocaine",
    "declopramide",
    "trimethobenzamide",
    "sulpiride",
    "procainamide",
    "troxipide",
)
SUBSETS = ("full", "BLOSUM1", "BLOSUM2", "BLOSUM3", "BLOSUM4", "NDT", "NRT", "AGCP")
COMMIT = "0882f67317f56dfd2a39474f8f548e73197c1e8d"
TASK_IDS = tuple(["rank_" + p for p in PRODUCTS] + ["compare_" + p + "_" + s for p in PRODUCTS[:2] for s in SUBSETS])


def summarize_campaign(original, audit):
    if (
        original.get("status") != "campaign_completed"
        or audit.get("campaign_completed") is not True
        or audit.get("source_integrity_passed") is not True
    ):
        raise ValueError("Terminal campaign with source-verified audit required")
    raw = original.get("tasks", [])
    replay = audit.get("tasks", [])
    if (
        len(raw) != len(TASK_IDS)
        or len(replay) != len(TASK_IDS)
        or {x["id"] for x in raw} != set(TASK_IDS)
        or {x["id"] for x in replay} != set(TASK_IDS)
    ):
        raise ValueError("All 25 notebook task identities must be independently audited")
    if any(x["status"] != "completed" for x in raw) or audit.get("all_selected_tasks_completed") is not True:
        raise ValueError("Not all notebook computations completed")
    indexed = {x["id"]: x for x in replay}
    numerical_failures, metric_failures, historical_failures, unavailable = [], [], [], []
    historical_passes, rank_order_passes = 0, 0
    for row in raw:
        check = indexed[row["id"]]
        rank = row["id"].startswith("rank_")
        if check.get("architectures") != (1 if rank else 40) or check.get("prediction_rows_per_architecture") != 160000:
            raise ValueError("Incomplete architecture or combinatorial-grid coverage")
        if type(check.get("numerical_prediction_replay_passed")) is not bool:
            raise ValueError("Missing independent numerical decision")
        if check["numerical_prediction_replay_passed"] is False:
            numerical_failures.append(row["id"])
        if check.get("test_metric_discrepancies"):
            metric_failures.append(row["id"])
        reference = row.get("reference_passed")
        available_reference = rank or row["id"].endswith("_full")
        if (available_reference and type(reference) is not bool) or (not available_reference and reference is not None):
            raise ValueError("Reference availability differs from the released notebooks")
        if check.get("historical_reference_passed") is not reference:
            raise ValueError("Historical-reference result changed")
        if reference is True:
            historical_passes += 1
        elif reference is False:
            historical_failures.append(row["id"])
        else:
            unavailable.append(row["id"])
        if rank and row.get("top_exact_order_match") is True:
            rank_order_passes += 1
    numerical_failures.sort()
    expected_prediction_pass = not numerical_failures
    if audit.get("independent_prediction_replay_passed") is not expected_prediction_pass:
        raise ValueError("Aggregate replay decision conflicts with task evidence")
    return {
        "allocated_tasks": 25,
        "completed_tasks": 25,
        "protein_scaffolds": 1,
        "ranked_products": 9,
        "comparison_tasks": 16,
        "architectures_per_comparison": 40,
        "ranking_cached_order_passes": rank_order_passes,
        "independent_prediction_replay_passed": expected_prediction_pass,
        "numerical_discrepancy_tasks": numerical_failures,
        "metric_discrepancy_tasks": sorted(metric_failures),
        "historical_reference_counts": {
            "passed": historical_passes,
            "failed": len(historical_failures),
            "unavailable": len(unavailable),
        },
        "historical_discrepancy_tasks": sorted(historical_failures),
        "computational_agreement_passed": not (numerical_failures or metric_failures or historical_failures)
        and rank_order_passes == 9,
    }


def canonical_mcba(report):
    if (
        report.get("execution_origin") not in {"new_computation", "verified_existing"}
        or report.get("source_commit") != COMMIT
    ):
        raise ValueError("McbA provenance missing or changed")
    summary = summarize_campaign(report["campaign_evidence"], report["audit_evidence"])
    if summary != report.get("summary"):
        raise ValueError("McbA summary differs from independently checked tasks")
    return {
        "computational_scope": "mcba_released_notebooks_processed_inputs",
        "execution_origin": report["execution_origin"],
        "source_campaign": report["source_campaign"],
        "source_commit": COMMIT,
        **summary,
        "upstream_zero_shot_regenerated": False,
        "repeat_computation_is_independent_efficacy_evidence": False,
        "new_algorithm_or_agent_superiority_established": False,
    }
