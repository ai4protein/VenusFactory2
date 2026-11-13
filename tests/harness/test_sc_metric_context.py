"""SC receives same-cohort aggregate facts without query data or approval."""
import copy
import json

import pytest
from test_sc_study_evidence import add_measurement

from agent.harness import scientific_acceptance as sc
from agent.harness.contracts import digest
from agent.harness.research_development import Development
from agent.harness.research_evaluation import scores


def measured(development, index=1, *, evaluation=None):
    row, path = add_measurement(development, index)
    queries = [
        {"id": "PRIVATE_QUERY_A0", "task_id": "PRIVATE_TASK_A", "target": 0,
         "inputs": {"support": [{"target": 2, "id": "PRIVATE_SUPPORT_A"}]}},
        {"id": "PRIVATE_QUERY_A1", "task_id": "PRIVATE_TASK_A", "target": 0,
         "inputs": {"support": [{"target": 2, "id": "PRIVATE_SUPPORT_A"}]}},
        {"id": "PRIVATE_QUERY_B0", "task_id": "PRIVATE_TASK_B", "target": 3,
         "inputs": {"support": [{"target": 3, "id": "PRIVATE_SUPPORT_B"}]}},
    ]
    row["validation"] = scores("regression", [0, 0, 0], [0, 0, 3], [1], query_rows=queries)
    row["validation"]["episodes"]["untrusted_extra"] = queries
    if evaluation:
        row["evaluation_sha256"] = evaluation
    row["receipt_sha256"] = digest({k: v for k, v in row.items() if k != "receipt_sha256"})
    path.write_text(json.dumps(row))
    return row, path


def metric(table, experiment, aggregation, method="model", name="mae"):
    return next(row for row in table["metrics"] if row["experiment_id"] == experiment
                and row["aggregation"] == aggregation and row["method"] == method and row["metric"] == name)


def comparison(table, left, right):
    return next(row for row in table["comparisons"] if row["left_id"] == left["id"] and row["right_id"] == right["id"])


def test_actual_scorer_macro_and_pooled_relations_remain_distinct(tmp_path):
    development = Development(tmp_path)
    row, path = measured(development)
    original = path.read_bytes()
    context = sc.evidence_context({"objective": "Evaluate unbalanced synthetic tasks"}, [], development)
    table = context["metric_evidence"]
    assert table["issues"] == [] and table["omitted_experiments"] == 0
    assert list(context)[:3] == ["objective", "measured_progress", "metric_evidence"]
    pooled, macro = (metric(table, row["id"], a) for a in ("pooled_query", "equal_task"))
    assert pooled["value"] == 1 and macro["value"] == 1.5
    pooled_control, macro_control = (metric(table, row["id"], a, "support_mean") for a in ("pooled_query", "equal_task"))
    assert pooled_control["value"] == pytest.approx(4 / 3) and macro_control["value"] == 1
    assert comparison(table, pooled, pooled_control)["relation"] == "lt"
    assert comparison(table, macro, macro_control)["relation"] == "gt"
    assert not any(item["left_id"] == pooled["id"] and item["right_id"] == macro_control["id"] for item in table["comparisons"])
    assert all(item["receipt_sha256"] == row["receipt_sha256"] for item in table["metrics"])
    assert "PRIVATE_" not in json.dumps(context) and "untrusted_extra" not in json.dumps(context)
    assert path.read_bytes() == original
    prompt = sc.prompt_for_review(context)
    assert "host-computed comparisons" in prompt and "not a usefulness" in prompt
    assert "not automatically an equal-protein" in prompt


@pytest.mark.parametrize("same_evaluation", [True, False])
def test_parent_comparisons_require_matching_frozen_evaluation(tmp_path, same_evaluation):
    development = Development(tmp_path)
    baseline, _ = measured(development)
    child, _ = measured(development, 2, evaluation=None if same_evaluation else "c" * 64)
    context = sc.evidence_context({}, [], development)
    table = context["metric_evidence"]
    ids = {item["id"]: item for item in table["metrics"]}
    pairs = [item for item in table["comparisons"] if ids[item["left_id"]]["experiment_id"] == child["id"]
             and ids[item["right_id"]]["experiment_id"] == baseline["id"]]
    assert bool(pairs) is same_evaluation
    assert all(item["relation"] == "eq" for item in pairs)
    assert not context["scientific_acceptance"]["models"]


def test_authored_scoreboard_and_reference_table_cannot_replace_host_values(tmp_path):
    development = Development(tmp_path)
    measured(development)
    claimed = {"metric_evidence": {"metrics": [{"value": -999}]},
               "supplied_materials": {"files": [{"material_excerpt": "Authored control_score = -999"}]}}
    result = sc.evidence_context(copy.deepcopy(claimed), [], development)
    expected = sc.evidence_context({}, [], development)["metric_evidence"]
    assert result["metric_evidence"] == expected
    assert all(row["value"] >= 0 for row in result["metric_evidence"]["metrics"])
    assert result["supplied_materials"] == claimed["supplied_materials"]
    assert "remain supplied claims" in sc.prompt_for_review(result)


def test_tampered_receipt_cannot_enter_scoreboard(tmp_path):
    development = Development(tmp_path)
    row, path = measured(development)
    row["validation"]["episodes"]["score"]["macro_score"] = 0
    path.write_text(json.dumps(row))
    result = sc.evidence_context({}, [], development)
    assert result["metric_evidence"]["metrics"] == []
    assert result["metric_evidence"]["comparisons"] == []
    assert result["development_measurements"]["receipt_errors"]


def test_scoreboard_retains_history_omission_and_does_not_accept_a_model(tmp_path):
    development = Development(tmp_path)
    for index in range(1, 21):
        measured(development, index)
    result = sc.evidence_context({}, [], development)
    table = result["metric_evidence"]
    assert table["omitted_experiments"] == 8 and len(table["metrics"]) == 96
    review = {"decision": "needs_revision", "summary": "A justified revision is still needed", "issues": [],
              "evidence_ids": [result["development_measurements"]["experiments"][0]["id"]],
              "revision_requests": [{"kind": "scientific_concern", "requirement": "scientific_value", "experiment_id": "",
                                     "reason": "Synthetic execution evidence does not prove biological value",
                                     "proposed_action": "Evaluate the actual biological use case"}]}
    assessment = sc.assess(review, result)
    assert not assessment["passed"] and assessment["approved_models"] == {}
