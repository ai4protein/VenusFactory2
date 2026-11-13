"""Receipt-bound metrics preserve task weighting without exposing query data."""

import copy
import hashlib
import json

import pytest

from agent.harness.research_evaluation import scores
from agent.harness.study_metric_evidence import build_metric_evidence, compare_metrics


def row(validation, identity=1, count=10, evaluation="b" * 64):
    return {
        "id": f"{identity:032x}",
        "receipt_sha256": f"{identity:064x}",
        "evaluation_sha256": evaluation,
        "split_counts": {"train": 20, "validation": count, "test": 8},
        "validation": validation,
        "score_origin": "host_validation",
        "fresh_process_replay": True,
        "test_used_for_feedback": False,
    }


def episode_validation(predictions):
    data = [
        {
            "task_id": "many" if i < 9 else "one",
            "target": 0.0,
            "inputs": {"support": [{"target": 0.0}, {"target": 2.0}]},
        }
        for i in range(10)
    ]
    return scores("regression", predictions, [0.0] * 10, [0.0, 2.0], query_rows=data)


def select(table, *, identity=1, aggregation="pooled_query", method="model", metric="mae"):
    values = [
        x
        for x in table["metrics"]
        if x["experiment_id"] == f"{identity:032x}"
        and x["aggregation"] == aggregation
        and x["method"] == method
        and x["metric"] == metric
    ]
    assert len(values) == 1
    return values[0]


def test_real_host_scores_preserve_opposite_pooled_and_equal_task_relations():
    before = row(episode_validation([2.0] * 9 + [0.0]))
    after = row(episode_validation([0.0] * 9 + [5.0]), identity=2)
    table = build_metric_evidence([before, after])
    assert table["issues"] == []
    pool = compare_metrics(table, select(table, identity=2)["id"], select(table)["id"])
    macro = compare_metrics(
        table, select(table, identity=2, aggregation="equal_task")["id"], select(table, aggregation="equal_task")["id"]
    )
    assert pool["relation"] == "lt" and macro["relation"] == "gt"
    assert (macro["left_value"], macro["right_value"]) == (2.5, 1.0)
    assert (pool["left_value"], pool["right_value"]) == (0.5, 1.8)
    assert not {"approved", "passed", "useful", "significant"} & set(pool)
    for value in table["metrics"]:
        source = before if value["experiment_id"] == before["id"] else after
        assert value["receipt_sha256"] == source["receipt_sha256"]
        assert value["evaluation_sha256"] == source["evaluation_sha256"]
        assert value["split"] == "validation"
        assert value["source_pointer"].startswith("/validation/")


def test_support_references_use_the_same_weighting_and_frozen_cohort():
    table = build_metric_evidence([row(episode_validation([2.0] * 9 + [0.0]))])
    model = select(table, aggregation="equal_task")
    reference = select(table, aggregation="equal_task", method="support_mean")
    assert compare_metrics(table, model["id"], reference["id"])["relation"] == "eq"
    assert model["cohort"] == {"query_count": 10, "task_count": 2}
    assert reference["source_pointer"] == "/validation/episodes/references/support_mean/macro_score"
    with pytest.raises(ValueError, match="aggregation and cohort"):
        compare_metrics(table, select(table)["id"], reference["id"])


def test_projection_is_stable_independent_of_order_and_excludes_private_content():
    first = row(episode_validation([2.0] * 9 + [0.0]))
    second = row(episode_validation([0.0] * 9 + [5.0]), identity=2)
    first.update(source="/private/users/sensitive.py", private_targets=["heldout-secret"])
    first["validation"].update(test={"score": 12345}, multilabel={"labels": ["secret-label"]})
    first["validation"]["episodes"]["queries"] = [{"target": "hidden-query-label", "inputs": "raw-query"}]
    original = copy.deepcopy(first)
    a = build_metric_evidence([first, second])
    b = build_metric_evidence([second, first])
    assert {x["id"] for x in a["metrics"]} == {x["id"] for x in b["metrics"]}
    assert a == build_metric_evidence([first, second]) and first == original
    encoded = json.dumps(a)
    assert all(
        x not in encoded
        for x in ("/private/", "heldout-secret", "secret-label", "hidden-query-label", "raw-query", "12345")
    )


@pytest.mark.parametrize(
    "fault", ["receipt", "evaluation", "id", "origin", "replay", "test", "count", "direction", "metric", "metric_type"]
)
def test_unbound_or_unknown_measurement_is_not_promoted(fault):
    item = row(episode_validation([0.0] * 10))
    if fault in {"receipt", "evaluation"}:
        item.pop(fault + "_sha256")
    elif fault == "id":
        item["id"] = "not-a-record"
    elif fault == "origin":
        item["score_origin"] = "authored_claim"
    elif fault == "replay":
        item["fresh_process_replay"] = False
    elif fault == "test":
        item["test_used_for_feedback"] = True
    elif fault == "count":
        item["split_counts"]["validation"] = True
    elif fault == "direction":
        item["validation"]["direction"] = "maximize"
    elif fault == "metric":
        item["validation"]["metric"] = "invented_quality"
    else:
        item["validation"]["metric"] = []
    result = build_metric_evidence([item])
    assert result["metrics"] == [] and len(result["issues"]) == 1


@pytest.mark.parametrize("value", [True, None, "1.0", -1, float("nan"), float("inf"), 10**400])
def test_nonfinite_boolean_or_unmeasured_values_never_enter_metrics(value):
    item = row(episode_validation([0.0] * 10))
    item["validation"]["score"] = value
    result = build_metric_evidence([item])
    assert result["metrics"] == [] and result["issues"][0]["code"] == "invalid_metric_value"


@pytest.mark.parametrize("fault", ["schema", "query_count", "task_count", "metric", "weighting", "pooled", "macro"])
def test_inconsistent_episode_group_is_omitted_without_erasing_independent_pooled_measurement(fault):
    item = row(episode_validation([2.0] * 9 + [0.0]))
    episode = item["validation"]["episodes"]
    if fault == "schema":
        episode["schema"] = "future-or-authored"
    elif fault == "query_count":
        episode["query_count"] = 9
    elif fault == "task_count":
        episode["task_count"] = 11
    elif fault == "metric":
        episode["metric"] = "accuracy"
    elif fault == "weighting":
        episode["weighting"] = "one equal weight per query"
    elif fault == "pooled":
        episode["score"]["pooled_score"] = 0.5
    else:
        episode["score"]["macro_score"] = float("nan")
    result = build_metric_evidence([item])
    assert result["issues"] and select(result)["value"] == 1.8
    assert all(x["aggregation"] == "pooled_query" and not x["method"].startswith("support_") for x in result["metrics"])


def test_unknown_authored_references_do_not_become_host_controls():
    item = row(episode_validation([0.0] * 10))
    item["reference.md"] = "support_knn_3 measured MAE = 0.0"
    item["validation"]["reference_method"] = "support_knn_3"
    item["validation"]["episodes"]["references"]["support_knn_3"] = {"macro_score": 0.0, "pooled_score": 0.0}
    result = build_metric_evidence([item])
    assert not any(x["method"] == "support_knn_3" for x in result["metrics"])
    assert {x["code"] for x in result["issues"]} == {"unsupported_host_reference", "unsupported_episode_reference"}
    assert select(result, method="support_mean", aggregation="equal_task")


@pytest.mark.parametrize("difference", ["evaluation", "query_count", "task_count", "metric"])
def test_mismatched_evaluation_or_cohort_cannot_be_compared(difference):
    a = row(episode_validation([2.0] * 9 + [0.0]))
    b = row(episode_validation([2.0] * 9 + [0.0]), identity=2)
    if difference == "evaluation":
        b["evaluation_sha256"] = "c" * 64
    elif difference == "query_count":
        b["split_counts"]["validation"] = 11
        b["validation"]["episodes"]["query_count"] = 11
    elif difference == "task_count":
        b["validation"]["episodes"]["task_count"] = 3
    else:
        b = row(scores("classification", ["a"] * 10, ["a"] * 10, ["a"]), identity=2)
    table = build_metric_evidence([a, b])
    aggregation = "equal_task" if difference == "task_count" else "pooled_query"
    left = select(table, aggregation=aggregation)
    right = select(table, identity=2, aggregation=aggregation, metric="accuracy" if difference == "metric" else "mae")
    with pytest.raises(ValueError, match="same evaluation"):
        compare_metrics(table, left["id"], right["id"])


@pytest.mark.parametrize(
    "field", ["value", "receipt_sha256", "evaluation_sha256", "source_pointer", "direction", "aggregation", "cohort_id"]
)
def test_changed_metric_projection_cannot_reuse_an_existing_id(field):
    table = build_metric_evidence([row(episode_validation([0.0] * 10))])
    original = select(table)["id"]
    table["metrics"][0][field] = 123 if field == "value" else "changed"
    with pytest.raises(ValueError, match="binding changed"):
        compare_metrics(table, original, original)


def test_legacy_non_episode_and_classification_keep_declared_metric_semantics():
    table = build_metric_evidence([row(scores("classification", ["a", "b"], ["a", "a"], ["a"]), count=2)])
    value = select(table, metric="accuracy")
    ref = select(table, metric="accuracy", method="training_majority")
    comparison = compare_metrics(table, value["id"], ref["id"])
    assert comparison["relation"] == "lt" and comparison["direction"] == "maximize"
    assert value["aggregation"] == "pooled_query"
    assert table["issues"] == []


def test_multilabel_label_macro_is_not_confused_with_equal_task_macro():
    train = [{"A": 0, "B": 1}, {"A": 1, "B": 0}]
    targets = [{"A": 0, "B": 1}, {"A": 0, "B": None}, {"A": 1, "B": 0}]
    predictions = [{"A": 1.0, "B": 0.0}, {"A": 1.0, "B": 0.0}, {"A": 1.0, "B": 0.0}]
    queries = [
        {
            "task_id": "first" if i < 2 else "second",
            "target": target,
            "inputs": {"support": [{"target": train[0]}, {"target": train[1]}]},
        }
        for i, target in enumerate(targets)
    ]
    validation = scores("multilabel", predictions, targets, train, query_rows=queries)
    table = build_metric_evidence([row(validation, count=3)])
    assert not table["issues"]
    pooled = select(table, metric="macro_brier")
    macro = select(table, metric="macro_brier", aggregation="equal_task")
    assert pooled["value"] == validation["score"]
    assert macro["value"] == validation["episodes"]["score"]["macro_score"]
    assert select(table, metric="macro_brier", method="support_label_prevalence", aggregation="equal_task")
    with pytest.raises(ValueError, match="aggregation and cohort"):
        compare_metrics(table, pooled["id"], macro["id"])


def test_limits_duplicate_id_unknown_id_and_schema_are_rejected():
    item = row(scores("regression", [0.0] * 10, [1.0] * 10, [1.0]))
    with pytest.raises(ValueError, match="bounded experiment"):
        build_metric_evidence([row(item["validation"], identity=i + 1) for i in range(13)])
    with pytest.raises(ValueError, match="Duplicate experiment"):
        build_metric_evidence([item, item])
    table = build_metric_evidence([item])
    identity = select(table)["id"]
    with pytest.raises(ValueError, match="cite metric IDs"):
        compare_metrics(table, "external-table-claim", identity)
    changed = copy.deepcopy(table)
    changed["metrics"].append(changed["metrics"][0])
    with pytest.raises(ValueError, match="Duplicate metric"):
        compare_metrics(changed, identity, identity)
    changed["schema"] = "untrusted"
    with pytest.raises(ValueError, match="schema"):
        compare_metrics(changed, identity, identity)


@pytest.mark.parametrize(
    "field,value", [("source_pointer", "/validation/private_targets"), ("method", "external_markdown_control")]
)
def test_rehashed_projection_still_requires_known_source_semantics(field, value):
    table = build_metric_evidence([row(episode_validation([0.0] * 10))])
    item = table["metrics"][0]
    item[field] = value
    body = {key: value for key, value in item.items() if key != "id"}
    item["id"] = (
        "metric_" + hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    )
    with pytest.raises(ValueError, match="source pointer|metric method"):
        compare_metrics(table, item["id"], item["id"])


def test_invalid_reference_value_is_not_partially_accepted():
    item = row(episode_validation([0.0] * 10))
    item["validation"]["episodes"]["references"]["support_mean"]["pooled_score"] = float("nan")
    table = build_metric_evidence([item])
    assert not any(metric["method"] == "support_mean" for metric in table["metrics"])
    assert select(table, method="support_median", aggregation="equal_task")
    assert table["issues"][0]["code"] == "invalid_metric_value"
