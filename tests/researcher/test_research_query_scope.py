"""Evidence selection must not turn one database page into an exhaustive survey."""
import copy
import json

import pytest

from backend.web_v2.research_chapter_context import MAX_CONTEXT_CHARS, source_context
from backend.web_v2.research_evidence_passages import resolve_passage, selected_passages, source_digest
from backend.web_v2.research_query_scope import query_result_scope
from backend.web_v2.research_source_review import packet, units


def source(*, identity="S1", count=8, total=1409, long=False):
    value = {"provider": "Open Targets", "status": "completed" if count else "empty",
             "records": [{"tissue": f"Pituitary fixture {i}", "unit": "TPM",
                          "description": "expression observation " * (400 if long else 4)} for i in range(count)],
             "count": count, "total": total, "has_more": count < total,
             "page": 0, "truncated_fields": []}
    return {"id": identity, "title": "Expression query fixture", "kind": "database",
            "query_tool": "query_opentargets", "evidence_level": "database_annotation",
            "query_arguments": {"operation": "target", "identifier": "ENSG00000090382",
                                "resource": "expression", "page": 0, "limit": max(count, 1)},
            "text": json.dumps(value)}


def test_record_ranking_retains_page_coverage_when_total_passage_is_not_selected():
    record = source(long=True)
    original, digest = copy.deepcopy(record), source_digest(record)
    selected = selected_passages(record, "Pituitary expression observation TPM", 14000)
    assert len(selected["passages"]) == 8
    assert all(p["location"].get("json_pointer", "").startswith("/records/") for p in selected["passages"])
    assert selected["query_result_scope"] == {
        "coverage": "database_query", "returned_records": 8, "reported_total": 1409,
        "more_records_available": True, "page": 0, "page_size": 8, "nested_list_truncation": False,
    }
    for passage in selected["passages"]:
        assert resolve_passage(record, passage["location"]) == passage["text"]
    assert record == original and source_digest(record) == digest


def test_chapter_budget_preserves_coverage_without_reinserting_unbounded_records():
    records = [source(identity=f"S{i}", long=True) for i in range(14)]
    selected = source_context(records, query="Pituitary expression observation TPM")
    assert len(selected) == 14 and any(s["excerpted"] for s in selected)
    assert len(json.dumps(selected, ensure_ascii=False)) <= MAX_CONTEXT_CHARS
    assert all(s["query_result_scope"]["reported_total"] == 1409 for s in selected)
    assert all(s["query_result_scope"]["returned_records"] == 8 for s in selected)
    assert all(len(json.dumps(s["query_result_scope"])) < 300 for s in selected)


def test_same_record_coverage_is_available_to_independent_review_packets():
    record = source()
    document = {"id": "expression", "title": "Expression",
                "text": "Pituitary expression observations use TPM. [S1]"}
    request = packet(units(document, [record]), [record])
    assert request["sources"][0]["query_result_scope"] == query_result_scope(record)
    assert request["sources"][0]["source_sha256"] == source_digest(record)


def test_empty_query_does_not_become_a_claim_of_universal_biological_absence():
    result = query_result_scope(source(count=0, total=0))
    assert result["coverage"] == "database_query"
    assert result["returned_records"] == result["reported_total"] == 0
    assert result["more_records_available"] is False
    assert "tissues_absent" not in result and "exhaustive" not in result


@pytest.mark.parametrize("changes", [
    {"count": True}, {"count": "8"}, {"count": 7}, {"count": -1},
    {"total": 7}, {"total": 2**54}, {"total": False}, {"has_more": "true"},
    {"records": {}}, {"status": "unavailable"}, {"status": []},
])
def test_inconsistent_or_untyped_metadata_is_not_presented_as_query_scope(changes):
    record = source()
    payload = json.loads(record["text"])
    payload.update(changes)
    record["text"] = json.dumps(payload)
    assert query_result_scope(record) == {}


def test_unrelated_material_text_and_malformed_json_do_not_gain_database_coverage():
    record = source()
    record.pop("query_tool")
    assert query_result_scope(record) == {}
    record = source()
    record["text"] = "A review mentioned 1409 records."
    assert query_result_scope(record) == {}
    record["text"] = json.dumps([{"count": 8, "total": 1409}])
    assert query_result_scope(record) == {}


def test_optional_page_and_truncation_fields_remain_small_and_do_not_expose_paths():
    record = source()
    payload = json.loads(record["text"])
    payload.update(page=True, truncated_fields=["x" * 10000 for _ in range(50)])
    record["text"] = json.dumps(payload)
    record["query_arguments"]["limit"] = True
    result = query_result_scope(record)
    assert "page" not in result and "page_size" not in result
    assert result["nested_list_truncation"] is True
    assert "x" * 100 not in json.dumps(result) and len(json.dumps(result)) < 300
