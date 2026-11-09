"""Evidence retrieval keeps source qualifiers when a draft names the wrong subject."""
import copy
import json

import pytest

from backend.web_v2.research_evidence_passages import resolve_passage, selected_passages


def source(data):
    return {"id": "S1", "kind": "database", "title": "Public annotation", "text": json.dumps(data)}


def assert_anchors(original, selected, budget):
    passages = selected["passages"]
    assert len(passages) <= 8
    assert sum(len(row["text"]) for row in passages) <= budget
    assert len({row["id"] for row in passages}) == len(passages)
    assert all(resolve_passage(original, row["location"]) == row["text"] for row in passages)
    assert all(not key.startswith("_") for row in passages for key in row)


def test_draft_keywords_cannot_displace_subject_organism_and_gene():
    original = source({
        "annotations": [{"text": "Enzyme function, activity and thermostability. " * 3, "entry": i} for i in range(20)],
        "genes": [{"geneName": {"value": "SAMPLE"}}],
        "organism": {"scientificName": "Saccharomyces cerevisiae", "taxonId": 559292},
    })
    before = copy.deepcopy(original)
    selected = selected_passages(original, "The mouse enzyme has this function, activity and thermostability.", 1600)
    pointers = {row["location"].get("json_pointer") for row in selected["passages"]}
    assert {"/genes", "/organism"} <= pointers
    assert any(pointer.startswith("/annotations/") for pointer in pointers)
    assert_anchors(original, selected, 1600)
    assert original == before


def test_service_hits_do_not_displace_the_input_structure_and_evidence_scope():
    original = source({"service": "structure_search", "data": {
        "hits": [{"description": "Structural alignment database match. " * 4, "score": i} for i in range(20)],
        "query_structure": {"origin": "experimental", "pdb_id": "1ABC", "chain": "B"},
        "interpretation": "Structure similarity, not a new predicted structure.",
    }})
    selected = selected_passages(original, "Structural alignment database match from a new sequence prediction", 1600)
    pointers = {row["location"].get("json_pointer") for row in selected["passages"]}
    assert {"/data/query_structure", "/data/interpretation"} <= pointers
    assert any(pointer.startswith("/data/hits/") for pointer in pointers)
    assert_anchors(original, selected, 1600)


def test_compact_prediction_keeps_its_original_single_record():
    original = source({"task": "stability", "values": [{"score": 0.4}], "units": None, "calibration": None})
    selected = selected_passages(original, "The melting temperature is 0.4 degrees", 600)
    assert len(selected["passages"]) == 1
    assert selected["passages"][0]["location"]["json_pointer"] == ""
    assert_anchors(original, selected, 600)


@pytest.mark.parametrize("budget", [80, 160, 400, 1600, 3200])
def test_context_shares_the_original_budget_and_selection_is_deterministic(budget):
    original = source({
        "organism": {"scientificName": "Example species", "lineage": ["group"] * 250},
        "annotations": [{"description": "Specific catalytic activity. " * 12, "entry": i} for i in range(30)],
        "data": {"query_structure": {"pdb_id": "1ABC", "chain": "A"}},
    })
    selected = selected_passages(original, "catalytic activity", budget)
    assert selected == selected_passages(original, "catalytic activity", budget)
    assert_anchors(original, selected, budget)


def test_context_still_leaves_room_for_a_relevant_long_passage():
    original = source({"organism": {"scientificName": "Example species"},
                       "description": "Catalytic activity was measured in this assay. " * 100})
    selected = selected_passages(original, "catalytic activity assay", 600)
    assert any(row["location"].get("json_pointer") == "/organism" for row in selected["passages"])
    assert any("Catalytic activity" in row["text"] for row in selected["passages"])
    assert_anchors(original, selected, 600)


def test_large_result_lists_cannot_exhaust_candidates_before_source_identity():
    original = source({"annotations": [{"entry": i, "text": "Enzyme activity."} for i in range(2200)],
                       "organism": {"scientificName": "Example species"}})
    selected = selected_passages(original, "enzyme activity", 600)
    assert any(row["location"].get("json_pointer") == "/organism" for row in selected["passages"])
    assert_anchors(original, selected, 600)


@pytest.mark.parametrize("key,value", [("unmodeled_residues", 5), ("matches_reference_sequence", False),
                                      ("units", None), ("assay/value~qualifier", "unknown")])
def test_scalar_fragments_keep_the_field_name_and_resolve_to_saved_json(key, value):
    original = source({"measurements": {"large_annotation": "background " * 1000, key: value}})
    selected = selected_passages(original, key.replace("_", " "), 600)
    escaped = key.replace("~", "~0").replace("/", "~1")
    row = next(row for row in selected["passages"] if row["location"].get("json_pointer") == "/measurements/" + escaped)
    assert json.loads(row["text"]) == {key: value}
    assert row["location"]["include_key"] is True
    assert_anchors(original, selected, 600)


def test_legacy_scalar_anchor_is_still_resolvable_and_new_projection_is_typed():
    original = source({"count": 5})
    old = {"json_pointer": "/count", "offset": 0, "length": 1}
    assert resolve_passage(original, old) == "5"
    with pytest.raises(ValueError, match="field anchor"):
        resolve_passage(original, {**old, "include_key": "count"})
