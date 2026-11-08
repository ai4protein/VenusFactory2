import copy
import json
import pytest
from backend.web_v2.research_chapter_context import (
    chapter_bundle, chapter_predictions, select_chapter_evidence,
)
from backend.web_v2 import research_longform as lf
from backend.web_v2 import research_report as research


def source(identity, kind, title=None):
    return {"id": identity, "kind": kind, "title": title or identity,
            "text": "Measured source evidence.", "evidence_level": "test",
            "url": "https://example.org/" + identity}


def test_literature_survives_dense_predictions_and_database_context():
    groups = [[source(prefix + str(i), kind) for i in range(count)]
              for prefix, kind, count in [
                  ("D", "database", 6), ("M", "user_material", 3),
                  ("P", "prediction", 16), ("R", "public_service", 5),
                  ("L", "literature", 6), ("W", "web", 6)]]
    core, materials, predictions, remote, papers, web = groups
    chosen = select_chapter_evidence(core, materials, predictions, remote, papers + web, papers)
    assert len(chosen) == 14 and len({s["id"] for s in chosen}) == 14
    assert {"L0", "L1", "W0", "W1", "M0", "M1"} <= {s["id"] for s in chosen}


def test_target_resources_precede_extra_shared_structures():
    shared = [source("S" + str(i), "database") for i in range(4)]
    targets = [source("T" + str(i), "database") | {
        "query_tool": "query_opentargets", "query_arguments": {"resource": name}}
        for i, name in enumerate(["genetic_constraint", "diseases"])]
    chosen = lf.chapter_database_sources(shared + targets, "variants_modifications")
    assert [s["id"] for s in chosen[:4]] == ["S0", "S1", "T0", "T1"]


def test_chapters_select_relevant_predictions_and_preserve_inputs():
    values = [source("P1", "prediction", "Solubility"),
              source("P2", "prediction", "Binding Site"),
              source("P3", "prediction", "Stability")]
    assert chapter_predictions(values, "structure_binding")[0]["id"] == "P2"
    assert chapter_predictions(values, "biophysics_production")[0]["id"] == "P1"
    assert [s["id"] for s in values] == ["P1", "P2", "P3"]


def test_bundle_keeps_citable_evidence_without_duplicating_raw_arrays():
    selected = [source("S1", "calculation"), source("S2", "prediction")]
    state = {"protein": {"name": "protein"}, "sections": [{"values": list(range(20000))}],
             "sources": selected, "remote_services": []}
    before = copy.deepcopy(state)
    bundle = chapter_bundle(state, {"focus": "binding"}, {"title": "Structure"},
                            "binding remit", selected)
    assert len(json.dumps(bundle)) < 2000
    assert bundle["evidence_scope"]["selected_source_ids"] == ["S1", "S2"]
    assert "calculations_and_predictions" not in bundle
    assert state == before


@pytest.mark.asyncio
async def test_all_specialists_get_bounded_citable_context_and_dossier_stays_complete(monkeypatch):
    raw = {"positions": list(range(15000))}
    sources = [source("S1", "calculation")]
    sources += [source("S" + str(i), "prediction", "Stability") for i in range(2, 18)]
    sources += [source("S18", "literature"), source("S19", "literature")]
    calls = []
    async def collect(payload, update, **kwargs):
        value = {"protein": {"name": "Public fixture"}, "sources": sources,
                 "sections": [{"id": "prediction_array", "data": raw}],
                 "steps": [], "predictions": [], "partial": False}
        update(value)
        return value
    async def complete(model, messages, tokens):
        bundle = json.loads(messages[1]["content"])
        if "chapter" in bundle:
            calls.append(bundle)
        return ("Observed protein measurements support this limited conclusion [S1].\n\n" * 70, False)
    monkeypatch.setattr(research, "run_research", collect)
    monkeypatch.setattr(research, "complete", complete)
    result = await lf.run_long_report(
        {"model_id": "fixture", "analyses": ["properties"], "lang": "en"},
        lambda value: None)
    assert len(calls) == 8
    assert all("S18" in {s["id"] for s in c["sources"]} for c in calls)
    assert max(len(json.dumps(c)) for c in calls) < 10000
    assert result["sections"][0]["data"] == raw
    assert len(result["sources"]) == len(sources)


def test_writer_context_has_a_total_bound_and_retains_each_selected_source():
    selected = [source(f"S{i}", "database") for i in range(1, 15)]
    for row in selected:
        row["text"] = "Measured conditions with assay-specific units and interpretation. " * 1000
        row["query_arguments"] = {"sequence": "M" * 10000}
    before = copy.deepcopy(selected)
    result = lf.source_context(selected, query="measured assay conditions")
    assert len(json.dumps(result, ensure_ascii=False)) <= 24000
    assert [s["id"] for s in result] == [s["id"] for s in selected]
    assert all(s["text"] and s["excerpted"] for s in result)
    assert all("query_arguments" not in s for s in result)
    assert selected == before


def test_chapter_remit_selects_late_processing_annotation():
    row = source("S1", "database")
    row["text"] = json.dumps({
        "comments": [{"description": "Human beta hemoglobin function and clinical observations " * 90} for _ in range(24)],
        "features": [
            {"type": "Initiator methionine", "description": "Removed", "location": {"start": 1, "end": 1}},
            {"type": "Chain", "description": "Beta-globin", "location": {"start": 2, "end": 147}},
        ],
    })
    state = {"sources": [row], "remote_services": []}
    bundle = chapter_bundle(state, {"focus": "Review function and maturation of human hemoglobin beta"}, {"title": "Variants and modifications"},
                            "Cleavage, maturation and residue numbering", [row])
    text = bundle["sources"][0]["text"]
    assert "Initiator methionine" in text and '"type":"Chain"' in text
    assert "Removed" in text and "147" in text
    assert "Signal" not in text
    assert bundle["evidence_scope"]["selected_source_ids"] == ["S1"]


def test_material_metadata_and_verbatim_excerpt_remain_available():
    material = source("S2", "user_material")
    material["text"] = "Page 7\nThe measured activity was 14.5 nmol/min at 25 C."
    material["extraction"] = {"format": "pdf", "units": 8, "truncated": True, "characters": 10000}
    output = lf.source_context([material], query="measured activity")
    assert output[0]["text"] == material["text"]
    assert output[0]["extraction"] == material["extraction"]


def test_shared_context_budget_also_applies_to_legacy_callers():
    selected = [source(f"S{i}", "user_material") for i in range(1, 15)]
    for row in selected:
        row["text"] = "Verbatim page with observations. " * 1000
    result = lf.source_context(selected)
    assert len(json.dumps(result, ensure_ascii=False)) <= 24000
    assert len(result) == 14 and all(row["excerpted"] for row in result)


def test_query_scope_keeps_database_chain_and_conditions_without_raw_sequence():
    row = source("S1", "public_service")
    row["query_arguments"] = {
        "database": "pdb100", "chain": "B", "evalue": 0.001, "sequence": "M" * 5000,
    }
    result = lf.source_context([row], query="structural homology")[0]
    assert result["query_arguments"] == {"database": "pdb100", "chain": "B", "evalue": 0.001}
    assert result["query_arguments_excerpted"] is True
    assert row["query_arguments"]["sequence"] == "M" * 5000


def test_existing_one_argument_context_builder_remains_callable():
    selected = [source("S1", "database")]
    called = []
    def legacy_context(values):
        called.append(values)
        return [{"id": "S1", "text": "Existing caller context."}]
    bundle = chapter_bundle({"remote_services": []}, {}, {"title": "Identity"},
                            "Sequence identity", selected, legacy_context)
    assert called == [selected]
    assert bundle["sources"] == [{"id": "S1", "text": "Existing caller context."}]
