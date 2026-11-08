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
                            "binding remit", selected, lf.source_context)
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
