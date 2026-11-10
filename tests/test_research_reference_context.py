"""A PDB construct and its reference annotations retain distinct roles."""
import copy
import json
import zipfile

import pytest

from backend.web_v2 import research_longform as longform
from backend.web_v2 import research_report as research
from backend.web_v2.research_dossier import build_dossier
from mcp_tools.tools.database.rcsb import research_target


def reference():
    return {
        "primaryAccession": "P00918", "entryType": "UniProtKB reviewed",
        "proteinDescription": {"recommendedName": {"fullName": {"value": "Carbonic anhydrase 2"}}},
        "genes": [{"geneName": {"value": "CA2"}}],
        "organism": {"scientificName": "Homo sapiens", "taxonId": 9606},
        "sequence": {"value": "ACDEFGHIKLMNPQRSTVWY", "length": 20},
        "uniProtKBCrossReferences": [{"database": "PDB", "id": "12CA"}],
        "references": [{"citation": {"citationCrossReferences": [
            {"database": "PubMed", "id": "123456"}]}}],
    }


def target():
    return {
        "protein": {"pdb_id": "12CA", "accession": "P00918", "name": "CA2 construct",
                    "organism": "Homo sapiens", "length": 12, "structure_chain": "A",
                    "url": "https://www.rcsb.org/structure/12CA"},
        "sequence": "A" * 12, "structure_pdb": "fixture coordinates", "structure_chain": "A",
        "source_chain": "A", "interpretation": "Observed construct, not full reference",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("origin", ["pdb", "uniprot"])
@pytest.mark.parametrize("resume", [False, True])
async def test_specialists_use_reference_identity_without_replacing_construct(monkeypatch, tmp_path, origin, resume):
    queries, bundles, fetches = [], [], []

    async def fetch(url, *args, **kwargs):
        fetches.append(url)
        if "rest.uniprot.org" in url:
            return reference()
        assert url == "https://data.rcsb.org/rest/v1/core/entry/12CA"
        return {"struct": {"title": "Fixture CA2 structure"}, "exptl": [{"method": "X-RAY DIFFRACTION"}]}

    async def load(*args):
        return target()

    async def literature(query):
        queries.append(query)
        return [{"title": "Fixture reference study", "url": "https://europepmc.org/article/MED/123456",
                 "kind": "literature", "evidence_level": "abstract", "text": "Fixture reference annotation."}]

    async def complete(model, messages, tokens):
        bundle = json.loads(messages[1]["content"])
        if "chapter" in bundle:
            bundles.append(bundle)
            # This tests collection and handoff, not scientific review or LLM quality.
            return "Short fixture chapter [S1]", False
        return json.dumps({"queries": ["Carbonic anhydrase 2 Homo sapiens structure"]}), False

    monkeypatch.setattr(research, "get_json", fetch)
    monkeypatch.setattr(research_target, "load_pdb_target", load)
    monkeypatch.setattr(research, "literature", literature)
    monkeypatch.setattr(research, "complete", complete)
    payload = {"model_id": "fixture", "analyses": ["literature", "structure", "properties"], "lang": "en"}
    payload.update({"pdb_id": "12CA", "structure_chain": "A"} if origin == "pdb" else {"accession": "P00918"})
    saved = None
    if resume:
        saved = await research.run_research(payload, lambda _: None, synthesize=False)
        saved.update(workflow="specialist_research_v2", phase="chapters", model_calls=1,
                     chapters=[{"id": key, "title": title, "status": "pending", "draft": "",
                                "source_ids": [], "attempts": 0} for key, title, *_ in longform.CHAPTERS])
        # Persisted PDB reference sources predate primaryAccession and reference IDs.
        if origin == "pdb":
            source = next(row for row in saved["sources"] if "uniprot.org" in row["url"])
            data = json.loads(source["text"])
            data["entry"].pop("primaryAccession", None)
            data["entry"].pop("references", None)
            source["text"] = json.dumps(data)
    before = copy.deepcopy(saved)
    result = await longform.run_long_report(payload, lambda _: None, checkpoint=saved)
    assert saved == before
    assert len({row["chapter"] for row in bundles}) == 8
    assert all('TITLE_ABS:"CA2"' in chapter["literature_query"]
               and '"Homo sapiens"' in chapter["literature_query"] for chapter in result["chapters"])
    assert len([url for url in fetches if "rest.uniprot.org" in url]) == 1
    assert "https://data.rcsb.org/rest/v1/core/entry/12CA" in fetches
    assert next(row for row in result["steps"] if row["id"] == "structure_metadata")["status"] == "completed"
    assert result["protein"]["length"] == (12 if origin == "pdb" else 20)
    assert result["sections"][0]["data"]["length_aa"] == (12 if origin == "pdb" else 20)
    # Draft fixtures must not become a purportedly accepted report.
    assert result["phase"] == "incomplete" and not result["minimum_met"]
    with build_dossier(result) as handle, zipfile.ZipFile(handle) as archive:
        refs = json.loads(archive.read("references.json"))
        uniprot = next(row for row in refs if "uniprot.org" in row["url"])
        exported = json.loads(archive.read(uniprot["file"]))
        original = next(row for row in result["sources"] if row["id"] == uniprot["id"])
        assert exported["text"] == original["text"]
        if origin == "pdb":
            assert json.loads(exported["text"])["annotation_scope"] == target()["interpretation"]


def test_compact_reference_keeps_bounded_curated_pubmed_queries():
    value = reference()
    value["references"] *= 2
    value["references"] += [{"citation": {"citationCrossReferences": [
        {"database": "PubMed", "id": str(100 + index)},
        {"database": "DOI", "id": "10.example/unused"}]}} for index in range(20)]
    value["references"].append({"citation": {"citationCrossReferences": [{"database": "PubMed", "id": "invalid OR query"}]}})
    compact = research.compact_entry(value)
    assert compact["primaryAccession"] == "P00918"
    queries = research.literature_queries(compact, "protein", ["structure"])
    assert "EXT_ID:123456" in queries[0] and queries[0].count("EXT_ID:") == 6
    assert len(compact["references"]) == 6 and "sequence" not in compact
    assert "invalid OR query" not in json.dumps(compact)


@pytest.mark.parametrize("bad", ["not-json", "[]", '{"proteinDescription": {}, "organism": "Homo sapiens"}',
                                '{"primaryAccession": "P69905", "proteinDescription": {}, "organism": {}}'])
def test_reference_selection_ignores_malformed_or_different_accession(bad):
    source = {"kind": "database", "url": "https://www.uniprot.org/uniprotkb/P00918", "text": bad}
    assert research.reference_entry([source], "P00918") == {}


def test_reference_selection_uses_database_identity_not_source_position():
    compact = research.compact_entry(reference())
    wanted = {"kind": "database", "url": "https://www.uniprot.org/uniprotkb/P00918",
              "text": json.dumps({"entry": compact, "annotation_scope": "Reference only"})}
    unrelated = {"kind": "database", "url": "https://www.rcsb.org/structure/12CA", "text": json.dumps(target()["protein"])}
    materials = {**wanted, "kind": "user_material"}
    wrong = {**wanted, "url": "https://www.uniprot.org/uniprotkb/P69905"}
    before = copy.deepcopy([unrelated, materials, wrong, wanted])
    assert research.reference_entry(before, "P00918")["genes"][0]["geneName"]["value"] == "CA2"
    assert research.reference_entry(before[:-1], "P00918") == {}
    assert research.reference_entry(before, "") == {}
    assert before == [unrelated, materials, wrong, wanted]


@pytest.mark.parametrize("genes", [None, []])
def test_reference_without_gene_name_keeps_protein_and_species_in_search(genes):
    saved = research.compact_entry(reference())
    saved["genes"] = genes
    queries = research.literature_queries(saved, "reference", ["structure"])
    assert any('TITLE_ABS:"Carbonic anhydrase 2"' in q and '"Homo sapiens"' in q for q in queries)
