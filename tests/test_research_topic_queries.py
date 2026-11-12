"""Specialist searches retain controlled target and topic constraints."""
import pytest

from backend.web_v2 import research_report as research
from backend.web_v2.research_longform import CHAPTERS


def reference():
    return {"primaryAccession": "P00918", "proteinDescription": {
        "recommendedName": {"fullName": {"value": "Carbonic anhydrase 2"}}},
        "genes": [{"geneName": {"value": "CA2"}}],
        "organism": {"scientificName": "Homo sapiens"},
        "references": [{"citation": {"citationCrossReferences": [{"database": "PubMed", "id": "123456"}]}}]}


@pytest.mark.parametrize("has_curated", [False, True])
def test_all_eight_specialists_have_distinct_anchored_topics(has_curated):
    entry = reference()
    if not has_curated:
        entry["references"] = []
    queries = []
    for _, _, _, _, terms in CHAPTERS:
        query = research.literature_topic_query(entry, "protein", terms)
        queries.append(query)
        assert 'TITLE_ABS:"Carbonic anhydrase 2"' in query
        assert 'TITLE_ABS:"CA2"' not in query and '"Homo sapiens"' in query
        assert all(word in query.lower() for word in terms.split())
    assert len(set(queries)) == len(CHAPTERS)


def test_unmapped_target_keeps_its_identity_and_the_topic_without_guessing_species():
    query = research.literature_topic_query(None, "lab enzyme", "sequence evolution ortholog isoform")
    assert '"lab enzyme"' in query and "evolution" in query and "ortholog" in query
    assert "human" not in query and "Homo sapiens" not in query


def test_untrusted_planner_text_cannot_supply_query_operators_or_unknown_keywords():
    query = research.literature_topic_query(reference(), "protein", 'binding OR * PMID:123 random ALL:everything')
    assert "binding" in query and "PMID" not in query and "everything" not in query and "*" not in query
    assert research.literature_queries(reference(), "protein", ["binding"])[0] == "SRC:MED AND (EXT_ID:123456)"


def test_resolved_synonyms_disambiguate_short_gene_symbols():
    entry = reference()
    entry["proteinDescription"]["alternativeNames"] = [
        {"fullName": {"value": "Carbonic anhydrase II"}, "shortNames": [{"value": "CA-II"}]},
        {"fullName": {"value": "Carbonate dehydratase II"}},
        {"fullName": {"value": "Carbonic anhydrase 2"}},
    ]
    query = research.literature_topic_query(entry, "CA2", "function catalysis")
    assert 'TITLE_ABS:"Carbonic anhydrase II"' in query
    assert 'TITLE_ABS:"Carbonate dehydratase II"' in query
    assert query.count('TITLE_ABS:"Carbonic anhydrase 2"') == 1
    assert 'TITLE_ABS:"CA2"' not in query and 'TITLE_ABS:"CA-II"' not in query


def test_single_word_full_name_is_retained_but_generic_name_uses_gene_fallback():
    entry = reference()
    entry["proteinDescription"] = {"recommendedName": {"fullName": {"value": "Insulin"}}}
    entry["genes"] = [{"geneName": {"value": "INS"}}]
    query = research.literature_topic_query(entry, "P01308", "structure")
    assert 'TITLE_ABS:"Insulin"' in query and 'TITLE_ABS:"INS"' not in query
    entry["proteinDescription"]["recommendedName"]["fullName"]["value"] = "Uncharacterized protein"
    query = research.literature_topic_query(entry, "P01308", "structure")
    assert 'TITLE_ABS:"INS"' in query and "Uncharacterized protein" not in query
