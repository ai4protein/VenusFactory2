from research_reviewer_fixture import reviewer_catalog as _reviewer_catalog  # noqa: F401
from research_source_review_fixture import with_source_review
"""Citation-format compatibility must not invent sources or validate code samples."""
import pytest
from unittest.mock import patch
from backend.web_v2 import research_longform as lf

SOURCES = [{"id":"S1", "title":"UniProt · P69905 · Hemoglobin subunit alpha"},
           {"id":"S2", "title":"Measured sequence properties"}]


def test_known_named_references_and_groups_become_canonical():
    text = ("Identity (UniProt · P69905 · Hemoglobin subunit alpha, S1). "
            "Measured properties [S1, S2].")
    assert lf.normalize_source_citations(text, SOURCES) == "Identity [S1]. Measured properties [S1] [S2]."


def test_normalization_never_fills_uncited_prose_or_changes_source_attribution():
    text = "Uncited claim. (An unrelated source, S1). (Measured sequence properties, S9)."
    assert lf.normalize_source_citations(text, SOURCES) == text
    unknown = lf.normalize_source_citations("Claim [S1, S99].", SOURCES)
    audit = lf.audit_chapter(unknown, ["S1","S2"], "en")
    assert audit["unknown_citations"] == ["S99"]


def test_source_marker_examples_in_code_do_not_count_as_evidence():
    text = "prose evidence " * 230 + "```text\n[S1]\n```\n`[S2]`"
    normalized = lf.normalize_source_citations(text, SOURCES)
    assert normalized == text
    assert lf.audit_chapter(text, ["S1","S2"], "en")["has_citations"] is False


def test_code_images_and_link_text_are_not_rewritten():
    text = "`(A study, S1) [S1, S2]` ![S1, S2](plot.png) [S1, S2](https://example.org)"
    assert lf.normalize_source_citations(text, SOURCES) == text


@pytest.mark.asyncio
async def test_actual_writer_pipeline_normalizes_only_declared_sources():
    calls = []
    async def collect(payload, update, **kwargs):
        state = {"protein":{"name":"Fixture"}, "sources":[{"id":"S1","title":"A study",
                 "url":"https://example.org", "kind":"literature", "evidence_level":"abstract",
                 "text":"Supplied observations."}], "steps":[], "sections":[], "partial":False, "report":""}
        update(state)
        return state
    @with_source_review
    async def complete(model, messages, tokens, **_review_metadata):
        calls.append(messages)
        return ("Observed protein evidence " * 160 + "(A study, S1).", False)
    with patch.object(lf.research, "run_research", side_effect=collect), patch.object(lf.research, "complete", side_effect=complete):
        result = await lf.run_long_report({"model_id":"fixture", "analyses":["properties"], "lang":"en"}, lambda state:None)
    assert result["completed_chapters"] == 8 and result["minimum_met"]
    assert len(calls) == 9
    assert all("[S1]" in ch["draft"] and "(A study, S1)" not in ch["draft"] for ch in result["chapters"])
