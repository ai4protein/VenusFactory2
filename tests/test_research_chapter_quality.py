from research_reviewer_fixture import reviewer_catalog as _reviewer_catalog  # noqa: F401
from research_source_review_fixture import with_source_review
"""Reject generated sequence hallucinations without discarding their audit trail."""
import hashlib
import json
import zipfile
from unittest.mock import patch

import pytest

from backend.web_v2 import research_longform as lf
from backend.web_v2.research_dossier import build_dossier


def source():
    return {"id":"S1","title":"Reference","url":"https://example.org/reference",
            "kind":"database","evidence_level":"annotation","text":"Verified metadata."}


def prose():
    return "Measured evidence supports the limited stated observation. " * 70 + "[S1]"


@pytest.mark.parametrize("sequence",["ACDEFGHIKLMNPQRSTVWY"*4,"Val-"*80+"Val"])
def test_full_generated_sequences_fail_quality_check_even_with_sufficient_cited_prose(sequence):
    text=prose()+"\n"+sequence
    audit=lf.audit_chapter(text,["S1"],"en")
    assert audit["substantial"] and audit["has_citations"]
    assert audit["long_sequence"]
    assert not lf.contains_long_sequence("Gly-X-Gly motif and the A23V variant.")


def test_partial_chapter_not_in_report_but_original_draft_preserved_in_zip():
    draft=prose()+"\n"+"Val-"*80+"Val"
    chapter={"id":"identity_evolution","title":"Identity","status":"partial",
             "draft":draft,"source_ids":["S1"],"audit":lf.audit_chapter(draft,["S1"],"en")}
    report={"id":"x"*32,"workflow":"specialist_research_v2","status":"completed","partial":True,
            "protein":{"name":"Example"},"sources":[source()],"chapters":[chapter],
            "report":"STALE "+draft,"predictions":[],"remote_services":[]}
    display=lf.assemble(report,"en")
    assert "Val-Val-Val" not in display
    assert "awaiting revision" in display
    with build_dossier(report) as handle,zipfile.ZipFile(handle) as bundle:
        assert "Val-Val-Val" not in bundle.read("report.md").decode()
        assert "Val-Val-Val" not in bundle.read("chapters/01.md").decode()
        assert bundle.read("drafts/01.md").decode()==draft
        index=json.loads(bundle.read("chapters/index.json"))
        assert index[0]["unaccepted_draft"]=="drafts/01.md"
        for row in json.loads(bundle.read("manifest.json"))["files"]:
            assert hashlib.sha256(bundle.read(row["path"])).hexdigest()==row["sha256"]
    assert chapter["draft"]==draft and chapter["status"]=="partial"


def test_explicit_parenthetical_sources_normalized_without_inventing_evidence():
    text="Known (S1), unknown (S2), grouped (S1, S2)."
    normalized=lf.normalize_source_citations(text,[source()])
    assert normalized=="Known [S1], unknown [S2], grouped [S1] [S2]."
    assert lf.audit_chapter(normalized,["S1"],"en")["unknown_citations"]==["S2"]


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_editor",[False,True])
async def test_failed_specialist_excluded_from_editor_and_readable_report(bad_editor):
    async def collect(payload,update,**kwargs):
        state={"protein":{"name":"Example"},"sources":[source()],"steps":[],"sections":[],
               "partial":False,"report":""}
        update(state);return state
    @with_source_review
    async def complete(model,messages,tokens, **_review_metadata):
        system=messages[0]["content"]
        if system.startswith("Act as the scientific editor"):
            data=json.loads(messages[1]["content"])
            identity=data["chapters"][0]
            assert identity["status"]=="partial" and identity["text"]==""
            return (prose()+(" [S999]" if bad_editor else ""),False)
        if lf.CHAPTERS[0][3] in system:
            return (prose()+"\n"+"Val-"*80+"Val",False)
        return prose(),False
    payload={"model_id":"fixture","analyses":["properties"],"lang":"en"}
    with patch.object(lf.research,"run_research",side_effect=collect),patch.object(lf.research,"complete",side_effect=complete):
        result=await lf.run_long_report(payload,lambda x:None)
    assert result["completed_chapters"]==7 and result["partial"]
    assert result["chapters"][0]["audit"]["long_sequence"]
    assert "Val-Val-Val" not in result["report"]
    assert result["review_status"]==("partial" if bad_editor else "completed")
    if bad_editor:
        assert "S999" not in result["report"]
        assert result["review_audit"]["unknown_citations"]==["S999"]
