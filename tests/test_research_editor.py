from research_reviewer_fixture import reviewer_catalog as _reviewer_catalog  # noqa: F401
from research_source_review_fixture import with_source_review
import copy
import hashlib
import json
import zipfile
from unittest.mock import patch

import pytest

from backend.web_v2 import research_longform as lf
from backend.web_v2.research_editor import editorial_input
from backend.web_v2.research_dossier import build_dossier


def prose(cited=True):
    return "Evidence supports this observation under the recorded assay conditions. " * 65 + ("[S1]" if cited else "")


def checkpoint():
    return {"workflow":"specialist_research_v2","phase":"completed","collection_complete":True,
            "collection_partial":False,"model_calls":8,"partial":False,"steps":[],"sections":[],
            "protein":{"name":"Example protein","accession":"P12345","organism":"Human"},
            "sources":[{"id":"S1","kind":"database","title":"Reference","url":"https://example.org/reference",
                        "text":"An experimentally measured value in a human specimen.","evidence_level":"measurement"}],
            "chapters":[{"id":key,"title":en,"status":"completed","source_ids":["S1"],"draft":prose()}
                        for key,en,_,_,_ in lf.CHAPTERS]}


async def run(state, complete):
    with patch.object(lf.research,"complete",side_effect=complete):
        return await lf.run_long_report({"model_id":"fixture","analyses":["properties"],"lang":"en"},
                                        lambda value:None,checkpoint=state)


@pytest.mark.asyncio
async def test_uncited_editor_revised_once_and_original_kept_in_dossier():
    calls=[]
    @with_source_review
    async def complete(model,messages,tokens, **_review_metadata):
        calls.append(messages)
        bundle=json.loads(messages[1]["content"])
        assert bundle["evidence_excerpts"][0]["text"].startswith("An experimentally")
        if len(calls)==1:
            return prose(False),False
        assert "Validation feedback" in messages[-1]["content"]
        assert "[S1]" in messages[-1]["content"]
        return prose(),False
    result=await run(checkpoint(),complete)
    assert len(calls)==2 and result["model_calls"]==13
    assert result["review_status"]=="completed" and not result["partial"]
    assert [a["status"] for a in result["review_attempts"]]==["partial","completed"]
    assert result["review_attempts"][0]["draft"]==prose(False)
    with build_dossier({**result,"id":"x"*32,"status":"completed"}) as handle,zipfile.ZipFile(handle) as bundle:
        assert bundle.read("drafts/editor/attempt-01.md").decode()==prose(False)
        assert json.loads(bundle.read("drafts/editor/index.json"))[0]["status"]=="partial"
        for row in json.loads(bundle.read("manifest.json"))["files"]:
            assert hashlib.sha256(bundle.read(row["path"])).hexdigest()==row["sha256"]


@pytest.mark.asyncio
async def test_resume_repairs_existing_editor_without_rewriting_specialists():
    state=checkpoint();state.update(model_calls=14,review=prose(False),review_status="partial")
    chapters=copy.deepcopy(state["chapters"]);calls=[]
    @with_source_review
    async def complete(model,messages,tokens, **_review_metadata):
        calls.append(messages)
        assert len(messages)==3 and prose(False) in messages[-1]["content"]
        return prose(),False
    result=await run(state,complete)
    assert result["model_calls"]==18 and len(calls)==1
    assert all(all(result["chapters"][i][k]==v for k,v in c.items()) for i,c in enumerate(chapters)) and result["review_status"]=="completed"
    assert result["review_attempts"][0]["origin"]=="checkpoint"
    second=await run(result,complete)
    assert len(calls)==1 and second["model_calls"]==18
    assert second["review_attempts"]==result["review_attempts"]


@pytest.mark.asyncio
async def test_repair_cannot_reset_global_model_budget():
    state=checkpoint();state["model_calls"]=0;calls=[]
    @with_source_review
    async def complete(*args):
        calls.append(args)
        return prose(False),False
    with patch.object(lf,"MAX_MODEL_CALLS",1):
        result=await run(state,complete)
    assert len(calls)==0 and result["model_calls"]==1
    assert result["review_status"]=="unavailable" and result["partial"]
    assert result.get("review_attempts",[])==[]


@pytest.mark.asyncio
@pytest.mark.parametrize("bad,truncated",[(prose()+" [S999]",False),(prose()+"\n"+"Val-"*80+"Val",False),(prose(),True)])
async def test_invalid_editor_never_inserted_and_bounded_attempts_retained(bad,truncated):
    calls=[]
    @with_source_review
    async def complete(*args):
        calls.append(args)
        return bad,truncated
    result=await run(checkpoint(),complete)
    assert len(calls)==2 and len(result["review_attempts"])==2
    assert result["review_status"]=="partial"
    assert "Integrated assessment" not in result["report"]
    assert result["review_attempts"][0]["draft"]==bad


@pytest.mark.asyncio
async def test_changed_evidence_invalidates_completed_editor_cache():
    @with_source_review
    async def complete(*args):
        return prose(),False
    first=await run(checkpoint(),complete)
    changed=copy.deepcopy(first)
    changed["sources"][0]["text"]="A corrected assay with a different measurement."
    calls=[]
    @with_source_review
    async def revise(model,messages,tokens, **_review_metadata):
        calls.append(messages)
        assert "corrected assay" in messages[1]["content"]
        return prose(),False
    second=await run(changed,revise)
    assert len(calls)==1 and second["review_input_sha256"]!=first["review_input_sha256"]


def test_editor_reads_bounded_excerpts_and_failed_chapter_has_no_prose():
    state=checkpoint();state["sources"][0]["text"]="Original evidence. "*1000
    state["chapters"][0].update(status="partial",draft="REJECTED CLAIM")
    bundle=editorial_input(state,{})
    assert bundle["chapters"][0]["text"]==""
    assert len(bundle["evidence_excerpts"][0]["text"])==8000
    assert bundle["evidence_excerpts"][0]["excerpted"] is True
