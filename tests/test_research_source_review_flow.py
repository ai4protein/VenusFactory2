import asyncio
import copy
import hashlib
import json
import zipfile
from unittest.mock import patch
import pytest

from backend.web_v2 import research_longform as lf, research_source_review as sr
from backend.web_v2.research_dossier import build_dossier

PROSE = "The reference annotates removal of the initiator methionine at the beginning of the sequence. " * 30 + "[S1]"
BAD = "A signal peptide is removed. [S1]\n\n"
SOURCE = {"id":"S1","title":"Reference","kind":"database","url":"https://example.org/evidence",
          "evidence_level":"annotation","text":json.dumps({"type":"Initiator methionine","description":"Removed"})}
PAYLOAD = {"model_id":"fixture","analyses":["properties"],"lang":"en"}


def review_response(request):
    rows=[]
    for unit in request["units"]:
        bad="A signal peptide is removed." in unit["text"]
        source=next(s for s in request["sources"] if s["id"] in unit["citations"])
        passage=source["passages"][0]
        rows.append({"unit_id":unit["id"],"classification":"factual","verdict":"unsupported" if bad else "supported",
                     "issues":[{"claim":"A signal peptide is removed.","reason":"Initiator methionine is a different annotation type."}] if bad else [],
                     "evidence":[{"source_id":source["id"],"passage_id":passage["id"],"quote":passage["text"]}]})
    return json.dumps({"assessments":rows})


class FixtureModel:
    def __init__(self, bad_chapter=False, bad_editor=False, unavailable=False):
        self.calls=[];self.bad_chapter=bad_chapter;self.bad_editor=bad_editor;self.unavailable=unavailable
        self.writers=0;self.editors=0
    async def __call__(self, model, messages, tokens):
        self.calls.append(messages);system=messages[0]["content"]
        if system==sr.SYSTEM:
            if self.unavailable:raise RuntimeError("fixture source-review interruption")
            return review_response(json.loads(messages[1]["content"])),False
        if "scientific editor" in system:
            self.editors+=1
            return (BAD if self.bad_editor and self.editors==1 else "")+PROSE,False
        if "Revise only this protein research chapter" in system:
            assert "Independent" not in system or len(messages)==2
            return PROSE,False
        self.writers+=1
        return (BAD if self.bad_chapter and self.writers==1 else "")+PROSE,False


async def run(model, checkpoint=None, update=None):
    async def collect(*args, **kwargs):
        return {"sources":[copy.deepcopy(SOURCE)],"steps":[],"sections":[],"protein":{"name":"Fixture protein"}}
    with patch.object(lf.research,"run_research",side_effect=collect),patch.object(lf.research,"complete",side_effect=model.__call__):
        return await lf.run_long_report(PAYLOAD,update or (lambda value:None),checkpoint=checkpoint)


@pytest.mark.asyncio
async def test_complete_report_resume_and_dossier():
    model=FixtureModel();state=await run(model)
    assert state["phase"]=="completed" and not state["partial"]
    assert state["completed_chapters"]==8 and state["review_status"]=="completed"
    assert len(model.calls)==11
    assert len(state["source_review"]["documents"])==9
    second=await run(model,state)
    assert len(model.calls)==11 and second["model_calls"]==11
    with build_dossier({**state,"id":"fixture","status":"completed"}) as handle,zipfile.ZipFile(handle) as bundle:
        assert len(json.loads(bundle.read("verification/claims.json")))==9
        assert json.loads(bundle.read("verification/source-review.json"))["status"]=="completed"
        for entry in json.loads(bundle.read("manifest.json"))["files"]:
            assert hashlib.sha256(bundle.read(entry["path"])).hexdigest()==entry["sha256"]

@pytest.mark.asyncio
@pytest.mark.parametrize("where",["chapter","editor"])
async def test_source_failure_triggers_author_revision_and_independent_recheck(where):
    model=FixtureModel(bad_chapter=where=="chapter",bad_editor=where=="editor")
    snapshots = []
    state=await run(model,update=lambda value: snapshots.append(copy.deepcopy(value)))
    if where == "chapter":
        assert any(c["status"] == "revising" for snapshot in snapshots for c in snapshot["chapters"])
    assert state["phase"]=="completed" and len(model.calls)==(14 if where=="chapter" else 13)
    assert "A signal peptide is removed." not in state["report"]
    assert any("unsupported" in a["response"] for a in state["source_review"]["attempts"])
    with build_dossier({**state,"id":"fixture","status":"completed"}) as handle,zipfile.ZipFile(handle) as bundle:
        drafts=[name for name in bundle.namelist() if name.startswith("drafts/") and name.endswith(".md")]
        assert any("A signal peptide is removed." in bundle.read(name).decode() for name in drafts)


@pytest.mark.asyncio
async def test_review_budget_cannot_be_reset_by_resume():
    model=FixtureModel()
    with patch.object(lf,"MAX_MODEL_CALLS",9):
        state=await run(model)
        assert state["phase"]=="incomplete" and state["partial"] and len(model.calls)==9
        second=await run(model,state)
    assert second["model_calls"]==9 and len(model.calls)==9
    assert second["review_status"]!="completed"


@pytest.mark.asyncio
async def test_interrupted_review_resumes_without_rewriting_specialists():
    model=FixtureModel(unavailable=True)
    first=await run(model)
    assert first["phase"]=="incomplete" and first["completed_chapters"]==0
    assert model.writers==8 and len(model.calls)==9
    model.unavailable=False
    second=await run(model,first)
    assert second["phase"]=="completed" and model.writers==8 and len(model.calls)==12
    assert len(second["source_review"]["attempts"])==3


@pytest.mark.asyncio
async def test_changed_source_requires_new_review_and_editor_synthesis():
    model=FixtureModel()
    first=await run(model)
    changed=copy.deepcopy(first);changed["sources"][0]["text"]=json.dumps({"type":"Initiator methionine","description":"Removed after processing"})
    snapshots = []
    second=await run(model,changed,update=lambda value: snapshots.append(copy.deepcopy(value)))
    assert snapshots[0]["completed_chapters"] == 0 and not snapshots[0]["minimum_met"]
    assert second["phase"]=="completed" and len(model.calls)==14 and model.writers==8
    assert first["chapters"][0]["source_review"]["input_sha256"]!=second["chapters"][0]["source_review"]["input_sha256"]

@pytest.mark.asyncio
async def test_changed_chapter_invalidates_synthesis_even_if_sources_unchanged():
    from backend.web_v2.research_editor import review_ready
    model=FixtureModel();state=await run(model)
    assert review_ready(state,"en")
    state["chapters"][0]["draft"]+=" New content [S1]."
    assert not review_ready(state,"en")
    state["model_calls"]=state["model_call_limit"]
    second=await run(model,state)
    assert second["partial"] and second["review_status"]!="completed"
    assert "Integrated assessment" not in second["report"]


@pytest.mark.asyncio
async def test_completed_chapters_are_checkpointed_before_later_review_is_cancelled(monkeypatch):
    monkeypatch.setattr(sr, "MAX_UNITS", 5)
    waiting = asyncio.Event()
    snapshots = []

    class PausingModel(FixtureModel):
        def __init__(self):
            super().__init__()
            self.review_batches = 0

        async def __call__(self, model, messages, tokens):
            if messages[0]["content"] == sr.SYSTEM:
                self.review_batches += 1
                if self.review_batches == 2:
                    self.calls.append(messages)
                    waiting.set()
                    await asyncio.Future()
            return await super().__call__(model, messages, tokens)

    model = PausingModel()
    task = asyncio.create_task(run(model, update=lambda value: snapshots.append(copy.deepcopy(value))))
    await asyncio.wait_for(waiting.wait(), timeout=2)
    during = snapshots[-1]
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    saved = snapshots[-1]
    for checkpoint in (during, saved):
        assert checkpoint["completed_chapters"] == 5
        assert checkpoint["minimum_met"] is True
        assert checkpoint["phase"] == "reviewing"
        assert sum(c["status"] == "completed" for c in checkpoint["chapters"]) == 5
        assert len(checkpoint["source_review"]["documents"]) == 8
        assert sum(r["status"] == "completed" for r in checkpoint["source_review"]["documents"].values()) == 5
        assert checkpoint.get("review_status") != "completed"
    assert saved["source_review"]["status"] == "interrupted"
    assert all(c["status"] in {"completed", "partial"} for c in saved["chapters"])
    assert saved["model_calls"] == 10 and model.writers == 8
    with build_dossier({**saved, "id": "interrupted-fixture", "status": "interrupted"}) as handle, zipfile.ZipFile(handle) as bundle:
        exported = json.loads(bundle.read("verification/source-review.json"))
        assert exported["status"] == "interrupted"
        assert sum(row["status"] == "completed" for row in exported["documents"].values()) == 5
    resumed = await run(model, saved)
    assert resumed["phase"] == "completed" and resumed["completed_chapters"] == 8
    assert resumed["model_calls"] == 13 and model.writers == 8
    assert model.review_batches == 4
    assert saved["source_review"]["attempts"][-1]["status"] == "interrupted"
