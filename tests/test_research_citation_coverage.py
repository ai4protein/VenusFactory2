import hashlib
import json
import zipfile

from backend.web_v2 import research_longform as lf
from backend.web_v2.research_editor import citation_coverage, editor_audit, review_ready
from backend.web_v2.research_dossier import build_dossier


def paragraph(cited):
    return "The measured observation depends on the documented assay conditions. "*45 + (" [S1]" if cited else "")


def test_a_single_citation_does_not_cover_other_uncited_sections():
    body="\n\n".join([paragraph(False),paragraph(False),paragraph(True)])
    audit=editor_audit(body,["S1"],"en")
    assert audit["has_citations"] and audit["substantial"]
    assert not audit["citation_coverage_ok"]
    assert audit["uncited_block_indices"]==[1,2]


def test_headings_and_code_do_not_count_as_cited_evidence():
    tick=chr(96)
    body="# Title\n\n"+paragraph(False)+"\n\n"+tick*3+"\n[S1]\n"+tick*3
    audit=citation_coverage(body)
    assert audit["citation_blocks"]==1 and audit["cited_blocks"]==0


def test_each_cited_substantial_section_passes_coverage():
    body="## Findings\n\n"+paragraph(True)+"\n\n## Interpretation\n\n"+paragraph(True)
    audit=editor_audit(body,["S1"],"en")
    assert audit["citation_coverage_ok"] and audit["citation_coverage"]==1


def test_old_completed_status_does_not_override_current_editor_check():
    draft="UNCITED_EDITOR_CLAIM\n\n"+paragraph(False)+"\n\n"+paragraph(False)+"\n\n"+paragraph(True)
    source={"id":"S1","title":"Reference","url":"https://example.org/reference","evidence_level":"annotation"}
    state={"workflow":"specialist_research_v2","protein":{"name":"Example"},"sources":[source],
           "chapters":[{"title":"Identity","status":"completed","draft":paragraph(True),"source_ids":["S1"]}],
           "review":draft,"review_status":"completed","predictions":[],"remote_services":[],
           "report":"STALE "+draft,"status":"completed","id":"x"*32}
    assert not review_ready(state,"en")
    assert "UNCITED_EDITOR_CLAIM" not in lf.assemble(state,"en")
    with build_dossier(state) as handle,zipfile.ZipFile(handle) as bundle:
        assert "UNCITED_EDITOR_CLAIM" not in bundle.read("report.md").decode()
        assert bundle.read("drafts/editor.md").decode()==draft
        assert not json.loads(bundle.read("drafts/editor-audit.json"))["citation_coverage_ok"]
        for row in json.loads(bundle.read("manifest.json"))["files"]:
            assert hashlib.sha256(bundle.read(row["path"])).hexdigest()==row["sha256"]
    assert state["review_status"]=="completed" and state["review"]==draft


def test_inline_code_marker_is_not_an_evidence_citation():
    tick=chr(96)
    assert citation_coverage(paragraph(False)+" "+tick+"[S1]"+tick)["cited_blocks"]==0


def test_owner_view_rechecks_editor_without_rewriting_historical_checkpoint(tmp_path):
    from backend.web_v2.research_jobs import ResearchJobs,write
    from backend.web_v2.job_queue import JobQueue
    import pytest
    queue=JobQueue(tmp_path/"queue");jobs=ResearchJobs(tmp_path/"reports",queue)
    identity="a"*32;qid=queue.submit("alice","protein-research","research")
    queue.finish(qid,"completed")
    request={"id":identity,"owner":"alice","created":1,"title":"Example","queue_id":qid,"resume_count":2,
             "payload":{"model_id":"free","analyses":["properties"],"lang":"en"}}
    draft=paragraph(False)+"\n\n"+paragraph(False)+"\n\n"+paragraph(True)
    state={"workflow":"specialist_research_v2","report":"saved","review":draft,"review_status":"completed",
           "partial":False,"model_calls":15,"minimum_met":True,
           "sources":[{"id":"S1","title":"Reference","url":"https://example.org","evidence_level":"annotation"}],
           "chapters":[{"title":"Identity","status":"completed","draft":paragraph(True),"source_ids":["S1"]}]}
    folder=jobs.folder(identity);write(folder/"request.json",request);write(folder/"state.json",state)
    jobs.records.register("report",identity,"alice","research","Example",1)
    before=(folder/"state.json").read_bytes()
    value=jobs.get("alice",identity)
    assert value["review_status"]=="partial" and value["partial"]
    assert not value["review_audit"]["citation_coverage_ok"] and not value["resumable"]
    assert "Integrated assessment" not in value["report"]
    assert (folder/"state.json").read_bytes()==before
    with pytest.raises(FileNotFoundError):jobs.get("bob",identity)
