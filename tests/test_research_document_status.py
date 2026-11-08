"""API and archive acceptance must describe the current reviewed documents."""
import asyncio
import copy
import hashlib
import json
import zipfile
from unittest.mock import patch

import pytest
from research_source_review_fixture import with_source_review
from backend.web_v2 import research_longform as lf, research_report as research
from backend.web_v2.job_queue import JobQueue
from backend.web_v2.research_dossier import build_dossier
from backend.web_v2.research_jobs import ResearchJobs, write

PROSE = "The reference records the measured observation under the stated assay conditions. " * 40 + "[S1]"


@pytest.fixture
def accepted():
    async def collect(*args, **kwargs):
        return {"protein": {"name": "Fixture protein"}, "steps": [], "sections": [],
                "sources": [{"id": "S1", "title": "Fixture", "url": "https://example.org/reference",
                             "kind": "database", "evidence_level": "annotation", "text": PROSE}]}

    @with_source_review
    async def complete(*args):
        return PROSE, False

    with patch.object(research, "run_research", side_effect=collect), patch.object(research, "complete", side_effect=complete):
        value = asyncio.run(lf.run_long_report(
            {"model_id": "fixture", "analyses": ["properties"], "lang": "en"}, lambda state: None))
    assert value["completed_chapters"] == 8 and not value["partial"]
    return {**value, "status": "completed", "id": "a" * 32}


def owner_job(tmp_path, state):
    queue = JobQueue(tmp_path / "queue")
    jobs = ResearchJobs(tmp_path / "reports", queue)
    identity = state["id"]
    qid = queue.submit("alice", "protein-research", "research")
    queue.finish(qid, "completed")
    jobs.records.register("report", identity, "alice", "research", "Fixture protein", 1)
    write(jobs.folder(identity) / "request.json", {
        "id": identity, "owner": "alice", "created": 1, "title": "Fixture protein",
        "queue_id": qid, "payload": {"model_id": "fixture", "analyses": ["properties"], "lang": "en"}})
    write(jobs.folder(identity) / "state.json", state)
    return jobs, identity


def invalidate_chapters(state, count):
    for chapter in state["chapters"][:count]:
        chapter["draft"] += " NEW_UNREVIEWED_CLAIM [S1]"
    return state


@pytest.mark.parametrize("changed", [1, 4])
def test_owner_counts_only_currently_reviewed_chapters(tmp_path, accepted, changed):
    state = invalidate_chapters(accepted, changed)
    jobs, identity = owner_job(tmp_path, state)
    path = jobs.folder(identity) / "state.json"
    before = path.read_bytes()

    result = jobs.get("alice", identity, include_evidence=True)

    assert result["completed_chapters"] == 8 - changed
    assert result["minimum_met"] is (8 - changed >= 5)
    assert result["partial"] and result["status"] == "incomplete"
    assert result["phase"] == "incomplete"
    assert result["resumable"] and result["model_calls"] == state["model_calls"]
    assert result["review_status"] == "partial"
    assert "NEW_UNREVIEWED_CLAIM" not in result["report"]
    assert result["chapters"][0]["draft"] == state["chapters"][0]["draft"]
    assert result["source_review"] == state["source_review"]
    assert path.read_bytes() == before
    with pytest.raises(FileNotFoundError):
        jobs.get("bob", identity)


def test_archive_reconciles_counts_and_keeps_original_drafts(accepted):
    state = invalidate_chapters(accepted, 4)
    original = copy.deepcopy(state)
    with build_dossier(state) as handle, zipfile.ZipFile(handle) as bundle:
        overview = json.loads(bundle.read("overview.json"))
        assert overview["completed_chapters"] == 4 and not overview["minimum_met"]
        assert overview["partial"] and overview["status"] == "incomplete"
        assert b"NEW_UNREVIEWED_CLAIM" not in bundle.read("report.md")
        assert bundle.read("drafts/01.md").decode() == state["chapters"][0]["draft"]
        assert bundle.read("drafts/editor.md").decode() == state["review"]
        assert json.loads(bundle.read("chapters/index.json"))[0]["status"] == "partial"
        manifest = json.loads(bundle.read("manifest.json"))
        assert manifest["partial"]
        for row in manifest["files"]:
            assert hashlib.sha256(bundle.read(row["path"])).hexdigest() == row["sha256"]
    assert state == original


@pytest.mark.parametrize("queue_status", ["queued", "running", "cancelled", "interrupted"])
def test_review_counters_do_not_override_queue_lifecycle(tmp_path, monkeypatch, accepted, queue_status):
    state = invalidate_chapters(accepted, 4)
    jobs, identity = owner_job(tmp_path, state)
    monkeypatch.setattr(jobs.queue, "public", lambda *args: {"status": queue_status, "message": ""})
    result = jobs.get("alice", identity)
    assert result["status"] == queue_status
    assert result["completed_chapters"] == 4 and not result["minimum_met"]
    assert result["resumable"] is (queue_status == "interrupted")


def test_report_level_review_policy_also_applies_to_legacy_chapter_rows(tmp_path, accepted):
    for chapter in accepted["chapters"]:
        chapter.pop("evidence_review_required")
        chapter.pop("source_review")
    jobs, identity = owner_job(tmp_path, accepted)
    result = jobs.get("alice", identity)
    assert result["completed_chapters"] == 0 and not result["minimum_met"]
    assert result["status"] == "incomplete"
    assert PROSE not in result["report"]


def test_accepted_report_has_identical_api_and_archive_completion(tmp_path, accepted):
    original = copy.deepcopy(accepted)
    jobs, identity = owner_job(tmp_path, accepted)
    result = jobs.get("alice", identity, include_evidence=True)
    assert result["status"] == "completed" and not result["partial"]
    assert result["completed_chapters"] == 8 and result["minimum_met"]
    assert not result["resumable"]
    with build_dossier(result) as handle, zipfile.ZipFile(handle) as bundle:
        overview = json.loads(bundle.read("overview.json"))
        assert overview["status"] == "completed" and not overview["partial"]
        assert overview["completed_chapters"] == 8 and overview["minimum_met"]
        assert bundle.read("report.md").decode() == result["report"]
    assert accepted == original


def test_rechecked_report_still_cannot_exceed_resume_or_model_budget(tmp_path, accepted):
    state = invalidate_chapters(accepted, 4)
    state["model_calls"] = state["model_call_limit"]
    jobs, identity = owner_job(tmp_path, state)
    result = jobs.get("alice", identity)
    assert result["status"] == "incomplete" and not result["resumable"]
    assert result["completed_chapters"] == 4
    with pytest.raises(ValueError, match="budget"):
        jobs.resume("alice", identity)
