import hashlib
import io
import json
import zipfile
import pytest
from backend.web_v2.research_dossier import build_dossier, archive_chunks


def report():
    return {
        "id": "a"*32, "title": "Protein study", "job_number": "VF-TEST",
        "status": "completed", "partial": True, "report": "# Protein\nEvidence [S1].",
        "chapters": [{"id": "../bad", "title": "Identity", "status": "partial",
                      "draft": "Measured evidence [S1].", "source_ids": ["S1"]}],
        "sources": [{"id": "../../secret", "title": "../.env", "kind": "user_material",
                     "url": "/api/report/materials/owned", "text": "[Row 1] 3.2",
                     "extraction": {"truncated": True}, "local_path": "/private/secret"}],
        "sections": [{"id": "properties", "title": "Properties", "data": {"length": 4}}],
        "remote_services": [{"id": "blast", "status": "completed", "count": 1,
                             "query": "MKWA", "hits": [{"accession": "P00001"}]}],
        "owner": "private-owner", "queue": {"log_path": "/private/secret"},
        "session_access_token": "not-exported",
    }


def test_dossier_preserves_evidence_and_hashes_without_using_source_filenames():
    handle = build_dossier(report(), {"sequence": "MKWA", "api_key": "not-exported"})
    with handle, zipfile.ZipFile(handle) as archive:
        names = archive.namelist()
        assert len(names) == len(set(names))
        assert all(not n.startswith("/") and ".." not in n.split("/") for n in names)
        assert "materials/001.txt" in names and "inputs/protein.fasta" in names
        assert json.loads(archive.read("overview.json"))["partial"] is True
        assert archive.read("materials/001.txt") == b"[Row 1] 3.2"
        assert "Original attachments" in archive.read("README.md").decode()
        contents = b"".join(archive.read(n) for n in names)
        assert b"not-exported" not in contents and b"/private/secret" not in contents
        assert b"private-owner" not in contents
        manifest = json.loads(archive.read("manifest.json"))
        assert manifest["partial"] and manifest["report_id"] == "a"*32
        assert {row["path"] for row in manifest["files"]} == set(names) - {"manifest.json"}
        for row in manifest["files"]:
            raw = archive.read(row["path"])
            assert len(raw) == row["bytes"]
            assert hashlib.sha256(raw).hexdigest() == row["sha256"]
        assert json.loads(archive.read("chapters/index.json"))[0]["status"] == "partial"


def test_package_limit_and_missing_report_are_rejected(monkeypatch):
    import backend.web_v2.research_dossier as module
    with pytest.raises(ValueError, match="ready"):
        build_dossier({})
    monkeypatch.setattr(module, "MAX_TOTAL_BYTES", 30)
    with pytest.raises(ValueError, match="size limit"):
        build_dossier(report())


def test_stream_closes_spooled_file_on_complete_and_on_early_close():
    for early in [False, True]:
        handle = build_dossier(report())
        chunks = archive_chunks(handle)
        if early:
            next(chunks)
            chunks.close()
        else:
            assert b"".join(chunks).startswith(b"PK")
        assert handle.closed

@pytest.mark.parametrize("saved_input", [False, True])
def test_download_zip_is_owner_scoped_and_private(tmp_path, monkeypatch, saved_input):
    from fastapi import FastAPI, Request
    from fastapi.testclient import TestClient
    from backend.web_v2 import report_api as api
    from backend.web_v2.research_jobs import ResearchJobs, write
    from backend.web_v2.job_queue import JobQueue
    queue = JobQueue(tmp_path / "queue")
    jobs = ResearchJobs(tmp_path / "reports", queue)
    identity = "b" * 32
    qid = queue.submit("alice", "protein-research", "research")
    queue.finish(qid, "completed")
    jobs.records.register("report", identity, "alice", "research", "Example", 1)
    write(jobs.folder(identity) / "request.json", {
        "id": identity, "owner": "alice", "title": "Example", "created": 1, "queue_id": qid,
        "payload": {"model_id": "fixture", "analyses": ["properties"], "sequence": "MKWA"}})
    state = {"report": "# Example", "sources": [],
             "source_review": {"policy": "source_review.v1", "status": "partial", "units": {},
                               "attempts": [{"response": "PRIVATE_REVIEW_RESPONSE"}]}}

    if saved_input:
        from backend.web_v2.research_input_snapshot import save_inputs
        ref = save_inputs(jobs.folder(identity) / "input_snapshots", {
            "sequence": "ACDE", "structure_pdb": "", "origin": "uniprot",
            "entry": {"primaryAccession": "P00918"}, "protein": {"name": "Reference"}})
        state["input_snapshot"] = ref
    write(jobs.folder(identity) / "state.json", state)
    monkeypatch.setattr(api, "ResearchJobs", lambda: jobs)
    app = FastAPI(); app.include_router(api.router)
    def test_owner(request: Request): return request.headers.get("x-test-owner", "bob")
    app.dependency_overrides[api.owner] = test_owner
    with TestClient(app) as client:
        path = "/api/report/jobs/" + identity + "/download?format=zip"
        assert client.get(path).status_code == 404
        response = client.get(path, headers={"x-test-owner": "alice"})
        detail = client.get("/api/report/jobs/" + identity, headers={"x-test-owner": "alice"})
        assert detail.status_code == 200 and "PRIVATE_REVIEW_RESPONSE" not in detail.text
        assert detail.json()["source_review_summary"]["policy"] == "source_review.v1"
        assert "source_review" in json.loads((jobs.folder(identity) / "state.json").read_text())

        if saved_input:
            snapshot = jobs.folder(identity) / "input_snapshots" / (ref["sha256"] + ".json")
            snapshot.write_bytes(b"changed")
            assert client.get(path, headers={"x-test-owner": "alice"}).status_code == 422
            assert client.get(path).status_code == 404
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/zip"
    assert response.headers["cache-control"] == "private, no-store"
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        assert b"PRIVATE_REVIEW_RESPONSE" in archive.read("verification/source-review.json")
        assert archive.read("inputs/protein.fasta") == (b">query\nACDE\n" if saved_input else b">query\nMKWA\n")
