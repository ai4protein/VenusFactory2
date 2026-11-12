"""Owned real-SQLite receipt replay; protocol doubles never train or infer biology."""

import copy
import hashlib
import json
from unittest.mock import AsyncMock

import pytest

from backend.web_v2 import research_receipts as receipts
from backend.web_v2.job_queue import JobQueue
from backend.web_v2.research_input_snapshot import save_inputs
from backend.web_v2.research_recovery import invalidate_chapter, recover_evidence
from mcp_tools.scheduler import client
from mcp_tools.scheduler import jobs as scheduler
from mcp_tools.scheduler.receipt_binding import value_digest


@pytest.fixture
def ledger(tmp_path, monkeypatch):
    queue = JobQueue(tmp_path / "queue")
    jobs = scheduler.ToolJobs(queue)
    parent, _ = queue.enqueue("owner-a", "protein-research", "research", {"fixture": True}, "receipt-test")
    assert queue.claim_next("receipt-test", "research")["id"] == parent
    monkeypatch.setenv("VENUS_QUEUE_OWNER", "owner-a")
    monkeypatch.setenv("VENUS_QUEUE_JOB_ID", "new-parent")
    calls = []

    async def rpc(name, args):
        calls.append(name)
        assert name == "tool_job", "Receipt recovery must never submit/reserve quota"
        return jobs.get(args["identity"], args["owner"])

    monkeypatch.setattr(client, "rpc", rpc)

    def create(tool, args, value, status="completed", owner="owner-a"):
        identity, _ = queue.enqueue(
            owner,
            "mcp:cpu_analysis",
            "cpu",
            {"fixture": True},
            "receipt-test",
            metadata={
                "tool": tool,
                "category": "cpu_analysis",
                "parent_job": parent,
                "arguments_sha256": value_digest(args),
            },
        )
        row = queue.claim_next("receipt-test", "cpu")
        assert row["id"] == identity
        if value is not None:
            scheduler.save_json(queue.files(identity) / "mcp-result.json", {"value": value, "is_error": False})
        if status != "running":
            queue.finish(identity, status, lease=row["lease"])
        link = {
            "version": 1,
            "tool": tool,
            "arguments": copy.deepcopy(args),
            "arguments_sha256": value_digest(args),
            "parent_job": parent,
            "owner_sha256": receipts.owner_digest(owner),
            "task_id": identity,
        }
        return link

    return queue, jobs, create, calls


@pytest.mark.asyncio
async def test_captured_running_link_recovers_finished_result_without_submit(ledger, monkeypatch):
    queue, jobs, create, calls = ledger
    args = {"query": "protein evidence", "limit": 6}
    value = {"status": "completed", "records": []}
    created = create("query_europepmc", args, value, status="running")
    row = queue.get(created["task_id"], "owner-a")
    monkeypatch.setenv("VENUS_QUEUE_JOB_ID", created["parent_job"])

    async def interrupted_delivery(name, arguments, **kwargs):
        kwargs["on_status"](jobs.get(row["id"], "owner-a"))
        queue.finish(row["id"], "completed", lease=row["lease"])
        raise KeyError("value")

    submit = AsyncMock(side_effect=interrupted_delivery)
    monkeypatch.setattr(client, "invoke", submit)
    state, snapshots = {}, []
    with receipts.receipt_scope(state, lambda: snapshots.append(copy.deepcopy(state)), "chapter:x:literature"):
        with pytest.raises(KeyError):
            await receipts.invoke("query_europepmc", args, owner="owner-a", parent_job=created["parent_job"])
        assert snapshots[-1]["tool_receipts"]["chapter:x:literature"]["task_id"] == row["id"]
        # Resuming has a new parent but the old task's original binding remains.
        result = await receipts.invoke("query_europepmc", args, owner="owner-a", parent_job="new-parent")
    assert result == value and submit.await_count == 1 and calls == ["tool_job"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fault", ["owner", "server_owner", "arguments", "server_arguments", "tool", "parent", "result"]
)
async def test_stale_or_mismatched_binding_cannot_deliver_success(ledger, monkeypatch, fault):
    queue, _, create, _ = ledger
    args = {"query": "protein", "limit": 6}
    link = create("query_europepmc", args, {"status": "completed", "records": []})
    if fault == "owner":
        monkeypatch.setenv("VENUS_QUEUE_OWNER", "owner-b")
    if fault == "server_owner":
        # Even a forged checkpoint owner hash does not bypass SQLite ownership.
        monkeypatch.setenv("VENUS_QUEUE_OWNER", "owner-b")
        link["owner_sha256"] = receipts.owner_digest("owner-b")
    if fault == "arguments":
        args = {**args, "limit": 5}
    if fault == "server_arguments":
        args = {**args, "limit": 5}
        link["arguments_sha256"] = value_digest(args)
    if fault == "tool":
        link["tool"] = "query_other"
    if fault == "parent":
        link["parent_job"] = "different-parent"
    if fault == "result":
        link["result_sha256"] = "0" * 64
    with pytest.raises(receipts.ReceiptUnavailable):
        await receipts.recover_result(link, "query_europepmc", args)
    assert len([j for j in queue.list("owner-a") if j["kind"].startswith("mcp:")]) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status,value",
    [
        ("completed", None),
        ("running", {"status": "completed"}),
        ("failed", {"status": "completed"}),
        ("cancelled", None),
    ],
)
async def test_non_successful_receipt_stays_typed_unavailable(ledger, status, value):
    _, _, create, _ = ledger
    args = {"query": "protein", "limit": 6}
    link = create("query_europepmc", args, value, status=status)
    with pytest.raises(receipts.ReceiptUnavailable, match="result_not_successful"):
        await receipts.recover_result(link, "query_europepmc", args)


@pytest.mark.asyncio
async def test_lost_submit_acknowledgement_does_not_speculatively_repeat(monkeypatch):
    state = {}
    submit = AsyncMock(side_effect=TimeoutError)
    monkeypatch.setattr(client, "invoke", submit)
    with receipts.receipt_scope(state, lambda: None, "prediction:x"):
        with pytest.raises(TimeoutError):
            await receipts.invoke("tool", {"x": 1}, owner="a", parent_job="p")
        with pytest.raises(receipts.ReceiptUnavailable):
            await receipts.invoke("tool", {"x": 1}, owner="a", parent_job="p")
    assert submit.await_count == 1


def collector(state):
    def add(rows):
        result = []
        for row in rows:
            found = next((s for s in state["sources"] if s.get("url") and s["url"] == row.get("url")), None)
            if found is None:
                found = {**row, "id": "S" + str(len(state["sources"]) + 1)}
                state["sources"].append(found)
            result.append(found)
        return result

    return add


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fault",
    [
        None,
        "input_changed",
        "output_changed",
        "output_missing",
        "snapshot_missing",
        "same_sequence_score",
        "dataset_label",
        "model_label",
        "preview_truncated",
        "preview_missing",
    ],
)
async def test_recovers_deepsol_and_topic_with_original_input_without_retraining(ledger, tmp_path, fault):
    queue, _, create, calls = ledger
    sequence = "MKTAYIAKQRQISFVK"
    folder = tmp_path / "report"
    prediction_dir = folder / "predictions"
    prediction_dir.mkdir(parents=True)
    (prediction_dir / "query.fasta").write_text(">query\n" + sequence + "\n")
    (prediction_dir / "deep.csv").write_text("header,sequence,score\nquery," + sequence + ",0.8\n")
    row = {
        "id": "deep",
        "task": "Solubility",
        "model": "ESM2-650M",
        "dataset": "DeepSol",
        "sequence_sha256": hashlib.sha256(sequence.encode()).hexdigest(),
        "status": "unavailable",
        "tool": "predict_protein_function",
    }
    args = {
        "fasta_file": str(prediction_dir / "query.fasta"),
        "output_file": str(prediction_dir / "deep.csv"),
        "task": row["task"],
        "model_name": row["model"],
        "adapter_path": "ckpt/DeepSol/esm",
    }
    saved_result = {
        "status": "success",
        "biological_metadata": {
            "task": row["task"],
            "dataset_name": row["dataset"],
            "model_name": row["model"],
            "fasta_file": args["fasta_file"],
            "adapter_path": args["adapter_path"],
            "sequence_count": 1,
        },
        "file_info": {
            "file_name": "deep.csv",
            "file_size": (prediction_dir / "deep.csv").stat().st_size,
            "format": "csv",
        },
        "content_preview": json.dumps([{"header": "query", "sequence": sequence, "score": 0.8}]),
    }
    if fault == "preview_truncated":
        saved_result["content_preview"] = saved_result["content_preview"][:-4]
    if fault == "preview_missing":
        saved_result.pop("content_preview")
    prediction_link = create(row["tool"], args, json.dumps(saved_result))
    query = 'TITLE_ABS:"Carbonic anhydrase 2" AND solubility'
    paper = {
        "title": "Public protein measurement",
        "url": "https://pubmed.ncbi.nlm.nih.gov/1/",
        "abstractText": "Experimental evidence",
        "evidence_level": "abstract",
    }
    topic_link = create("query_europepmc", {"query": query, "limit": 6}, {"status": "completed", "records": [paper]})
    state = {
        "predictions": [row],
        "sources": [],
        "sections": [],
        "collection_partial": True,
        "collection_nonprediction_partial": False,
        "input_snapshot": save_inputs(folder / "inputs", {"sequence": sequence}),
        "steps": [{"id": "ai_predictions", "status": "partial"}],
        "chapters": [
            {
                "id": "biophysics_production",
                "literature_query": query,
                "retrieval": [{"tool": "literature", "status": "unavailable"}],
            }
        ],
        "tool_receipts": {"prediction:deep": prediction_link, "chapter:biophysics_production:literature": topic_link},
    }
    if fault == "input_changed":
        (prediction_dir / "query.fasta").write_text(">query\nAAAAAAAAAAAA\n")
    if fault == "output_changed":
        (prediction_dir / "deep.csv").write_text("header,sequence,score\nquery,AAAAAAAAAAAA,0.8\n")
    if fault == "output_missing":
        (prediction_dir / "deep.csv").unlink()
    if fault == "snapshot_missing":
        state.pop("input_snapshot")
    if fault == "same_sequence_score":
        (prediction_dir / "deep.csv").write_text("header,sequence,score\nquery," + sequence + ",0.9\n")
    if fault == "dataset_label":
        row["dataset"] = "DifferentDataset"
    if fault == "model_label":
        row["model"] = "DifferentModel"
    topics = await recover_evidence(
        state, lambda: None, collector(state), service_cache=folder / "remote", input_store=folder / "inputs"
    )
    assert topics == {"biophysics_production"}
    if fault:
        assert row["status"] == "unavailable" and row["recovery"]["status"] == "unavailable"
        assert state["collection_partial"] is True and not state["sections"]
        assert len(state["sources"]) == 1
        return
    assert row["status"] == "completed" and row["values"] == [{"score": 0.8}]
    assert len(state["sources"]) == 2 and len(state["sections"]) == 1
    assert state["collection_partial"] is False
    assert state["chapters"][0]["topic_source_ids"] == ["S2"]
    assert (
        calls == ["tool_job", "tool_job"]
        and len([j for j in queue.list("owner-a") if j["kind"].startswith("mcp:")]) == 2
    )
    await recover_evidence(
        state, lambda: None, collector(state), service_cache=folder / "remote", input_store=folder / "inputs"
    )
    assert len(calls) == 2 and len(state["sources"]) == 2
    # Missing bindings in old checkpoints must not be guessed from queue rows.
    old = {"predictions": [{**row, "status": "unavailable"}], "sources": [], "chapters": []}
    await recover_evidence(
        old, lambda: None, collector(old), service_cache=folder / "remote", input_store=folder / "inputs"
    )
    assert old["predictions"][0]["status"] == "unavailable" and len(calls) == 2


def test_invalidating_only_changed_chapter_preserves_versions_and_removes_its_reviews():
    chapter = {
        "id": "biophysics_production",
        "draft": "Old evidence [S1]",
        "audit": {},
        "source_ids": ["S1"],
        "status": "completed",
        "draft_status": "completed",
        "source_review": {"status": "completed"},
    }
    unaffected = {"id": "structure_binding", "draft": "Other evidence", "status": "completed"}
    review = {
        "documents": {
            chapter["id"]: {"status": "completed"},
            "structure_binding": {"status": "completed"},
            "editor": {"status": "completed"},
        },
        "units": {
            "a": {"document_id": chapter["id"]},
            "b": {"document_id": "structure_binding"},
            "c": {"document_id": "editor"},
        },
    }
    state = {
        "chapters": [chapter, unaffected],
        "source_review": copy.deepcopy(review),
        "review": "Old synthesis",
        "review_status": "completed",
    }
    invalidate_chapter(state, chapter, "new evidence")
    assert chapter["draft_status"] == "pending" and "source_review" not in chapter
    assert chapter["draft_history"][0]["draft"] == "Old evidence [S1]"
    assert chapter["evidence_history"][0]["source_review"]["status"] == "completed"
    assert state["source_review_history"][0] == review
    assert state["source_review"]["units"] == {"b": {"document_id": "structure_binding"}}
    assert unaffected["status"] == "completed" and state["review_status"] == "pending" and "review" not in state


def test_scheduler_generates_parameter_digest_without_publishing_arguments(tmp_path, monkeypatch):
    queue = JobQueue(tmp_path / "submit-queue")
    monkeypatch.setattr(scheduler, "assert_tool_call_allowed", lambda *args: None)
    monkeypatch.setattr(scheduler, "native_tools", lambda: {})
    monkeypatch.setattr(
        scheduler, "classify", lambda *args: {"category": "cpu_analysis", "resource": "cpu", "timeout_seconds": 60}
    )

    def launch(command, **kwargs):
        return queue.enqueue(
            kwargs["owner"],
            kwargs["kind"],
            kwargs["resource"],
            {"fixture": True},
            "submit-binding",
            metadata=kwargs["metadata"],
        )

    monkeypatch.setattr(scheduler, "launch", launch)
    args = {"query": "public protein", "limit": 6}
    observed = scheduler.ToolJobs(queue).submit("query_europepmc", args, owner="owner-a", route="mcp")
    assert observed["arguments_sha256"] == value_digest(args)
    assert "arguments" not in observed
    assert queue.get(observed["id"], "owner-a")["metadata"]["arguments_sha256"] == value_digest(args)


@pytest.mark.asyncio
@pytest.mark.parametrize("two_topics", [False, True])
async def test_resume_rewrites_and_reaudits_only_chapter_with_new_topic_evidence(ledger, monkeypatch, two_topics):
    from backend.web_v2 import research_longform as lf
    from backend.web_v2.research_documents import chapter_document
    from backend.web_v2.research_source_review import VERSION, document_receipt, units
    from research_source_review_fixture import with_source_review

    _, _, create, calls = ledger
    source = {
        "id": "S1",
        "kind": "literature",
        "title": "Old measured evidence",
        "url": "https://example.org/old",
        "text": "Measured evidence",
        "evidence_level": "abstract",
    }
    query = 'TITLE_ABS:"Carbonic anhydrase 2" AND solubility'
    value = {
        "status": "completed",
        "records": [
            {
                "title": "Recovered production study",
                "url": "https://example.org/new",
                "abstractText": "Measured evidence",
                "evidence_level": "abstract",
            }
        ],
    }
    link = create("query_europepmc", {"query": query, "limit": 6}, value)
    checkpoint = {
        "workflow": "specialist_research_v2",
        "phase": "incomplete",
        "collection_complete": True,
        "collection_partial": False,
        "partial": True,
        "model_calls": 0,
        "model_call_limit": 96,
        "sources": [source],
        "protein": {"name": "Carbonic anhydrase 2"},
        "steps": [],
        "sections": [],
        "chapters": [],
        "report": "Old report",
        "evidence_review_required": VERSION,
        "tool_receipts": {"chapter:biophysics_production:literature": link},
    }
    cached = {}
    for identity, title, *_ in lf.CHAPTERS:
        chapter = {
            "id": identity,
            "title": title,
            "status": "completed",
            "draft_status": "completed",
            "attempts": 1,
            "draft": "Measured evidence " * 260 + " [S1]",
            "source_ids": ["S1"],
            "evidence_review_required": VERSION,
        }
        doc = chapter_document(chapter)
        for unit in units(doc, [source]):
            cached[unit["id"]] = {
                "unit_id": unit["id"],
                "document_id": identity,
                "input_sha256": unit["input_sha256"],
                "passed": True,
            }
        chapter["source_review"] = document_receipt(doc, [source], cached)
        if identity == "biophysics_production":
            chapter.update(literature_query=query, retrieval=[{"tool": "literature", "status": "unavailable"}])
        checkpoint["chapters"].append(chapter)
    checkpoint["source_review"] = {
        "policy": VERSION,
        "units": cached,
        "attempts": [],
        "documents": {c["id"]: c["source_review"] for c in checkpoint["chapters"]},
    }
    changed_ids = {"biophysics_production"}
    if two_topics:
        changed_ids.add("engineering_design")
        chapter = checkpoint["chapters"][6]
        chapter.update(literature_query=query, retrieval=[{"tool": "literature", "status": "unavailable"}])
        checkpoint["tool_receipts"]["chapter:engineering_design:literature"] = copy.deepcopy(link)
    original = copy.deepcopy(checkpoint)
    writes = []

    @with_source_review
    async def writer(model, messages, tokens):
        if "assigned chapter" in messages[0]["content"]:
            writes.append(json.loads(messages[1]["content"])["chapter"])
            return "Measured evidence " * 260 + " [S1] [S2]", False
        return "Measured evidence " * 260 + " [S1]", False

    monkeypatch.setattr(lf.research, "complete", writer)
    forbidden = AsyncMock(side_effect=AssertionError("No computation resubmit"))
    monkeypatch.setattr(client, "invoke", forbidden)
    monkeypatch.setenv("VENUS_REPORT_MODEL_CALL_LIMIT", "96")
    result = await lf.run_long_report(
        {"model_id": "fixture", "analyses": ["literature"], "lang": "en"}, lambda state: None, checkpoint=checkpoint
    )
    assert checkpoint == original
    assert set(writes) == {c["title"] for c in checkpoint["chapters"] if c["id"] in changed_ids}
    revised = next(c for c in result["chapters"] if c["id"] == "biophysics_production")
    assert revised["attempts"] == 2 and revised["source_review"]["status"] == "completed"
    assert "S2" in revised["source_ids"] and revised["draft_history"][0]["draft"] == original["chapters"][5]["draft"]
    assert len(result["source_review_history"]) == 1 and result["model_calls"] > 1
    for before, after in zip(original["chapters"], result["chapters"]):
        if before["id"] not in changed_ids:
            assert before["attempts"] == after["attempts"] and before["source_review"] == after["source_review"]
    forbidden.assert_not_awaited()
    assert all(name == "tool_job" for name in calls)


@pytest.mark.parametrize("include_evidence", [False, True])
def test_private_receipt_parameters_never_leave_owned_report_response(tmp_path, include_evidence):
    import zipfile

    from backend.web_v2.research_dossier import build_dossier
    from backend.web_v2.research_jobs import ResearchJobs, write

    queue = JobQueue(tmp_path / "queue")
    reports = ResearchJobs(tmp_path / "reports", queue)
    identity = "a" * 32
    qid = queue.submit("owner-a", "protein-research", "research")
    queue.finish(qid, "interrupted")
    reports.records.register("report", identity, "owner-a", "research", "Example", 1)
    folder = reports.folder(identity)
    write(
        folder / "request.json",
        {
            "id": identity,
            "owner": "owner-a",
            "created": 1,
            "title": "Example",
            "queue_id": qid,
            "payload": {"model_id": "fixture", "analyses": ["properties"]},
        },
    )
    private = {
        "task_id": "private-internal-link",
        "arguments": {"fasta_file": "/private/internal-input"},
        "owner_sha256": "private-owner-binding",
    }
    write(folder / "state.json", {"report": "# Report", "tool_receipts": {"prediction:x": private}, "sources": []})
    result = reports.get("owner-a", identity, include_evidence=include_evidence)
    assert "tool_receipts" not in result
    assert json.loads((folder / "state.json").read_text())["tool_receipts"]["prediction:x"] == private
    with pytest.raises(FileNotFoundError):
        reports.get("owner-b", identity, include_evidence=include_evidence)
    handle = build_dossier(result)
    with handle, zipfile.ZipFile(handle) as archive:
        contents = b"".join(archive.read(name) for name in archive.namelist())
    assert b"private-internal-link" not in contents and b"private-owner-binding" not in contents
