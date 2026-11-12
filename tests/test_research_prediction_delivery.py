"""Real files, two-owner SQLite jobs and HTTP downloads; no inference or quota."""

import copy
import csv
import hashlib
import io
import json
import os
import zipfile
from types import SimpleNamespace

import pytest

from backend.web_v2 import research_prediction_delivery as delivery
from backend.web_v2.job_queue import JobQueue
from backend.web_v2.research_dossier import build_dossier
from backend.web_v2.research_input_snapshot import save_inputs
from backend.web_v2.research_jobs import ResearchJobs, write
from backend.web_v2.research_receipts import owner_digest
from mcp_tools.models.factory.prediction.output_scale import output_scale
from mcp_tools.scheduler import jobs as scheduler
from mcp_tools.scheduler.receipt_binding import value_digest


@pytest.fixture
def reports(tmp_path):
    queue = JobQueue(tmp_path / "queue")
    service = ResearchJobs(tmp_path / "reports", queue)
    counter = 0

    def create(owner="alice", *, worker_hash=True, preview="complete", status="completed", vector=False):
        nonlocal counter
        counter += 1
        identity = f"{counter:032x}"
        folder = service.folder(identity)
        prediction_dir = folder / "predictions"
        prediction_dir.mkdir(parents=True)
        sequence = "MKTAYIAKQRQISFVK"
        (prediction_dir / "query.fasta").write_text(">query\n" + sequence + "\n")
        key = "a" * 16
        path = prediction_dir / (key + ".csv")
        score = [[0.2, 0.8]] * 100 if vector else -0.9709106683731079
        column = "probabilities" if vector else "prediction"
        sample = {"header": "query", "sequence": sequence, column: score}
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=sample)
            writer.writeheader()
            writer.writerow({k: json.dumps(v) if isinstance(v, list) else v for k, v in sample.items()})
        raw = path.read_bytes()
        scale = output_scale({"problem_type": "regression", "normalize": "standard"})
        row = {
            "id": key,
            "task": "Optimal Temperature",
            "model": "ESM2-650M",
            "dataset": "DeepET_Topt",
            "tool": "predict_protein_function",
            "status": "completed",
            "values": [{column: str(score) if not vector else json.dumps(score)}],
            "sequence_sha256": hashlib.sha256(sequence.encode()).hexdigest(),
            "output_scale": scale,
        }
        args = {
            "fasta_file": str(prediction_dir / "query.fasta"),
            "output_file": str(path),
            "task": row["task"],
            "model_name": row["model"],
            "adapter_path": "/private/checkpoint",
        }
        value = {
            "status": "success",
            "file_info": {"file_name": path.name, "file_size": len(raw), "format": "csv"},
            "biological_metadata": {
                "task": row["task"],
                "dataset_name": row["dataset"],
                "model_name": row["model"],
                "fasta_file": args["fasta_file"],
                "adapter_path": args["adapter_path"],
                "sequence_count": 1,
                "output_scale": scale,
            },
            "content_preview": json.dumps([sample]) if preview == "complete" else "[{truncated",
        }
        if worker_hash:
            value["file_info"]["sha256"] = hashlib.sha256(raw).hexdigest()
        parent, _ = queue.enqueue(
            owner, "protein-research", "research", {"fixture": True}, "delivery-test", metadata={"report_id": identity}
        )
        parent_row = queue.claim_next("delivery-test", "research")
        assert parent_row["id"] == parent
        task, _ = queue.enqueue(
            owner,
            "mcp:cpu_analysis",
            "cpu",
            {"fixture": True},
            "delivery-test",
            metadata={
                "tool": row["tool"],
                "category": "cpu_analysis",
                "parent_job": parent,
                "arguments_sha256": value_digest(args),
            },
        )
        claimed = queue.claim_next("delivery-test", "cpu")
        assert claimed["id"] == task
        scheduler.save_json(queue.files(task) / "mcp-result.json", {"value": json.dumps(value), "is_error": False})
        queue.finish(task, status, lease=claimed["lease"])
        queue.finish(parent, "completed", lease=parent_row["lease"])
        link = {
            "version": 1,
            "tool": row["tool"],
            "arguments": args,
            "arguments_sha256": value_digest(args),
            "owner_sha256": owner_digest(owner),
            "parent_job": parent,
            "task_id": task,
        }
        state = {
            "report": "# Protein report",
            "predictions": [row],
            "sources": [],
            "input_snapshot": save_inputs(folder / "input_snapshots", {"sequence": sequence}),
            "tool_receipts": {"prediction:" + key: link},
        }
        request = {
            "id": identity,
            "owner": owner,
            "created": counter,
            "title": "Protein report",
            "queue_id": parent,
            "payload": {"model_id": "fixture", "analyses": ["ai_predictions"], "sequence": sequence},
        }
        service.records.register("report", identity, owner, "research", request["title"], counter)
        write(folder / "request.json", request)
        write(folder / "state.json", state)
        return SimpleNamespace(
            id=identity,
            owner=owner,
            folder=folder,
            row=row,
            state=state,
            link=link,
            args=args,
            value=value,
            raw=raw,
            path=path,
            parent=parent,
            task=task,
            request=request,
        )

    return service, queue, create


def public(service, item):
    write(item.folder / "state.json", item.state)
    return service.get(item.owner, item.id, include_evidence=True)


def collect(service, item):
    return delivery.prediction_artifacts(service, item.owner, public(service, item))


def test_real_owned_hash_bound_csv_is_downloaded_without_reopening_or_submitting(reports, monkeypatch):
    service, queue, create = reports
    item = create(vector=True, preview="truncated")
    before = queue.list(item.owner)
    monkeypatch.setattr(
        scheduler.ToolJobs, "submit", lambda *a, **k: pytest.fail("Download must not submit or debit quota")
    )
    result = public(service, item)
    artifacts = delivery.prediction_artifacts(service, item.owner, result)
    assert artifacts[0]["status"] == "included" and artifacts[0]["validation"] == "worker_sha256"
    # Frozen bytes remain valid even if the user's file changes before ZIP creation.
    item.path.write_bytes(b"changed after validated copy")
    artifacts[0]["archive_path"] = "../../secret"
    with build_dossier(result, prediction_files=artifacts) as handle, zipfile.ZipFile(handle) as archive:
        assert archive.read("predictions/001.csv") == item.raw
        index = json.loads(archive.read("predictions/index.json"))[0]
        assert index["artifact"]["status"] == "included"
        assert index["artifact"]["sha256"] == hashlib.sha256(item.raw).hexdigest()
        assert index["output_scale"]["value_space"] == "standardized_target"
        manifest = json.loads(archive.read("manifest.json"))
        entry = next(row for row in manifest["files"] if row["path"] == "predictions/001.csv")
        assert entry["sha256"] == index["artifact"]["sha256"] and entry["bytes"] == len(item.raw)
        all_bytes = b"".join(archive.read(name) for name in archive.namelist())
        assert b"/private/checkpoint" not in all_bytes and item.task.encode() not in all_bytes
        assert "../../secret" not in archive.namelist()
    assert [(x["id"], x["status"]) for x in queue.list(item.owner)] == [(x["id"], x["status"]) for x in before]


@pytest.mark.parametrize("preview,expected", [("complete", "included"), ("truncated", "unverified")])
def test_legacy_complete_values_only_fallback_is_explicit_not_original_byte_proof(reports, preview, expected):
    service, _, create = reports
    item = create(worker_hash=False, preview=preview)
    artifacts = collect(service, item)
    assert artifacts[0]["status"] == expected
    if expected == "included":
        assert artifacts[0]["validation"] == "complete_saved_values"
    with build_dossier(public(service, item), prediction_files=artifacts) as handle, zipfile.ZipFile(handle) as archive:
        index = json.loads(archive.read("predictions/index.json"))[0]
        assert index["artifact"]["status"] == expected
        assert ("predictions/001.csv" in archive.namelist()) == (expected == "included")


@pytest.mark.parametrize(
    "fault",
    [
        "value",
        "same_value_different_bytes",
        "dataset",
        "scale",
        "arguments",
        "missing_link",
        "missing_server_hash",
        "missing_result",
        "parent_report",
        "snapshot",
        "input",
        "failed",
        "unfinished",
    ],
)
def test_tamper_missing_binding_and_unfinished_steps_stay_unverified(reports, fault):
    service, queue, create = reports
    item = create(status="failed" if fault == "failed" else "completed")
    if fault == "value":
        item.path.write_bytes(item.raw.replace(b"-0.9709106683731079", b"-0.9709106683731078"))
    elif fault == "same_value_different_bytes":
        item.path.write_bytes(item.raw.replace(b"query,", b'"query",'))
    elif fault == "dataset":
        item.row["dataset"] = "OtherDataset"
    elif fault == "scale":
        item.row["output_scale"] = {**item.row["output_scale"], "unit": "Celsius"}
    elif fault == "arguments":
        item.link["arguments"]["adapter_path"] = "/other/model"
        item.link["arguments_sha256"] = value_digest(item.link["arguments"])
    elif fault == "missing_link":
        item.state.pop("tool_receipts")
    elif fault == "missing_server_hash":
        with queue.connection() as con:
            meta = queue.get(item.task, item.owner)["metadata"]
            meta.pop("arguments_sha256")
            con.execute("UPDATE jobs SET metadata=? WHERE id=?", (json.dumps(meta), item.task))
    elif fault == "missing_result":
        (queue.files(item.task) / "mcp-result.json").unlink()
    elif fault == "parent_report":
        with queue.connection() as con:
            con.execute("UPDATE jobs SET metadata=? WHERE id=?", (json.dumps({"report_id": "f" * 32}), item.parent))
    elif fault == "snapshot":
        (item.folder / "input_snapshots" / (item.state["input_snapshot"]["sha256"] + ".json")).write_bytes(b"changed")
    elif fault == "input":
        (item.path.parent / "query.fasta").write_text(">query\nAAAAAAAAAA\n")
    elif fault == "unfinished":
        item.row["status"] = "unavailable"
    artifact = collect(service, item)[0]
    assert artifact["status"] == "unverified" and artifact["reason"]
    assert "content" not in artifact


@pytest.mark.parametrize("link_owner", ["alice", "bob"])
def test_two_real_reports_cannot_borrow_other_report_or_owner_receipts(reports, link_owner):
    service, _, create = reports
    item = create("alice")
    other = create(link_owner)
    # Copying the foreign task and forging its checkpoint owner/args is insufficient:
    # owner checks and original server-side task parameters are independent.
    foreign = copy.deepcopy(other.link)
    foreign["owner_sha256"] = owner_digest("alice")
    foreign["arguments"] = item.args
    foreign["arguments_sha256"] = value_digest(item.args)
    item.state["tool_receipts"]["prediction:" + item.row["id"]] = foreign
    assert collect(service, item)[0]["status"] == "unverified"
    with pytest.raises(FileNotFoundError):
        delivery.prediction_artifacts(service, "bob", public(service, item))
    assert collect(service, other)[0]["status"] == "included"


@pytest.mark.parametrize(
    "fault", ["file_symlink", "dir_symlink", "snapshot_symlink", "hardlink", "fifo", "missing", "oversize"]
)
def test_untrusted_filesystem_entries_are_not_followed(reports, tmp_path, fault, monkeypatch):
    service, _, create = reports
    item = create()
    outside = tmp_path / "outside.csv"
    outside.write_bytes(item.raw)
    if fault in {"file_symlink", "hardlink", "fifo", "missing"}:
        item.path.unlink()
    if fault == "file_symlink":
        item.path.symlink_to(outside)
    elif fault == "hardlink":
        os.link(outside, item.path)
    elif fault == "fifo":
        os.mkfifo(item.path)
    elif fault == "dir_symlink":
        moved = tmp_path / "outside-predictions"
        item.path.parent.rename(moved)
        item.path.parent.symlink_to(moved, target_is_directory=True)
    elif fault == "snapshot_symlink":
        path = item.folder / "input_snapshots" / (item.state["input_snapshot"]["sha256"] + ".json")
        outside_snapshot = tmp_path / "outside-snapshot.json"
        path.rename(outside_snapshot)
        path.symlink_to(outside_snapshot)
    elif fault == "oversize":
        monkeypatch.setattr(delivery, "MAX_CSV_BYTES", len(item.raw) - 1)
    artifact = collect(service, item)[0]
    assert artifact["status"] == "unverified" and "content" not in artifact
    assert outside.read_bytes() == item.raw


@pytest.mark.parametrize("mutation", ["checkpoint", "request", "ownership"])
def test_download_detects_checkpoint_and_owner_changes_during_receipt_read(reports, monkeypatch, mutation):
    service, _, create = reports
    item = create()
    result = public(service, item)
    original = scheduler.ToolJobs.get

    def change(jobs, identity, owner):
        value = original(jobs, identity, owner)
        if mutation == "checkpoint":
            write(item.folder / "state.json", {**item.state, "new_revision": True})
        else:
            request = {**item.request, "title": "Changed"}
            if mutation == "ownership":
                request["owner"] = "bob"
            write(item.folder / "request.json", request)
        return value

    monkeypatch.setattr(scheduler.ToolJobs, "get", change)
    if mutation == "ownership":
        with pytest.raises(FileNotFoundError):
            delivery.prediction_artifacts(service, item.owner, result)
    else:
        artifact = delivery.prediction_artifacts(service, item.owner, result)[0]
        assert artifact["status"] == "unverified" and artifact["reason"] == "checkpoint_changed"
        assert "content" not in artifact


def test_packaging_rechecks_copied_content_and_total_budget(reports, monkeypatch):
    service, _, create = reports
    item = create()
    result = public(service, item)
    artifacts = delivery.prediction_artifacts(service, item.owner, result)
    artifacts[0]["content"] += b"tamper"
    with pytest.raises(ValueError, match="changed"):
        build_dossier(result, prediction_files=artifacts)
    monkeypatch.setattr(delivery, "MAX_CSV_TOTAL_BYTES", len(item.raw) - 1)
    assert collect(service, item)[0]["reason"] == "artifact_total_limit"


def test_http_zip_with_two_owners_delivers_csv_only_to_report_owner(reports, monkeypatch):
    from fastapi import FastAPI, Request
    from fastapi.testclient import TestClient

    from backend.web_v2 import report_api as api

    service, _, create = reports
    alice, bob = create("alice"), create("bob")
    monkeypatch.setattr(api, "ResearchJobs", lambda: service)
    app = FastAPI()
    app.include_router(api.router)

    def owner(request: Request):
        return request.headers["x-test-owner"]

    app.dependency_overrides[api.owner] = owner
    with TestClient(app) as client:
        for item, other in [(alice, bob), (bob, alice)]:
            url = "/api/report/jobs/" + item.id + "/download?format=zip"
            assert client.get(url, headers={"x-test-owner": other.owner}).status_code == 404
            response = client.get(url, headers={"x-test-owner": item.owner})
            assert response.status_code == 200 and response.headers["cache-control"] == "private, no-store"
            with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
                assert archive.read("predictions/001.csv") == item.raw
                assert json.loads(archive.read("predictions/index.json"))[0]["artifact"]["status"] == "included"


def test_factory_receipt_records_original_csv_hash_even_when_preview_is_truncated(tmp_path):
    from mcp_tools.models.factory.prediction import fintuned_operations as operations

    path = tmp_path / "prediction.csv"
    raw = b"header,sequence,probabilities\r\nquery,MKWA," + b"1," * 1000 + b"\r\n"
    path.write_bytes(raw)
    result = json.loads(operations._download_success_response(str(path), "x" * 20000, {"sequence_count": 1}))
    assert result["file_info"]["sha256"] == hashlib.sha256(raw).hexdigest()
    assert result["file_info"]["file_size"] == len(raw)
    assert len(result["content_preview"]) < 20000
    path.write_bytes(b"later changed")
    assert result["file_info"]["sha256"] != hashlib.sha256(path.read_bytes()).hexdigest()


def test_manifest_explicitly_marks_missing_original_in_partial_package(reports):
    service, _, create = reports
    item = create()
    missing = copy.deepcopy(item.row)
    missing.update(id="b" * 16, status="unavailable")
    item.state["predictions"].append(missing)
    result = public(service, item)
    artifacts = delivery.prediction_artifacts(service, item.owner, result)
    with build_dossier(result, prediction_files=artifacts) as handle, zipfile.ZipFile(handle) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        assert [row["status"] for row in manifest["predictions"]] == ["included", "unverified"]
        assert manifest["predictions"][1]["reason"] == "prediction_not_completed"
        assert "predictions/001.csv" in archive.namelist() and "predictions/002.csv" not in archive.namelist()


@pytest.mark.parametrize(
    "tool,task,problem",
    [
        ("predict_protein_function", "Solubility", "multi_label_classification"),
        ("predict_residue_function", "Activity Site", "residue_single_label_classification"),
    ],
)
def test_real_factory_wrappers_bind_long_original_csv_not_truncated_preview(tmp_path, monkeypatch, tool, task, problem):
    from mcp_tools.models.factory.prediction import fintuned_operations as operations

    adapter = tmp_path / "adapter"
    adapter.mkdir()
    config = {"problem_type": problem, "normalize": "standard"}
    (adapter / "lr5e-4_bt12k_ga8.json").write_text(json.dumps(config))
    fasta = tmp_path / "query.fasta"
    sequence = "MKWA" * 200
    fasta.write_text(">query\n" + sequence + "\n")
    path = tmp_path / "prediction.csv"
    predicted = {"predicted_class": [1] * len(sequence), "probabilities": [[0.2, 0.8]] * len(sequence)}
    module = SimpleNamespace(
        parse_fasta=lambda _: [("query", sequence)],
        load_model_and_tokenizer=lambda args: (None, None, None, "cpu"),
        predict=lambda *args: predicted,
    )
    monkeypatch.setattr(
        operations, "_get_module", lambda _: (module, lambda *args: {"input_ids": [], "attention_mask": []})
    )
    result = json.loads(
        getattr(operations, tool).runtime_implementation(
            str(fasta), task, "ESM2-650M", str(adapter), output_file=str(path)
        )
    )
    assert result["status"] == "success"
    assert result["file_info"]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert result["file_info"]["file_size"] == path.stat().st_size
    assert result["biological_metadata"]["output_scale"]["value_space"] == "class_index_and_probability"
    with pytest.raises(json.JSONDecodeError):
        json.loads(result["content_preview"])
    with path.open(newline="") as handle:
        row = list(csv.DictReader(handle))[0]
    assert row["sequence"] == sequence and json.loads(row["probabilities"]) == predicted["probabilities"]


def test_old_bound_receipt_without_scale_remains_unknown_in_zip(reports):
    service, queue, create = reports
    item = create(worker_hash=False)
    item.row.pop("output_scale")
    item.value["biological_metadata"].pop("output_scale")
    scheduler.save_json(
        queue.files(item.task) / "mcp-result.json", {"value": json.dumps(item.value), "is_error": False}
    )
    result = public(service, item)
    artifacts = delivery.prediction_artifacts(service, item.owner, result)
    assert artifacts[0]["status"] == "included" and artifacts[0]["output_scale"]["value_space"] == "unknown"
    with build_dossier(result, prediction_files=artifacts) as handle, zipfile.ZipFile(handle) as archive:
        index = json.loads(archive.read("predictions/index.json"))[0]
        assert index["output_scale"]["unit"] == "unknown"
        assert index["artifact"]["validation"] == "complete_saved_values"


def test_http_rechecks_ownership_after_archive_build_and_closes_handle(reports, monkeypatch):
    from fastapi import FastAPI, Request
    from fastapi.testclient import TestClient

    from backend.web_v2 import report_api as api
    from backend.web_v2 import research_dossier as dossier

    service, _, create = reports
    item = create("alice")
    handles = []
    real = dossier.build_dossier

    def revoke(*args):
        handle = real(*args)
        handles.append(handle)
        write(item.folder / "request.json", {**item.request, "owner": "bob"})
        return handle

    monkeypatch.setattr(dossier, "build_dossier", revoke)
    monkeypatch.setattr(api, "ResearchJobs", lambda: service)
    app = FastAPI()
    app.include_router(api.router)

    def owner(request: Request):
        return "alice"

    app.dependency_overrides[api.owner] = owner
    with TestClient(app) as client:
        response = client.get("/api/report/jobs/" + item.id + "/download?format=zip")
    assert response.status_code == 404
    assert handles and handles[0].closed
