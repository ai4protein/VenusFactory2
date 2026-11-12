"""Copy only owned, task-bound prediction bytes into a portable report dossier."""

import csv
import hashlib
import io
import json
import os
import re
import stat
import tempfile
from contextlib import contextmanager
from pathlib import Path

from mcp_tools.models.factory.prediction.output_scale import result_output_scale
from mcp_tools.scheduler.jobs import ToolJobs
from mcp_tools.scheduler.receipt_binding import value_digest

from .research_receipts import ReceiptUnavailable, validate_job, validate_link
from .research_recovery import prediction_values

MAX_CSV_BYTES = 2_000_000
MAX_CSV_TOTAL_BYTES = 32_000_000
MAX_PREDICTIONS = 32


@contextmanager
def directory(name, parent=None):
    fd = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
    try:
        yield fd
    finally:
        os.close(fd)


def read_at(parent, name, limit):
    if "/" in name or "\\" in name or name in {".", ".."}:
        raise ReceiptUnavailable("invalid_filename")
    fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
    with os.fdopen(fd, "rb") as handle:
        before = os.fstat(handle.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_size > limit:
            raise ReceiptUnavailable("invalid_file")
        raw = handle.read(limit + 1)
        after = os.fstat(handle.fileno())
    if len(raw) > limit or (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ):
        raise ReceiptUnavailable("file_changed")
    return raw


def _inputs(report_fd, reference):
    if not isinstance(reference, dict) or reference.get("schema") != "venusfactory.research-input.v1":
        raise ReceiptUnavailable("missing_input_snapshot")
    digest = reference.get("sha256", "")
    if not isinstance(digest, str) or not re.fullmatch("[a-f0-9]{64}", digest):
        raise ReceiptUnavailable("invalid_input_snapshot")
    with directory("input_snapshots", report_fd) as folder:
        raw = read_at(folder, digest + ".json", 4_000_000)
    if len(raw) != reference.get("bytes") or hashlib.sha256(raw).hexdigest() != digest:
        raise ReceiptUnavailable("input_snapshot_changed")
    return json.loads(raw)


def _scalar(value):
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except ValueError:
        return value


def _values(raw, sequence):
    rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8"), newline="")))
    if len(rows) != 1 or rows[0].get("sequence") != sequence:
        raise ReceiptUnavailable("prediction_input_mismatch")
    return [{key: _scalar(value) for key, value in rows[0].items() if key not in {"header", "sequence"}}]


def _validated_bytes(raw, row, value, arguments, sequence, scratch):
    metadata = value.get("biological_metadata", {})
    expected = {
        "task": row["task"],
        "dataset_name": row["dataset"],
        "model_name": row["model"],
        "fasta_file": arguments["fasta_file"],
        "adapter_path": arguments["adapter_path"],
    }
    if (
        not isinstance(metadata, dict)
        or any(metadata.get(key) != val for key, val in expected.items())
        or type(metadata.get("sequence_count")) is not int
        or metadata["sequence_count"] != 1
    ):
        raise ReceiptUnavailable("prediction_metadata_mismatch")
    info = value.get("file_info", {})
    filename = row["id"] + ".csv"
    if (
        not isinstance(info, dict)
        or info.get("file_name") != filename
        or info.get("format") != "csv"
        or type(info.get("file_size")) is not int
        or info["file_size"] != len(raw)
    ):
        raise ReceiptUnavailable("prediction_file_mismatch")
    actual_hash = hashlib.sha256(raw).hexdigest()
    recorded_hash = info.get("sha256")
    if recorded_hash is not None:
        if (
            not isinstance(recorded_hash, str)
            or not re.fullmatch("[a-f0-9]{64}", recorded_hash)
            or actual_hash != recorded_hash
        ):
            raise ReceiptUnavailable("prediction_hash_mismatch")
        checked = _values(raw, sequence)
        method = "worker_sha256"
    else:
        # Older strongly bound tasks may have a complete saved preview. Validate
        # an isolated copy of these exact bytes; never reopen the user's path.
        path = scratch / filename
        path.write_bytes(raw)
        checked = prediction_values(path, sequence, value, row, arguments)
        method = "complete_saved_values"
    expected_values = [{key: _scalar(value) for key, value in item.items()} for item in row.get("values", [])]
    if value_digest(checked) != value_digest(expected_values):
        raise ReceiptUnavailable("report_values_changed")
    scale = result_output_scale(value, row.get("output_scale"))
    return {"content": raw, "sha256": actual_hash, "bytes": len(raw), "output_scale": scale, "validation": method}


def prediction_artifacts(service, owner, report):
    """Read only: no model invocation, submission, quota reservation or mutation."""
    identity, _ = service.owned(owner, report["id"])
    rows = report.get("predictions", [])
    if len(rows) > MAX_PREDICTIONS:
        raise ValueError("Too many prediction artifacts")
    outcomes = [{"id": row.get("id"), "status": "unverified", "reason": "missing_receipt_binding"} for row in rows]
    if not rows:
        return outcomes
    try:
        with directory(str(service.root)) as root, directory(identity, root) as folder:
            request_raw = read_at(folder, "request.json", 4_000_000)
            request = json.loads(request_raw)
            if request.get("owner") != owner or request.get("id") != identity:
                raise ReceiptUnavailable("report_owner_changed")
            state_raw = read_at(folder, "state.json", 32_000_000)
            state = json.loads(state_raw)
            if state.get("input_snapshot") != report.get("input_snapshot"):
                raise ReceiptUnavailable("checkpoint_changed")
            inputs = _inputs(folder, state.get("input_snapshot"))
            sequence = inputs.get("sequence")
            if not isinstance(sequence, str) or not sequence:
                raise ReceiptUnavailable("missing_sequence")
            saved = {row["id"]: row for row in state.get("predictions", [])}
            links = state.get("tool_receipts", {})
            jobs = ToolJobs(service.queue)
            total = 0
            seen = set()
            with (
                directory("predictions", folder) as prediction_dir,
                tempfile.TemporaryDirectory(prefix="venus-dossier-") as temp,
            ):
                scratch = Path(temp)
                for row, outcome in zip(rows, outcomes):
                    try:
                        key = row.get("id")
                        if not isinstance(key, str) or not re.fullmatch("[a-f0-9]{16}", key) or key in seen:
                            raise ReceiptUnavailable("invalid_prediction_identity")
                        seen.add(key)
                        if row.get("status") != "completed":
                            raise ReceiptUnavailable("prediction_not_completed")
                        if saved.get(key) != row:
                            raise ReceiptUnavailable("checkpoint_changed")
                        link = links.get("prediction:" + key)
                        if not isinstance(link, dict):
                            raise ReceiptUnavailable("missing_receipt_binding")
                        arguments = link.get("arguments")
                        base = service.folder(identity) / "predictions"
                        if (
                            not isinstance(arguments, dict)
                            or row.get("tool") not in {"predict_protein_function", "predict_residue_function"}
                            or arguments
                            != {
                                "fasta_file": str(base / "query.fasta"),
                                "output_file": str(base / (key + ".csv")),
                                "task": row["task"],
                                "model_name": row["model"],
                                "adapter_path": arguments.get("adapter_path"),
                            }
                            or not isinstance(arguments.get("adapter_path"), str)
                            or row.get("sequence_sha256") != hashlib.sha256(sequence.encode()).hexdigest()
                        ):
                            raise ReceiptUnavailable("prediction_input_mismatch")
                        validate_link(link, row["tool"], arguments, owner)
                        job = jobs.get(link["task_id"], owner)
                        validate_job(link, job)
                        parent = service.queue.get(link["parent_job"], owner)
                        if parent["kind"] != "protein-research" or parent["metadata"].get("report_id") != identity:
                            raise ReceiptUnavailable("parent_report_mismatch")
                        result = job.get("result", {})
                        if job.get("status") != "completed" or result.get("is_error") or "value" not in result:
                            raise ReceiptUnavailable("result_not_successful")
                        value = result["value"]
                        if link.get("result_sha256") and value_digest(value) != link["result_sha256"]:
                            raise ReceiptUnavailable("receipt_changed")
                        value = json.loads(value) if isinstance(value, str) else value
                        if not isinstance(value, dict) or value.get("status") != "success":
                            raise ReceiptUnavailable("prediction_not_successful")
                        fasta = read_at(prediction_dir, "query.fasta", 20_000).decode()
                        if "".join(line.strip() for line in fasta.splitlines() if not line.startswith(">")) != sequence:
                            raise ReceiptUnavailable("prediction_input_changed")
                        raw = read_at(prediction_dir, key + ".csv", MAX_CSV_BYTES)
                        checked = _validated_bytes(raw, row, value, arguments, sequence, scratch)
                        if total + len(raw) > MAX_CSV_TOTAL_BYTES:
                            raise ReceiptUnavailable("artifact_total_limit")
                        total += len(raw)
                        outcome.update(checked, status="included", archive_path="predictions/" + key + ".csv")
                        outcome.pop("reason", None)
                    except Exception as exc:
                        outcome["reason"] = exc.code if isinstance(exc, ReceiptUnavailable) else "artifact_unavailable"
            # Do not mix a prior response with a checkpoint replaced during download.
            if (
                read_at(folder, "state.json", 32_000_000) != state_raw
                or read_at(folder, "request.json", 4_000_000) != request_raw
            ):
                raise ReceiptUnavailable("checkpoint_changed")
    except Exception as exc:
        reason = exc.code if isinstance(exc, ReceiptUnavailable) else "artifact_unavailable"
        outcomes = [{"id": row.get("id"), "status": "unverified", "reason": reason} for row in rows]
    # Ownership/online visibility can be revoked during collection. A stale
    # report must not be delivered, even when all artifacts were rejected.
    service.owned(owner, identity)
    return outcomes
