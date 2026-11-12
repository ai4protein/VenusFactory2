"""Reconcile linked, successful MCP receipts into a resumed report's evidence."""

import copy
import csv
import hashlib
import json
import time
from pathlib import Path

from .research_receipts import ReceiptUnavailable, recover_result


def invalidate_chapter(state, chapter, reason, *, archive_shared=True):
    """Keep the old work, but never reuse its acceptance for changed evidence."""
    from .research_documents import remember_chapter

    if chapter.get("draft"):
        remember_chapter(chapter, chapter["draft"], chapter.get("audit", {}))
    chapter.setdefault("evidence_history", []).append(
        {
            key: copy.deepcopy(chapter[key])
            for key in ("source_ids", "source_review", "evidence_sha256", "retrieval")
            if key in chapter
        }
        | {"reason": reason, "recorded": time.time()}
    )
    chapter.update(status="pending", draft_status="pending", source_ids=[])
    chapter.pop("source_review", None)
    # review_documents uses a cache of individual assessments. Remove only the
    # affected document's entries; even an unchanged claim needs new review.
    review = state.get("source_review")
    if review:
        if archive_shared:
            state.setdefault("source_review_history", []).append(copy.deepcopy(review))
        review.get("documents", {}).pop(chapter["id"], None)
        review["units"] = {
            key: row
            for key, row in review.get("units", {}).items()
            if row.get("document_id") not in {chapter["id"], "editor"}
        }
        review.get("documents", {}).pop("editor", None)
        review["status"] = "pending"
    if state.get("review"):
        state.setdefault("recovered_editor_history", []).append(
            {
                key: copy.deepcopy(state[key])
                for key in ("review", "review_status", "review_source_review")
                if key in state
            }
        )
    state.pop("review", None)
    state.pop("review_source_review", None)
    state["review_status"] = "pending"


def prediction_evidence(row):
    value = {key: row[key] for key in ("task", "model", "dataset", "sequence_sha256", "values", "evidence_level")}
    section = {"id": "prediction_" + row["id"], "title": row["task"] + " · " + row["model"], "data": value}
    source = {
        "title": row["task"] + " · " + row["model"] + " · " + row["dataset"],
        "url": "",
        "kind": "prediction",
        "task": row["task"],
        "prediction_id": row["id"],
        "evidence_level": "model_prediction",
        "text": json.dumps(value),
    }
    return section, source


def prediction_arguments(row, link, folder, sequence):
    arguments = link.get("arguments")
    if not isinstance(arguments, dict):
        raise ReceiptUnavailable("missing_arguments")
    expected = {
        "fasta_file": str(folder / "query.fasta"),
        "output_file": str(folder / (row["id"] + ".csv")),
        "task": row["task"],
        "model_name": row["model"],
        "adapter_path": arguments.get("adapter_path"),
    }
    if (
        arguments != expected
        or row.get("tool") not in {"predict_protein_function", "predict_residue_function"}
        or not isinstance(arguments.get("adapter_path"), str)
        or row.get("sequence_sha256") != hashlib.sha256(sequence.encode()).hexdigest()
    ):
        raise ReceiptUnavailable("prediction_input_mismatch")
    fasta = folder / "query.fasta"
    if fasta.is_symlink() or not fasta.is_file() or fasta.stat().st_size > 20_000:
        raise ReceiptUnavailable("prediction_input_changed")
    current = "".join(line.strip() for line in fasta.read_text().splitlines() if not line.startswith(">"))
    if current != sequence:
        raise ReceiptUnavailable("prediction_input_changed")
    return arguments


def prediction_values(path, sequence, result, row, arguments):
    """Match the complete saved numeric row, not merely its sequence or filename.

    This is semantic value verification, not a file-byte checksum. No numeric
    tolerance is used; older truncated or differently rounded previews stay
    unavailable rather than silently accepting unrecorded values.
    """
    from mcp_tools.scheduler.receipt_binding import value_digest

    if path.is_symlink() or not path.is_file() or path.stat().st_size > 2_000_000:
        raise ReceiptUnavailable("prediction_output_unavailable")
    metadata = result.get("biological_metadata", {})
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
    info = result.get("file_info", {})
    if (
        not isinstance(info, dict)
        or info.get("file_name") != path.name
        or info.get("format") != "csv"
        or type(info.get("file_size")) is not int
        or info["file_size"] != path.stat().st_size
    ):
        raise ReceiptUnavailable("prediction_file_mismatch")
    try:
        preview = json.loads(result["content_preview"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ReceiptUnavailable("prediction_preview_incomplete") from exc
    if (
        not isinstance(preview, list)
        or len(preview) != 1
        or not isinstance(preview[0], dict)
        or preview[0].get("sequence") != sequence
    ):
        raise ReceiptUnavailable("prediction_preview_incomplete")
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    def scalar(value):
        try:
            return json.loads(value)
        except (ValueError, TypeError):
            return value

    actual = [{key: scalar(value) for key, value in item.items()} for item in rows]
    if value_digest(actual) != value_digest(preview):
        raise ReceiptUnavailable("prediction_output_mismatch")
    # Use the server receipt's saved values after validating the complete CSV.
    return [{key: value for key, value in preview[0].items() if key not in {"header", "sequence"}}]


async def recover_evidence(state, publish, add, *, service_cache=None, input_store=None):
    """Return recovered topic IDs. No submit, quota reservation or model call."""
    from mcp_tools.scheduler.receipt_binding import value_digest

    from .research_input_snapshot import load_inputs
    from .research_report import literature_arguments, literature_records

    links = state.get("tool_receipts", {})
    recovered_topics = set()
    if not links:
        return recovered_topics
    snapshot = None
    for row in state.get("predictions", []):
        if row.get("status") == "completed" and any(
            s.get("id") == "prediction_" + row["id"] for s in state.get("sections", [])
        ):
            continue
        link = links.get("prediction:" + row["id"])
        if not link:
            continue  # Legacy job numbers alone cannot prove original arguments.
        try:
            if input_store is None or not state.get("input_snapshot") or service_cache is None:
                raise ReceiptUnavailable("missing_input_snapshot")
            if snapshot is None:
                snapshot = load_inputs(input_store, state["input_snapshot"])
            folder = Path(service_cache).parent / "predictions"
            arguments = prediction_arguments(row, link, folder, snapshot["sequence"])
            value = await recover_result(link, row["tool"], arguments)
            if isinstance(value, str):
                value = json.loads(value)
            if not isinstance(value, dict) or value.get("status") != "success":
                raise ReceiptUnavailable("prediction_not_successful")
            output = folder / (row["id"] + ".csv")
            values = prediction_values(output, snapshot["sequence"], value, row, arguments)
            completed = {
                **row,
                "status": "completed",
                "values": values,
                "output_file": output.name,
                "evidence_level": "model_prediction",
            }
            section, source = prediction_evidence(completed)
            inserted = add([source])
            if not inserted:
                raise ReceiptUnavailable("evidence_capacity")
            state.setdefault("sections", []).append(section)
            row.update(
                completed,
                recovery={"status": "completed", "task_id": link["task_id"], "validation": "complete_saved_values"},
            )
            row.pop("message", None)
            link.update(consumed=True, recovered_at=time.time())
        except Exception as exc:
            row["recovery"] = {
                "status": "unavailable",
                "code": exc.code if isinstance(exc, ReceiptUnavailable) else "invalid_prediction_receipt",
            }
        publish()
    predictions = state.get("predictions", [])
    if predictions:
        completed = sum(row.get("status") == "completed" for row in predictions)
        for step in state.get("steps", []):
            if step["id"] == "ai_predictions":
                step.update(
                    status="completed" if completed == len(predictions) else "partial" if completed else "empty",
                    message=f"{completed}/{len(predictions)}",
                )
        if "collection_nonprediction_partial" in state:
            state["collection_partial"] = bool(state["collection_nonprediction_partial"]) or completed != len(
                predictions
            )
    for chapter in state.get("chapters", []):
        retrieval = next((item for item in chapter.get("retrieval", []) if item.get("tool") == "literature"), None)
        if retrieval and retrieval.get("status") == "completed":
            continue
        link = links.get("chapter:" + chapter["id"] + ":literature")
        if not link:
            continue
        try:
            query = chapter.get("literature_query")
            if not isinstance(query, str) or not query:
                raise ReceiptUnavailable("missing_query")
            arguments = literature_arguments(query)
            value = await recover_result(link, "query_europepmc", arguments)
            sources = literature_records(value)
            inserted = add(sources)
            if sources and not inserted:
                raise ReceiptUnavailable("evidence_capacity")
            chapter["topic_source_ids"] = list(
                dict.fromkeys(chapter.get("topic_source_ids", []) + [s["id"] for s in inserted])
            )
            if retrieval is None:
                retrieval = {"tool": "literature"}
                chapter.setdefault("retrieval", []).append(retrieval)
            retrieval.update(status="completed", recovery="stored_receipt")
            link.update(consumed=True, result_sha256=value_digest(value), recovered_at=time.time())
            if inserted:
                recovered_topics.add(chapter["id"])
        except Exception as exc:
            chapter["retrieval_recovery"] = {
                "status": "unavailable",
                "code": exc.code if isinstance(exc, ReceiptUnavailable) else "invalid_literature_receipt",
            }
        publish()
    return recovered_topics
