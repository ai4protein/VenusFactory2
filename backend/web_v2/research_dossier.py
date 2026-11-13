"""Portable Researcher dossiers built only from an owner-checked report."""
import hashlib
import json
import tempfile
import zipfile

MAX_ENTRY_BYTES = 16 * 1024 * 1024
MAX_TOTAL_BYTES = 64 * 1024 * 1024


def build_dossier(report, inputs=None, prediction_files=None):
    """Return a seekable archive. Never follow paths or fetch URLs in evidence."""
    if not report.get("report"):
        raise ValueError("The report is not ready yet")
    from .research_documents import report_view
    lang = inputs.get("lang", "en") if inputs else "en"
    report = report_view(report, lang)
    target = tempfile.SpooledTemporaryFile(max_size=2 * 1024 * 1024, mode="w+b")
    files = []
    total = 0
    try:
        with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            def add(name, content):
                nonlocal total
                raw = content.encode("utf-8") if isinstance(content, str) else content
                if len(raw) > MAX_ENTRY_BYTES or total + len(raw) > MAX_TOTAL_BYTES:
                    raise ValueError("Report evidence exceeds the download size limit")
                total += len(raw)
                archive.writestr(name, raw)
                files.append({"path": name, "bytes": len(raw),
                              "sha256": hashlib.sha256(raw).hexdigest()})

            def data(name, value):
                add(name, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))

            from .research_documents import assemble
            body = assemble(report, lang) if report.get("workflow") == "specialist_research_v2" else report["report"]
            add("report.md", body)
            from .research_documents import editor_audit, review_ready
            if report.get("review") and not review_ready(report, lang):
                add("drafts/editor.md", report["review"])
                data("drafts/editor-audit.json", editor_audit(report["review"], [s["id"] for s in report.get("sources", [])], lang))
            review_attempts = []
            for index, attempt in enumerate(report.get("review_attempts", []), 1):
                path = f"drafts/editor/attempt-{index:02d}.md"
                add(path, attempt.get("draft", ""))
                item = {k: v for k, v in attempt.items() if k not in {"draft", "raw_draft"}}
                item["path"] = path
                if "raw_draft" in attempt:
                    item["raw_path"] = f"drafts/editor/raw-{index:02d}.md"
                    add(item["raw_path"], attempt["raw_draft"])
                review_attempts.append(item)
            if review_attempts:
                data("drafts/editor/index.json", review_attempts)
            metadata = {k: report.get(k) for k in (
                "id", "job_number", "title", "created", "protein", "focus", "model_id",
                "status", "partial", "completed_chapters", "minimum_met", "analyses",
                "model_calls", "model_call_limit", "queries", "literature_queries", "input_snapshot", "evidence_review_required")}
            metadata["steps"] = [
                {k: step.get(k) for k in ("id", "status", "message")}
                for step in report.get("steps", [])
            ]
            data("overview.json", metadata)
            from .research_documents import chapter_ready, chapter_text
            index = []
            for i, chapter in enumerate(report.get("chapters", []), 1):
                name = f"chapters/{i:02d}.md"
                add(name, chapter_text(chapter, lang, report.get("sources", [])))
                draft_file = None
                if chapter.get("draft") and not chapter_ready(chapter, report.get("sources", [])):
                    draft_file = f"drafts/{i:02d}.md"
                    add(draft_file, chapter["draft"])
                attempts = []
                for attempt_number, attempt in enumerate(chapter.get("draft_history", []), 1):
                    path = f"drafts/chapters/{i:02d}/attempt-{attempt_number:02d}.md"
                    add(path, attempt.get("draft", ""))
                    attempts.append({k: v for k, v in attempt.items() if k != "draft"} | {"path": path})
                if attempts:
                    data(f"drafts/chapters/{i:02d}/index.json", attempts)
                index.append({**{k: chapter.get(k) for k in (
                    "id", "title", "status", "attempts", "audit", "source_ids",
                    "literature_query", "web_query", "retrieval", "source_review", "draft_status")}, "file": name,
                    "unaccepted_draft": draft_file})
            data("chapters/index.json", index)
            if report.get("source_review"):
                data("verification/source-review.json", report["source_review"])
                data("verification/claims.json", list(report["source_review"].get("units", {}).values()))
            if report.get("source_review_history"):
                data("verification/previous-reviews.json", report["source_review_history"])

            references = []
            for i, source in enumerate(report.get("sources", []), 1):
                item = {k: source.get(k) for k in (
                    "id", "title", "url", "kind", "service", "evidence_level",
                    "retrieved", "text", "extraction", "sha256", "query_tool", "query_arguments")}
                name = f"sources/{i:03d}.json"
                data(name, item)
                references.append({k: item[k] for k in (
                    "id", "title", "url", "kind", "evidence_level")} | {"file": name})
                if source.get("kind") == "user_material":
                    add(f"materials/{i:03d}.txt", str(source.get("text") or ""))
            data("references.json", references)
            for i, section in enumerate(report.get("sections", []), 1):
                data(f"analyses/{i:03d}.json",
                     {k: section.get(k) for k in ("id", "title", "data", "text")})
            prediction_index = []
            analysis_files = {section.get("id"): f"analyses/{i:03d}.json"
                              for i, section in enumerate(report.get("sections", []), 1)}
            artifacts = {row.get("id"): row for row in prediction_files or []}
            for number, prediction in enumerate(report.get("predictions", []), 1):
                item = {k: prediction.get(k) for k in (
                    "id", "task", "model", "dataset", "tool", "status", "message",
                    "sequence_sha256", "job_number", "started", "finished", "evidence_level", "output_scale", "classification")}
                item["analysis_file"] = analysis_files.get("prediction_" + str(prediction.get("id")))
                artifact = artifacts.get(prediction.get("id"), {})
                item["artifact"] = {"status": "unverified", "reason": artifact.get("reason", "missing_receipt_binding")}
                if artifact.get("status") == "included":
                    raw = artifact.get("content")
                    if (not isinstance(raw, bytes) or len(raw) != artifact.get("bytes")
                            or hashlib.sha256(raw).hexdigest() != artifact.get("sha256")
                            or (prediction.get("output_scale") is not None
                                and artifact.get("output_scale") != prediction["output_scale"])):
                        raise ValueError("Prediction artifact changed before packaging")
                    # Archive paths are generated here, never taken from a tool result.
                    path = f"predictions/{number:03d}.csv"
                    add(path, raw)
                    item["artifact"] = {key: artifact.get(key) for key in ("status", "bytes", "sha256", "validation", "output_scale")}
                    item["artifact"]["file"] = path
                    item["output_scale"] = artifact.get("output_scale")
                prediction_index.append(item)
            if prediction_index:
                data("predictions/index.json", prediction_index)
            for i, service in enumerate(report.get("remote_services", []), 1):
                data(f"services/{i:03d}.json", {k: service.get(k) for k in (
                    "id", "status", "summary", "count", "query", "parameters",
                    "evidence_level", "url", "results", "hits", "data")})
            inputs = inputs or {}
            if inputs.get("sequence"):
                add("inputs/protein.fasta", ">query\n" + str(inputs["sequence"]) + "\n")
            if inputs.get("structure_pdb"):
                add("inputs/structure.pdb", str(inputs["structure_pdb"]))
            if report.get("input_snapshot"):
                data("inputs/metadata.json", {**report["input_snapshot"],
                     **{k: inputs.get(k) for k in ("origin", "accession", "pdb_id", "structure_chain",
                                                  "source_chain", "recorded_at", "protein")}})
                if isinstance(inputs.get("entry"), dict):
                    data("inputs/reference-uniprot.json", inputs["entry"])
            status = "partial" if report.get("partial") else str(report.get("status", "unknown"))
            add("README.md", "# Protein research dossier\n\n"
                "Start with [report.md](report.md).\n\n"
                f"Report status: {status}.\n\n"
                "- overview.json: target, scope, model, collection queries and step outcomes.\n"
                "- chapters/: completed chapters, revision placeholders and citation checks.\n"
                "- drafts/: unaccepted outputs retained for inspection, excluded from the report.\n"
                "- references.json and sources/: references and evidence snapshots.\n"
                "- verification/: independent claim assessments, quoted passages and original-source locations, when available.\n"
                "- analyses/, predictions/ and services/: calculation results, verified CSVs, prediction status and service receipts.\n"
                "- predictions/index.json: each CSV is included or unverified, with a reason and output scale. "
                "worker_sha256 matches the original worker receipt byte for byte; complete_saved_values "
                "matches a complete legacy preview semantically, not its original byte representation.\n"
                "- materials/: extracted text with available page, sheet or row markers.\n"
                "- inputs/: saved computation inputs and reference metadata, when present.\n"
                "- manifest.json: file sizes and SHA-256 checksums.\n\n"
                "Sources retain collected excerpts and their extraction limits. "
                "Original attachments remain available in the conversation. "
                "Chapter completion checks do not establish biological validity.\n")
            manifest = {"schema": "venusfactory.research-dossier.v1", "report_id": report.get("id"),
                        "partial": bool(report.get("partial")), "files": files,
                        "predictions": [{"id": item["id"], **item["artifact"]} for item in prediction_index]}
            archive.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
        target.seek(0)
        return target
    except BaseException:
        target.close()
        raise


def archive_chunks(handle):
    try:
        while chunk := handle.read(128 * 1024):
            yield chunk
    finally:
        handle.close()
