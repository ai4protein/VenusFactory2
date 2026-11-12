"""Bounded host facts for scientific acceptance, not a scientific verdict.

Only caller-supplied private receipts and registered inputs are authoritative.
An exact sequence match proves byte identity after documented FASTA parsing;
an unmatched sequence can be a legitimate design and is never rejected here.
"""
from __future__ import annotations

import hashlib
import json
import re

from utils.scoped_files import parts, read_bytes

from .research_analysis_evidence import JOB, SHA, validate_evidence
from .session_boundary import public_value

SCHEMA = "research-analysis-provenance.v1"
MAX_INPUT_BYTES = 16 * 1024**2
MAX_ARTIFACT_BYTES = 16 * 1024**2
MAX_OUTPUT_BYTES = 16 * 1024**2
MAX_OUTPUT_FILES = 8
MAX_MATERIALS = 32
MAX_SEQUENCES = 64
MAX_COMPUTATIONS = 12
MAX_FIELDS = 64
MAX_VALUE_CHARS = 800
MAX_INDEX_CHARS = 24000
MAX_MATERIAL_CHARS = 6500
MAX_REPORT_CHARS = 3500
MAX_VISITS = 4096
MAX_DEPTH = 16
FASTA_SUFFIXES = (".fasta", ".fa", ".faa", ".fna", ".fas")
SEQUENCE = re.compile(r"[A-Za-z*.-]+")
SEQUENCE_KEYS = frozenset({"sequence", "sequences", "protein_sequence", "aa_sequence"})
SENSITIVE_KEYS = frozenset({"api_key", "apikey", "authorization", "password", "access_token",
                            "refresh_token", "service_token", "secret", "token", "credentials",
                            "credential", "private_key"})


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _json(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def _pointer(path, key):
    return path + "/" + str(key).replace("~", "~0").replace("/", "~1")


def _load(raw):
    def invalid_constant(value):
        raise ValueError("Non-finite JSON")
    return json.loads(raw, parse_constant=invalid_constant)


def _projection(value, path="", units=None, *, max_fields=None, omit_previews=False):
    """Flatten recorded fields; every limit explicitly marks the view incomplete."""
    fields, omissions = [], []
    units = units if isinstance(units, dict) else {}
    limit = MAX_FIELDS if max_fields is None else max_fields

    def visit(item, pointer, depth, key=""):
        if omit_previews and key.lower() == "content_preview":
            omissions.append({"reason": "preview_not_measurement", "pointer": pointer})
            return True
        if key.lower() in SENSITIVE_KEYS:
            omissions.append({"reason": "sensitive_field_redacted", "pointer": pointer})
            return True
        if len(fields) >= limit:
            omissions.append({"reason": "field_limit", "pointer": pointer})
            return False
        if depth > MAX_DEPTH:
            omissions.append({"reason": "depth_limit", "pointer": pointer})
            return True
        if isinstance(item, (dict, list)) and item:
            children = item.items() if isinstance(item, dict) else enumerate(item)
            for child_key, child in children:
                if not visit(child, _pointer(pointer, child_key), depth + 1, str(child_key)):
                    return False
            return True
        text = _json(item)
        row = {"pointer": pointer, "value": item}
        if len(text) > MAX_VALUE_CHARS:
            row = {"pointer": pointer, "value_prefix": text[:MAX_VALUE_CHARS],
                   "value_sha256": _sha(text.encode()), "truncated": True,
                   "serialized_characters": len(text)}
            omissions.append({"reason": "value_limit", "pointer": pointer})
        unit = units.get(pointer, units.get(key))
        row["unit"] = unit if isinstance(unit, str) and len(unit) <= 120 else None
        row["unit_status"] = "reported" if row["unit"] is not None else "unknown"
        fields.append(row)
        return True

    visit(value, path, 0)
    return {"fields": fields, "complete": not omissions, "omissions": omissions}


def _fasta(raw):
    text = raw.decode("utf-8-sig")
    records, header, sequence = [], None, []

    def append():
        if header is None or not sequence:
            raise ValueError("Incomplete FASTA record")
        value = "".join(sequence).upper()
        if not SEQUENCE.fullmatch(value):
            raise ValueError("Unsupported FASTA characters")
        records.append({"record_index": len(records), "id": header.split()[0][:200],
                        "length": len(value), "sequence_sha256": _sha(value.encode())})

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                append()
                if len(records) == MAX_SEQUENCES:
                    return records, False
            header, sequence = line[1:].strip(), []
            if not header:
                raise ValueError("Missing FASTA identifier")
        elif header is None:
            raise ValueError("Missing FASTA header")
        else:
            sequence.append("".join(line.split()))
    append()
    return records, True


def _materials(workspace, inputs, source_reader):
    entries, omissions = [], []
    if not isinstance(inputs, list):
        return {"status": "unknown", "records": [], "omissions": [{"reason": "invalid_input_registry"}]}
    if len(inputs) > MAX_MATERIALS:
        omissions.append({"reason": "material_limit", "count": len(inputs) - MAX_MATERIALS})
    def add(row, number):
        if len(_json([*entries, row])) > MAX_MATERIAL_CHARS:
            omissions.append({"reason": "material_summary_limit", "remaining_inputs": len(inputs) - number})
            return False
        entries.append(row)
        return True

    for number, item in enumerate(inputs[:MAX_MATERIALS]):
        row = {"material_index": number, "status": "unknown"}
        try:
            if not isinstance(item, dict):
                raise ValueError("Invalid input record")
            name, expected = item["path"], item["sha256"]
            parts(name)
            if not isinstance(expected, str) or not SHA.fullmatch(expected):
                raise ValueError("Invalid input hash")
            row.update(path=name, sha256=expected, origin="registered_input")
            size = item.get("size", item.get("bytes"))
            if type(size) is int and size > MAX_INPUT_BYTES:
                row["reason"] = "input_too_large"
                if not add(row, number):
                    break
                continue
            raw = source_reader(name) if source_reader is not None else read_bytes(
                workspace, name, max_bytes=MAX_INPUT_BYTES)
            if not isinstance(raw, bytes) or len(raw) > MAX_INPUT_BYTES:
                row["reason"] = "input_too_large_or_invalid"
            elif _sha(raw) != expected:
                row["reason"] = "original_hash_mismatch"
            else:
                row.update(status="verified", bytes=len(raw))
                if name.lower().endswith(FASTA_SUFFIXES):
                    try:
                        sequences, complete = _fasta(raw)
                        row.update(sequences=sequences, sequences_complete=complete,
                                   parsing="fasta.v1: strip sequence whitespace; uppercase; preserve *.-")
                        if not complete:
                            row["omissions"] = [{"reason": "sequence_limit"}]
                    except (UnicodeError, ValueError):
                        row.update(sequence_status="unknown", reason="fasta_parse_failed")
                else:
                    row["content"] = "not_exposed"
                    row["sequence_status"] = "not_parsed"
                    # In particular, protected table labels never enter this packet.
        except (OSError, ValueError, TypeError, KeyError):
            row["reason"] = "registered_input_unavailable"
        if not add(row, number):
            break
    return {"status": "not_registered" if not inputs else "registered",
            "records": entries, "omissions": omissions}


def _sequence_relations(arguments, materials):
    known = [(item["material_index"], sequence)
             for item in materials["records"] if item["status"] == "verified"
             for sequence in item.get("sequences", [])]
    sources_complete = (
        bool(materials["records"]) and not materials["omissions"]
        and all(item["status"] == "verified" and item.get("sequences_complete") is True
                for item in materials["records"]))
    result, omissions, visits = [], [], 0

    def visit(value, pointer="", key="", depth=0):
        nonlocal visits
        visits += 1
        if key.lower() in SENSITIVE_KEYS:
            return True
        if depth > MAX_DEPTH or len(result) >= MAX_SEQUENCES or visits > MAX_VISITS:
            omissions.append({"reason": "sequence_argument_limit", "pointer": pointer})
            return False
        if isinstance(value, dict):
            for name, child in value.items():
                if not visit(child, _pointer(pointer, name), str(name), depth + 1):
                    return False
        elif isinstance(value, list):
            for i, child in enumerate(value):
                if not visit(child, _pointer(pointer, i), key, depth + 1):
                    return False
        elif key.lower() in SEQUENCE_KEYS and isinstance(value, str):
            # The actual argument is not normalized: only exact matches are asserted.
            sha = _sha(value.encode())
            matches = [{"material_index": i, "record_index": seq["record_index"], "id": seq["id"]}
                       for i, seq in known if seq["sequence_sha256"] == sha]
            result.append({"argument_pointer": pointer, "length": len(value),
                           "sequence_sha256": sha, "matches": matches,
                           "relation": "exact_match" if matches else ("unmatched" if known else "unknown"),
                           "comparison_complete": sources_complete,
                           "interpretation": "Identity evidence only; unmatched may be a valid derived or novel sequence."})
        return True

    visit(arguments)
    return {"records": result, "omissions": omissions}


def _verified_document(workspace, row):
    saved = row["result"]
    expected_keys = {"version", "task_id", "parent_job", "session_id", "owner_sha256", "evidence", "artifact"}
    if (set(saved) != expected_keys or type(saved["version"]) is not int or saved["version"] != 1
            or not isinstance(saved["task_id"], str) or not JOB.fullmatch(saved["task_id"])
            or not saved["parent_job"] or not saved["session_id"]
            or not isinstance(saved["owner_sha256"], str) or not SHA.fullmatch(saved["owner_sha256"])):
        raise ValueError("Invalid private receipt")
    artifact = saved["artifact"]
    if (set(artifact) != {"path", "sha256"}
            or artifact["path"] != "analysis/" + saved["task_id"] + ".json"):
        raise ValueError("Invalid analysis artifact")
    raw = read_bytes(workspace, artifact["path"], max_bytes=MAX_ARTIFACT_BYTES)
    if _sha(raw) != artifact["sha256"]:
        raise ValueError("Analysis artifact changed")
    validate_evidence(saved["evidence"], workspace)
    document = _load(raw)
    evidence = saved["evidence"]
    if (not isinstance(document, dict) or document.get("tool") != evidence["tool"]
            or document.get("category") != evidence["category"]
            or document.get("input_files") != evidence["input_files"]
            or document.get("output_files") != evidence["output_files"]
            or not isinstance(document.get("arguments"), dict) or "result" not in document):
        raise ValueError("Analysis artifact does not match private receipt")
    return saved, document


def _output_results(workspace, document, protected_hashes, byte_budget, field_limit):
    """Read only host-bound JSON outputs; previews and copied inputs are not results."""
    fields, records, omissions = [], [], []
    outputs = document["output_files"]
    protected = protected_hashes | set(document["input_files"].values())

    def invalid_constant(value):
        raise ValueError("Non-finite JSON")

    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Ambiguous output JSON")
            result[key] = value
        return result

    for number, (name, expected) in enumerate(outputs.items()):
        if number == MAX_OUTPUT_FILES:
            omissions.append({"reason": "output_file_limit", "count": len(outputs) - number})
            break
        pointer = _pointer("/output_files", name)
        record = {"path": name, "sha256": expected, "pointer": pointer, "status": "unknown"}
        reason = None
        if expected in protected:
            reason = "input_copy_not_exposed"
        elif not name.lower().endswith(".json"):
            reason = "unsupported_output_format"
        elif field_limit <= len(fields):
            reason = "field_limit"
        elif byte_budget[0] <= 0:
            reason = "output_byte_limit"
        else:
            try:
                raw = read_bytes(workspace, name, max_bytes=byte_budget[0])
                byte_budget[0] -= len(raw)
                if _sha(raw) != expected:
                    raise ValueError("Output changed")
                # A duplicate key must not silently select one of two measurements.
                value = json.loads(raw, object_pairs_hook=unique_object,
                                   parse_constant=invalid_constant)
                units = value.get("units", {}) if isinstance(value, dict) else {}
                projection = _projection(value, pointer, units, max_fields=field_limit - len(fields),
                                         omit_previews=True)
                fields.extend(projection["fields"])
                omissions.extend(projection["omissions"])
                record.update(status="verified", bytes=len(raw), complete=projection["complete"])
            except (OSError, ValueError, TypeError, RecursionError):
                reason = "output_unavailable_invalid_or_too_large"
        if reason:
            record["reason"] = reason
            omissions.append({"reason": reason, "pointer": pointer})
        records.append(record)
    return {"fields": fields, "records": records, "omissions": omissions}


def _computation(workspace, row, materials, protected_hashes, output_budget):
    saved, document = _verified_document(workspace, row)
    value = document["result"]
    decoded = False
    if isinstance(value, str):
        try:
            value, decoded = _load(value), True
        except (ValueError, TypeError, RecursionError):
            pass
    path = "/decoded_result" if decoded else "/result"
    units = value.get("units", {}) if isinstance(value, dict) else {}
    envelope_fields = list(value) if isinstance(value, dict) else []
    if isinstance(value, dict) and "data" in value:
        value, path = value["data"], path + "/data"
    result = _projection(value, path, units, omit_previews=True)
    outputs = _output_results(workspace, document, protected_hashes, output_budget,
                              MAX_FIELDS - len(result["fields"]))
    result["fields"].extend(outputs["fields"])
    result["omissions"].extend(outputs["omissions"])
    result["complete"] = not result["omissions"]
    return {
        "status": "verified", "task_id": saved["task_id"], "tool": document["tool"],
        "category": document["category"], "artifact": saved["artifact"],
        "arguments_sha256": saved["evidence"]["arguments_sha256"],
        "result_sha256": saved["evidence"]["result_sha256"],
        "arguments": _projection(document["arguments"]),
        "input_files": document["input_files"], "output_files": document["output_files"],
        "result_root": path, "result_envelope_fields": envelope_fields[:MAX_FIELDS],
        "result_envelope_fields_omitted": max(0, len(envelope_fields) - MAX_FIELDS),
        "result_fields": result["fields"], "result_complete": result["complete"],
        "result_omissions": result["omissions"],
        "output_results": {"records": outputs["records"], "omissions": outputs["omissions"]},
        "sequence_relations": _sequence_relations(document["arguments"], materials),
        "boundary": "Arguments/results are the host-captured public representation; paths and secrets may be redacted.",
    }


def _reports(workspace, checkpoint):
    records, omissions = [], []
    artifacts = checkpoint.get("artifacts", []) if isinstance(checkpoint, dict) else []
    if not isinstance(artifacts, list):
        return {"records": [], "omissions": [{"reason": "invalid_checkpoint"}]}
    for item in artifacts[:MAX_MATERIALS]:
        row = {"status": "unknown"}
        try:
            name, expected = item["path"], item["sha256"]
            parts(name)
            if not isinstance(expected, str) or not SHA.fullmatch(expected):
                raise ValueError("Invalid report hash")
            row.update(path=name, sha256=expected)
            raw = read_bytes(workspace, name, max_bytes=MAX_ARTIFACT_BYTES)
            row["status"] = "unchanged" if _sha(raw) == expected else "changed"
        except (OSError, ValueError, TypeError, KeyError):
            row["reason"] = "checkpoint_artifact_unavailable"
        if len(_json([*records, row])) > MAX_REPORT_CHARS:
            omissions.append({"reason": "checkpoint_summary_limit", "remaining_artifacts": len(artifacts) - len(records)})
            break
        records.append(row)
    else:
        if len(artifacts) > MAX_MATERIALS:
            omissions.append({"reason": "checkpoint_artifact_limit", "count": len(artifacts) - MAX_MATERIALS})
    return {"records": records, "omissions": omissions}


def build_evidence_index(workspace, *, receipts, inputs, checkpoint=None, source_reader=None):
    """Project trusted host receipts. Callers must not pass model-authored receipts.

    source_reader is the owner-checked original-material reader, never a model
    callback. Table contents are hash-checked but never included in the index.
    This does not establish scientific usefulness or report-claim coverage.
    """
    materials = _materials(workspace, inputs, source_reader)
    # Protect all registered hashes, including inputs omitted from the bounded public index.
    protected_hashes = {item["sha256"] for item in inputs if isinstance(item, dict)
                        and isinstance(item.get("sha256"), str) and SHA.fullmatch(item["sha256"])} if isinstance(inputs, list) else set()
    output_budget = [MAX_OUTPUT_BYTES]
    index = {"schema": SCHEMA, "materials": materials, "computations": [],
             "checkpoint_artifacts": _reports(workspace, checkpoint), "omissions": [],
             "boundary": "Host provenance and recorded values only; not scientific acceptance, derived-sequence validation or report-claim verification."}
    candidates = [row for row in receipts if isinstance(row, dict) and row.get("tool") == "queued_analysis"]
    for row in candidates[:MAX_COMPUTATIONS]:
        try:
            record = _computation(workspace, row, materials, protected_hashes, output_budget)
        except (OSError, ValueError, TypeError, KeyError, RecursionError):
            saved = row.get("result")
            task = saved.get("task_id") if isinstance(saved, dict) else None
            record = {"status": "unverified", "reason": "private_receipt_or_bound_artifact_invalid"}
            if isinstance(task, str) and JOB.fullmatch(task):
                record["task_id"] = task
        record = public_value(record, workspace)
        if len(_json(index)) + len(_json(record)) > MAX_INDEX_CHARS - 256:
            index["omissions"].append({"reason": "index_size_limit",
                                       "remaining_computations": len(candidates) - len(index["computations"])})
            break
        index["computations"].append(record)
    else:
        if len(candidates) > MAX_COMPUTATIONS:
            index["omissions"].append({"reason": "computation_limit",
                                       "count": len(candidates) - MAX_COMPUTATIONS})
    # Material and checkpoint metadata have their own explicit record limits.
    return public_value(index, workspace)
