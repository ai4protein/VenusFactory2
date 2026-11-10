"""Validate review responses and source anchors without calls, state or retries."""
from __future__ import annotations

import json
import re

from .research_evidence_passages import resolve_passage, source_digest
from .research_measurements import measured_values_anchored


def _unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("Duplicate JSON object key")
        value[key] = item
    return value


def parse_response(text):
    """Normalize unambiguous envelopes; all rows still pass the same source checks."""
    tick = chr(96) * 3
    match = re.fullmatch(r"\s*" + tick + r"(?:json)?\s*([\s\S]*?)\s*" + tick + r"\s*", text)
    value = json.loads(match.group(1) if match else text, object_pairs_hook=_unique_object)
    if (isinstance(value, list) and len(value) == 1 and isinstance(value[0], dict)
            and set(value[0]) == {"assessments"}):
        value = value[0]
    if isinstance(value, dict) and set(value) == {"assessments"}:
        value = value["assessments"]
    if not isinstance(value, list):
        raise ValueError("Return an assessments array only")
    return value


def explicit_proposal(text):
    return bool(re.search(r"\b(?:we propose|we recommend|proposed|recommended|future experiments?|test whether|could be tested|should be tested)\b|建议|拟开展|可检验|下一步",text,re.I))


def validate_assessments(response, request, sources):
    """Validate coverage and source anchors independently of the model verdict."""
    return _validate_assessment_rows(parse_response(response), request, sources)


def _validate_assessment_rows(rows, request, sources):
    """Apply the same source contract to complete responses and recovery rows."""
    expected = {row["id"]: row for row in request["units"]}
    if (len(rows) != len(expected) or any(not isinstance(row, dict) for row in rows)
            or any(not isinstance(row.get("unit_id"), str) for row in rows)
            or {row.get("unit_id") for row in rows} != set(expected)):
        raise ValueError("Every requested unit must be assessed exactly once")
    originals = {row["id"]: row for row in sources}
    passages = {p["id"]: p for s in request["sources"] for p in s["passages"]}
    checked = []
    for row in rows:
        if set(row) != {"unit_id", "classification", "verdict", "issues", "evidence"}:
            raise ValueError("Unexpected assessment fields")
        unit = expected[row["unit_id"]]
        if row["classification"] not in {"factual", "proposal"} or row["verdict"] not in {"supported", "unsupported", "uncertain"}:
            raise ValueError("Invalid assessment classification or verdict")
        if not isinstance(row["issues"], list) or len(row["issues"]) > 8:
            raise ValueError("Invalid issue list")
        for issue in row["issues"]:
            if (not isinstance(issue, dict) or set(issue) != {"claim", "reason"}
                    or not isinstance(issue["claim"], str) or not issue["claim"].strip()
                    or issue["claim"] not in unit["text"] or not isinstance(issue["reason"], str)
                    or not 1 <= len(issue["reason"].strip()) <= 1500):
                raise ValueError("Issue must identify a claim in this unit")
        if not isinstance(row["evidence"], list) or len(row["evidence"]) > 12:
            raise ValueError("Invalid evidence list")
        evidence = []
        for item in row["evidence"]:
            if not isinstance(item, dict) or set(item) != {"source_id", "passage_id", "quote"}:
                raise ValueError("Invalid source anchor")
            passage = passages.get(item["passage_id"])
            source = originals.get(item["source_id"])
            quote = item["quote"]
            if (not passage or not source or item["source_id"] not in unit["citations"]
                    or passage["source_id"] != item["source_id"]
                    or not isinstance(quote, str) or len(quote.strip()) < 4
                    or quote not in passage["text"]
                    or quote not in resolve_passage(source, passage["location"])):
                raise ValueError("Quote is not in the cited original source passage")
            evidence.append({**item, "location": passage["location"], "source_sha256": source_digest(source)})
        if any(identity not in originals for identity in unit["citations"]):
            raise ValueError("Unknown source citation")
        if row["verdict"] == "supported":
            if row["issues"]:
                raise ValueError("Supported units cannot have unresolved issues")
            if row["classification"] == "factual" and not evidence:
                raise ValueError("Factual assertions require original-source evidence")
            if row["classification"] == "factual" and not measured_values_anchored(unit["text"], evidence):
                raise ValueError("Quoted evidence does not contain the claimed measured value and unit; preserve source values and units")
            if row["classification"] == "proposal" and not explicit_proposal(unit["text"]):
                raise ValueError("Uncited proposals must be explicitly proposed")
        elif not row["issues"]:
            raise ValueError("Unsupported or uncertain units require an identified issue")
        checked.append({**row, "evidence": evidence, "input_sha256": unit["input_sha256"],
                        "document_id": unit["document_id"], "text": unit["text"],
                        "passed": row["verdict"] == "supported"})
    return checked



def partial_assessments(response, request, sources):
    """Preserve individually valid assessments; never accept a partial prose unit."""
    rows = parse_response(response)
    expected = {unit["id"]: unit for unit in request["units"]}
    if (any(not isinstance(row, dict) or not isinstance(row.get("unit_id"), str) for row in rows)
            or len({row["unit_id"] for row in rows}) != len(rows)
            or not {row["unit_id"] for row in rows} <= set(expected)):
        raise ValueError("Ambiguous or unknown review unit identities")
    checked, invalid = [], {identity: "Missing assessment" for identity in expected}
    for row in rows:
        identity = row["unit_id"]
        single = {**request, "units": [expected[identity]]}
        try:
            # Use the full scientific and source-anchor validator for each unit.
            checked.extend(_validate_assessment_rows([row], single, sources))
            invalid.pop(identity)
        except (ValueError, KeyError, TypeError) as exc:
            invalid[identity] = str(exc)[:500]
    return checked, invalid
