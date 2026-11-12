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
    """Conservative prospective-only syntax, supplementary to semantic review.

    One proposal marker cannot exempt another sentence, heading or observation.
    Ambiguous mixed prose must be split and reviewed with factual evidence; this
    guard does not attempt to determine the truth of arbitrary natural language.
    """
    neutral = re.compile(
        r"(?:future experiments?|next steps?|research priorities|research suggestions|"
        r"(?:proposed|recommended) experiments?|experimental proposals|"
        r"后续实验|研究建议|下一步|建议实验|未来实验)", re.I)
    prefix = re.compile(
        r"(?:we\s+(?:propose|recommend)\b|(?:proposed|recommended)\s+(?:experiments?|tests?|analyses|assays?)\b|"
        r"future experiments?\s+(?:should|could|will)\b|(?:test|measure|compare|evaluate|assess|investigate)\b|"
        r"(?:(?:the|this|these|that)\s+)?"
        r"(?:(?:binding|expression|regulatory|functional|proposed|candidate|predicted|possible|"
        r"temperature|dose|concentration|physiological)\s+){0,4}"
        r"(?:hypotheses|hypothesis|effects?|activity|expression|binding|stability|solubility|"
        r"affinity|localization|kinetics|yield)\s+(?:could|should|may|can)\s+be\s+"
        r"(?:tested|evaluated|assessed|measured|compared|investigated)\b|"
        r"(?:下一步)?建议|拟开展|下一步(?:检验|测试|测量|比较)|"
        r"(?:该|这一|此)?(?:假设|效应|活性|表达|结合|稳定性|溶解度|亲和力|定位|产量)"
        r"(?:可以|可|应该|应|需要|需)(?:(?:通过|在)[^，,。;；!?！？]{1,60})?"
        r"(?:检验|测试|测量|比较|评估|验证))", re.I)
    observation = re.compile(
        r"\b(?:was|were|observed|(?<!be )measured|reported|confirmed|demonstrated|established|already|"
        r"shows?|demonstrates?)\b|"
        r"已知|已有|已经|观察到|测得|证实|表明|显示", re.I)
    assertion = re.compile(r"\b(?:is|are|has|have|contains?|binds?|cataly[sz]es?)\b", re.I)
    explanation = re.compile(r"\b(?:because|since|although|whereas|but)\b|因为|由于|然而", re.I)
    list_item = re.compile(
        r"(?:(?:and\s+)?(?:expression|yield|stability|solubility|activity|affinity|purity|"
        r"aggregation|folding|localization|kinetics|controls?)"
        r"(?:\s+(?:at|under|in|with)\s+[^,:;]+)?|"
        r"(?:e\.g\.|i\.e\.|for example)\s+(?:with|using|via|at|under|in)\s+[^,:;]+)", re.I)
    parts = []
    for line in text.splitlines():
        line = re.sub(r"\[S\d+\]", "", line).strip()
        clean = re.sub(r"^(?:#{1,6}\s*|[-*>]+\s*|\d+[.)]\s*)", "", line).strip(" *")
        if neutral.fullmatch(clean):
            continue
        # Protect abbreviation dots, not following prose: E. coli and e.g.
        # remain in their clause; a later sentence still needs its own scope.
        clean = re.sub(r"\b[A-Z]\.(?=\s+[a-z]{2,}\b)|\b(?:e\.g\.|i\.e\.)",
                       lambda match: match.group().replace(".", "\u2024"), clean)
        parts.extend(part.replace("\u2024", ".") for part in
                     re.split(r"(?<=[!?;。！？；])\s*|(?<!\d)\.(?=\s|$)", clean))
    parts = [part.strip(" .!?;。！？；") for part in parts if part.strip(" .!?;。！？；")]
    if not parts:
        return False
    for part in parts:
        if not prefix.match(part) or observation.search(part) or explanation.search(part):
            return False
        # A declared hypothesis may contain a present-tense proposition; an
        # independent sentence or comma clause cannot borrow its "whether".
        clauses = re.split(r"[,，]", part)
        for index, clause in enumerate(clauses):
            prospective = bool(prefix.match(clause.strip()))
            if index and not prospective and not list_item.fullmatch(clause.strip()):
                return False
            if assertion.search(clause) and not re.search(r"\bwhether\b", clause, re.I):
                return False
            if index and assertion.search(clause) and not prospective:
                return False
    return True


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
                raise ValueError("Proposal-only units must contain only explicitly prospective actions. Split existing facts or observations into separate factual units and supply their source evidence; planned numerical conditions may remain in pure proposals.")
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
