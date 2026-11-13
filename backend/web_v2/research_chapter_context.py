"""Allocate compact, citable evidence to independent report chapters."""
from __future__ import annotations

import json

from .research_evidence import evidence_excerpt
from .research_evidence_passages import selected_passages
from .research_query_scope import query_result_scope

MAX_CONTEXT_CHARS = 24000

PREDICTION_TERMS = {
    "identity_evolution": ("conserved", "motif"),
    "function_pathways": ("activity site", "metal ion"),
    "structure_binding": ("binding site", "metal ion", "stability"),
    "interactions_regulation": ("localization", "sortingsignal"),
    "variants_modifications": ("conserved", "stability"),
    "biophysics_production": ("solubility", "stability"),
    "engineering_design": ("stability", "activity site", "binding site"),
    "datasets_applications": (),
}


def chapter_predictions(sources, chapter_id):
    terms = PREDICTION_TERMS.get(chapter_id, ())
    def priority(source):
        title = str(source.get("title", "")).casefold()
        return next((i for i, term in enumerate(terms) if term in title), len(terms))
    return sorted(sources, key=priority)


def select_chapter_evidence(core, materials, predictions, remote, retrieved, curated):
    """Reserve room for literature instead of truncating it behind model outputs."""
    literature = [s for s in retrieved + curated if s.get("kind") == "literature"]
    web = [s for s in retrieved if s.get("kind") != "literature"]
    groups = [(core, 4), (materials, 2), (predictions, 2),
              (remote, 2), (literature, 2), (web, 2)]
    result, seen = [], set()
    def add(source):
        if len(result) < 14 and source["id"] not in seen:
            result.append(source)
            seen.add(source["id"])
    for sources, reserve in groups:
        for source in sources[:reserve]:
            add(source)
    # Reuse empty allocations without allowing one source type to consume them all.
    for offset in range(max((len(sources) for sources, _ in groups), default=0)):
        for sources, reserve in groups:
            index = reserve + offset
            if index < len(sources):
                add(sources[index])
        if len(result) >= 14:
            break
    return result


def _source_metadata(source):
    # Full URLs, query payloads and source records remain in the dossier.
    # A specialist needs citation identity and the scientific query scope.
    limits = {"title": 256, "kind": 64, "evidence_level": 120, "year": 32,
              "sha256": 64, "sequence_sha256": 64, "query_tool": 80}
    row = {"id": source["id"]}
    coverage = query_result_scope(source)
    if coverage:
        row["query_result_scope"] = coverage
    for key, maximum in limits.items():
        if key in source:
            value = source[key]
            row[key] = value[:maximum] if isinstance(value, str) else value
    extraction = source.get("extraction")
    if isinstance(extraction, dict):
        row["extraction"] = {k: extraction[k] for k in
                             ("format", "units", "truncated", "characters") if k in extraction}
    arguments = source.get("query_arguments")
    if isinstance(arguments, dict):
        selected = {}
        payloads = {"sequence", "structure", "structure_pdb", "pdb", "input",
                    "content", "text", "path", "file_path", "local_path"}
        for key, value in arguments.items():
            if (key in payloads or not isinstance(value, (str, int, float, bool, type(None)))
                    or len(str(value)) > 240):
                continue
            candidate = {**selected, key: value}
            if len(json.dumps(candidate, ensure_ascii=False)) <= 600:
                selected = candidate
        if selected:
            row["query_arguments"] = selected
        if selected != arguments:
            row["query_arguments_excerpted"] = True
    return row


def source_context(sources, limit=14, *, query=""):
    """Select chapter-specific excerpts under one serialized context budget."""
    rows = []
    for source in sources[:min(limit, 14)]:
        kind = source.get("kind")
        maximum = 14000 if kind == "database" else 12000 if kind == "user_material" else 9000 if kind == "public_service" else 2400
        if query:
            selected = selected_passages(source, query, maximum)
            excerpts = []
            for passage in selected["passages"]:
                pointer = passage["location"].get("json_pointer")
                label = ("Source field " + json.dumps(pointer, ensure_ascii=False) + ":\n"
                         if pointer else "")
                excerpts.append(label + passage["text"])
            text = "\n\n".join(excerpts)
            excerpted = bool(source.get("excerpted")) or text != str(source.get("text", ""))
        else:
            text, excerpted = evidence_excerpt(source, maximum)
        rows.append({**_source_metadata(source), "text": text, "excerpted": excerpted})

    def capped(maximum):
        result = []
        for row in rows:
            text = row["text"]
            if len(text) > maximum:
                boundary = max(text.rfind("\n\n", maximum // 2, maximum),
                               text.rfind(". ", maximum // 2, maximum))
                text = text[:boundary + 1 if boundary >= 0 else maximum].rstrip()
            result.append({**row, "text": text,
                           "excerpted": row["excerpted"] or text != row["text"]})
        return result

    def size(value):
        return len(json.dumps(value, ensure_ascii=False))

    if size(rows) <= MAX_CONTEXT_CHARS:
        return rows
    if size(capped(0)) > MAX_CONTEXT_CHARS:
        raise ValueError("Source metadata exceeds chapter context budget")
    low, high = 0, max((len(row["text"]) for row in rows), default=0)
    while low < high:
        midpoint = (low + high + 1) // 2
        if size(capped(midpoint)) <= MAX_CONTEXT_CHARS:
            low = midpoint
        else:
            high = midpoint - 1
    return capped(low)


def chapter_bundle(state, payload, chapter, remit, evidence, context=None):
    """Raw prediction arrays stay in the dossier, never duplicated in every prompt."""
    # The shared question stays in the writing brief; chapter evidence follows
    # its own remit so broad protein names cannot displace specialist annotations.
    query = " ".join(str(value or "") for value in (chapter["title"], remit))
    # Older callers can still supply a one-argument context builder.
    selected = source_context(evidence, query=query) if context is None else context(evidence)
    return {
        "protein": state.get("protein"),
        "research_question": payload.get("focus"),
        "chapter": chapter["title"],
        "remit": remit,
        "sources": selected,
        "evidence_scope": {
            "selected_source_ids": [s["id"] for s in evidence],
            "full_results": "Original source records and prediction files remain in the dossier.",
            "prediction_interpretation": "Class indices and scores retain their original meaning. Do not infer class labels, units or calibration absent from the source.",
        },
        "public_service_status": [
            {k: row.get(k) for k in ("id", "status", "count", "summary")}
            for row in state.get("remote_services", [])
        ],
    }
