"""Allocate compact, citable evidence to independent report chapters."""
from __future__ import annotations

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


def chapter_bundle(state, payload, chapter, remit, evidence, context):
    """Raw prediction arrays stay in the dossier, never duplicated in every prompt."""
    selected = context(evidence)
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
