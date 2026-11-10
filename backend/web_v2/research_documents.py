"""Report document validation and rendering, independent of execution and storage.

Writers, owner views and dossier exports share these acceptance rules. Reading a
saved report never starts retrieval, model calls or background work.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
import time

MINIMUM_CHAPTERS = 5


def text_size(text, lang):
    return len(re.findall(r'[\u3400-\u9fff]', text)) if lang == 'zh' else len(re.findall(r'\b[\w-]+\b', text))


def chapter_body(text):
    # The coordinator owns section numbering; child agents own their subsection content.
    return re.sub(r'^#{1,2}\s+', '### ', text.strip(), flags=re.M)


def normalize_source_citations(text, sources):
    """Normalize explicit source references; never infer citations for prose."""
    titles = {s["id"]: str(s.get("title", "")).strip().casefold()
              for s in sources if re.fullmatch(r"S\d+", str(s.get("id", "")))}
    def named(match):
        title, identity = match.group(1).strip(), match.group(2)
        return "[" + identity + "]" if titles.get(identity) == title.casefold() else match.group(0)
    def group(match):
        return " ".join("[" + identity + "]" for identity in re.findall(r"S\d+", match.group(1)))
    # Code samples remain verbatim; an example marker is not prose evidence.
    parts = re.split(r"(```[\s\S]*?```|~~~[\s\S]*?~~~|`[^`\n]*`)", text)
    for index in range(0, len(parts), 2):
        body = re.sub(r"\(([^()\n]{1,300})[,;]\s*(S\d+)\s*\)", named, parts[index])
        body = re.sub(r"\((S\d+(?:\s*[,;]\s*S\d+)*)\)", group, body)
        parts[index] = re.sub(r"(?<!!)\[(S\d+(?:\s*[,;]\s*S\d+)+)\](?!\()", group, body)
    return "".join(parts)


def contains_long_sequence(text):
    # Sequences are delivered from verified input snapshots, not generated prose.
    if re.search(r"\b[ACDEFGHIKLMNPQRSTVWY]{60,}\b", text):
        return True
    residues = "Ala|Arg|Asn|Asp|Cys|Gln|Glu|Gly|His|Ile|Leu|Lys|Met|Phe|Pro|Ser|Thr|Trp|Tyr|Val"
    return bool(re.search(r"\b(?:" + residues + r")(?:[-\s]+(?:" + residues + r")){29,}\b", text, re.I))


def chapter_document(chapter):
    return {"id": chapter["id"], "title": chapter["title"], "text": chapter.get("draft", "")}


def chapter_ready(chapter, sources=None):
    if chapter.get("status") != "completed" or not chapter.get("draft") or contains_long_sequence(chapter["draft"]):
        return False
    if chapter.get("evidence_review_required"):
        from .research_source_review import document_ready
        if sources is None or not chapter.get("id") or not chapter.get("title"):
            return False
        return document_ready(chapter_document(chapter), sources, chapter.get("source_review"))
    return True


def refresh_review_progress(state):
    """Project bound review receipts into chapter progress without completing a report."""
    from .research_source_review import VERSION, document_ready, document_signature

    review = state.get("source_review", {})
    receipts = review.get("documents", {})
    sources = state.get("sources", [])
    changed = False
    for chapter in state.get("chapters", []):
        if chapter.get("status") in {"searching", "writing", "revising"}:
            continue
        receipt = receipts.get(chapter["id"])
        if not isinstance(receipt, dict) or receipt.get("policy") != VERSION:
            continue
        if chapter.get("draft_status") != "completed" and not chapter_ready(chapter, sources):
            continue
        document = chapter_document(chapter)
        if receipt.get("input_sha256") != document_signature(document, sources):
            continue
        status = ("completed" if document_ready(document, sources, receipt)
                  else "reviewing" if review.get("status") == "checking" else "partial")
        changed |= chapter.get("source_review") != receipt or chapter.get("status") != status
        chapter.update(source_review=copy.deepcopy(receipt), status=status)
    completed = sum(chapter_ready(chapter, sources) for chapter in state.get("chapters", []))
    minimum_met = completed >= MINIMUM_CHAPTERS
    changed |= state.get("completed_chapters") != completed or state.get("minimum_met") != minimum_met
    state.update(completed_chapters=completed, minimum_met=minimum_met)
    return changed


def remember_chapter(chapter, draft, audit, truncated=False):
    history = chapter.setdefault("draft_history", [])
    if not history or history[-1]["draft"] != draft:
        history.append({"draft": draft, "audit": audit, "truncated": bool(truncated),
                        "attempt": chapter.get("attempts", 0), "recorded": time.time()})


def chapter_text(chapter, lang, sources=None):
    if chapter_ready(chapter, sources):
        return chapter_body(chapter['draft'])
    if chapter.get('status') == 'reviewing':
        return '正在核对来源。' if lang == 'zh' else 'Checking sources.'
    if chapter.get('status') in {'partial', 'unavailable', 'completed'}:
        return '本章待修订。' if lang == 'zh' else 'This chapter is awaiting revision.'
    return '正在研究。' if lang == 'zh' else 'Research in progress.'


def audit_chapter(text, source_ids, lang):
    prose = re.sub(r"```[\s\S]*?```|~~~[\s\S]*?~~~|`[^`\n]*`", "", text)
    cited = set(re.findall(r'\[(S\d+)\]', prose))
    unknown = sorted(cited - set(source_ids))
    return {'length': text_size(text, lang), 'unknown_citations': unknown, 'cited_sources': sorted(cited & set(source_ids)),
            'substantial': text_size(text, lang) >= (900 if lang == 'zh' else 400), 'has_citations': bool(cited & set(source_ids)),
            'long_sequence': contains_long_sequence(text)}


def citation_coverage(text):
    tick = chr(96)
    prose = re.sub(tick * 3 + r"[\s\S]*?" + tick * 3 + r"|~~~[\s\S]*?~~~", "", text)
    prose = re.sub(tick + r"[^" + tick + r"\n]*" + tick, "", prose)
    checked = []
    for block in re.split(r"\n\s*\n", prose):
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        lines = [line for line in lines if not line.startswith("#")
                 and not re.fullmatch(r"\*\*[^*]+\*\*", line)]
        body = " ".join(lines)
        if not body:
            continue
        table = any(line.startswith("|") for line in lines)
        words = len(re.findall(r"\b[\w-]+\b", body))
        cjk = len(re.findall(r"[\u3400-\u9fff]", body))
        if words < (16 if table else 35) and cjk < (25 if table else 80):
            continue
        checked.append(bool(re.search(r"\[S\d+\]", body)))
    cited = sum(checked)
    coverage = cited / len(checked) if checked else 1.0
    return {"citation_blocks": len(checked), "cited_blocks": cited,
            "uncited_block_indices": [index + 1 for index, ok in enumerate(checked) if not ok],
            "citation_coverage": round(coverage, 3), "citation_coverage_ok": coverage >= 0.75}


def editor_audit(text, source_ids, lang):
    return {**audit_chapter(text, source_ids, lang), **citation_coverage(text)}


def chapter_signature(state):
    return hashlib.sha256(json.dumps([
        {key: chapter.get(key) for key in ("id", "title", "draft", "source_ids")}
        for chapter in state.get("chapters", [])
    ], sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def source_review_required(state):
    """Specialist reports require review even when older checkpoints lack the flag."""
    return state.get("workflow") == "specialist_research_v2" or bool(state.get("evidence_review_required"))


def review_ready(state, lang):
    if state.get("review_status", "completed") != "completed" or not state.get("review"):
        return False
    sources = state.get("sources", [])
    audit = editor_audit(state["review"], [s["id"] for s in sources], lang)
    if source_review_required(state):
        from .research_source_review import document_ready
        document = {"id": "editor", "title": "Integrated assessment", "text": state["review"]}
        if (not document_ready(document, sources, state.get("review_source_review"))
                or state.get("review_chapters_sha256") != chapter_signature(state)):
            return False
    return not incomplete_review(audit, state.get("review_truncated", False), bool(sources))


def incomplete_chapter(audit, truncated=False, has_sources=True):
    """Structural eligibility for source review, shared by writers and recovery."""
    return bool(truncated or audit["long_sequence"] or audit["unknown_citations"]
                or not audit["substantial"] or (has_sources and not audit["has_citations"]))


def incomplete_review(audit, truncated, has_sources):
    return incomplete_chapter(audit, truncated, has_sources) or not audit.get("citation_coverage_ok", True)


def cite_report(text, sources):
    # Only fetched source IDs become links. Arbitrary links/images from a model are removed.
    text = re.sub(r'!?\[([^\]]*)\]\([^)]*\)', r'\1', text)
    text = re.sub(r'https?://[^\s<>]+', '', text)
    known = {r['id']: r for r in sources}
    return re.sub(r'\[(S\d+)\]', lambda m: f"[{m[1]}]({known[m[1]]['url']})" if m[1] in known and known[m[1]].get('url') else ('[' + m[1] + ']' if m[1] in known else '[source unavailable]'), text)


def report_view(state, lang):
    """Reconcile a saved document without altering its checkpoint or lifecycle.

    Stored completion counts are observations from the previous execution. The
    owner API and dossier must count only chapters accepted by the current rules.
    A queue failure, cancellation or running state is never promoted to success.
    """
    value = copy.deepcopy(state)
    if source_review_required(value):
        from .research_source_review import VERSION
        value["evidence_review_required"] = VERSION
    sources = value.get("sources", [])
    chapters = value.get("chapters")
    demoted = False
    if chapters is not None:
        completed = 0
        for chapter in chapters:
            if value.get("evidence_review_required"):
                chapter["evidence_review_required"] = value["evidence_review_required"]
            accepted = chapter_ready(chapter, sources)
            completed += accepted
            if chapter.get("status") == "completed" and not accepted:
                chapter["status"] = "partial"
                demoted = True
        # A synthesis may have used a chapter whose receipt is no longer valid,
        # even if the chapter's prose and its saved signature have not changed.
        if demoted and value.get("review"):
            value["review_status"] = "partial"
        minimum = value.get("minimum_chapters", MINIMUM_CHAPTERS)
        minimum = max(MINIMUM_CHAPTERS, minimum) if type(minimum) is int else MINIMUM_CHAPTERS
        value.update(completed_chapters=completed, minimum_chapters=minimum,
                     minimum_met=completed >= minimum)
        requires_editor = source_review_required(value)
        value["partial"] = bool(value.get("partial") or value.get("collection_partial")
                                or completed != len(chapters) or not value["minimum_met"]
                                or (requires_editor and not review_ready(value, lang)))
    if value.get("review"):
        value["review_audit"] = editor_audit(value["review"], [source["id"] for source in sources], lang)
        if value.get("review_status", "completed") == "completed" and not review_ready(value, lang):
            value.update(review_status="partial", partial=True)
    if value.get("partial") or value.get("minimum_met") is False:
        if value.get("status") == "completed":
            value["status"] = "incomplete"
        if value.get("phase") == "completed":
            value["phase"] = "incomplete"
    return value


def assemble(state, lang):
    from mcp_tools.tools.research import services as public_services

    from . import research_predictions

    zh = lang == 'zh'
    title = state.get('protein', {}).get('name', 'Protein research')
    parts = ['# ' + title, state.get('overview', ''), '## ' + ('目录' if zh else 'Contents')]
    parts.extend(f"{i}. {c['title']}" for i, c in enumerate(state['chapters'], 1))
    for i, chapter in enumerate(state['chapters'], 1):
        parts += [f"## {i}. {chapter['title']}"]
        if chapter_ready(chapter, state.get("sources", [])):
            allowed = [s for s in state['sources'] if s['id'] in chapter['source_ids']]
            parts.append(cite_report(chapter_body(chapter['draft']), allowed))
        else:
            parts.append(chapter_text(chapter, lang, state.get("sources", [])))
    if review_ready(state, lang):
        parts += ['## ' + ('综合评述与研究建议' if zh else 'Integrated assessment & research priorities'), cite_report(state['review'], state['sources'])]
    parts.append(research_predictions.report_appendix(state.get('predictions', []), lang))
    parts.append(public_services.report_appendix(state.get('remote_services', []), lang))
    parts += ['## ' + ('来源' if zh else 'Sources')]
    parts += [f"- [{s['id']} · {s['title']}]({s['url']}) · {s['evidence_level']}" for s in state['sources'] if s.get('url')]
    return '\n\n'.join(x for x in parts if x)
