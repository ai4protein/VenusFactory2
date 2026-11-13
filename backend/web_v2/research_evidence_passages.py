"""Bounded, source-addressable passages for independent evidence review."""
from __future__ import annotations

import hashlib
import json
import re

from .research_query_scope import query_result_scope

MAX_PASSAGE = 1600
# Reserve a bounded share for input identity and interpretation fields in the
# original database/service response, even when the draft names a wrong subject.
QUALIFIER_FIELDS = frozenset({
    "organism", "genes", "query_structure", "input_summary", "parameters",
    "units", "calibration", "interpretation", "evidence_level",
})


STOP_WORDS = set((
    "the and for with that this from have has are was were which their into than "
    "its not can could should would been also about these those such use using "
    "protein an as at be by do if in is it no of on or so to up us we"
).split())


def source_qualifier(location):
    parts = location.get("json_pointer", "").strip("/").split("/")
    if parts[0] == "data":
        parts = parts[1:]
    return bool(parts and parts[0] in QUALIFIER_FIELDS)


def digest(value):
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


def source_digest(source):
    return digest({key: source.get(key) for key in
                   ("id", "title", "url", "kind", "evidence_level", "text", "sha256",
                    "service", "query_tool", "query_arguments", "query", "parameters", "sequence_sha256")})


def terms(text):
    # Keep scientific abbreviations (pH, Km, Kd) and full identifiers while
    # matching structured fields such as kineticParameters to ordinary prose.
    names = re.sub(r"([A-Z])([A-Z][a-z])", r"\1 \2", text)
    names = re.sub(r"([a-z0-9])([A-Z][a-z])", r"\1 \2", names).replace("_", " ")
    words = {
        word.casefold()
        for word in re.findall(r"[A-Za-z0-9_]{2,}", text + " " + names)
        if len(word) >= 3 or word.isalpha()
    }
    for phrase in re.findall(r"[\u3400-\u9fff]{2,}", text):
        words.update(phrase[index:index+2] for index in range(len(phrase)-1))
    return words - STOP_WORDS


def chunks(text, limit=MAX_PASSAGE):
    offset = 0
    while offset < len(text):
        stop = min(len(text), offset+limit)
        if stop < len(text):
            boundary = max(text.rfind("\n", offset+limit//2, stop),
                           text.rfind(". ", offset+limit//2, stop))
            if boundary > offset:
                stop = boundary+1
        yield offset, text[offset:stop]
        offset = stop


def render_json_fragment(value, location):
    include_key = location.get("include_key", False)
    if type(include_key) is not bool:
        raise ValueError("Invalid JSON field anchor")
    if include_key:
        pointer = location["json_pointer"]
        if not pointer or isinstance(value, (dict, list)):
            raise ValueError("A named JSON field must identify a scalar")
        key = pointer.rsplit("/", 1)[-1].replace("~1", "/").replace("~0", "~")
        value = {key: value}
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def candidates(source):
    """JSON pointers and text offsets refer to saved source content, never a draft."""
    original = str(source.get("text", ""))
    rows = []
    def add(text, location, hint=""):
        if re.search(r"\b[ACDEFGHIKLMNPQRSTVWY]{60,}\b", text):
            return
        pointer = location.get("json_pointer", "").replace("~1", "/").replace("~0", "~")
        field_terms = {word for word in terms(pointer) if not word.isdecimal()}
        for offset, piece in chunks(text):
            if not piece.strip():
                continue
            anchor = {**location, "offset": offset, "length": len(piece)}
            rows.append({"id": source["id"]+":"+digest([anchor,piece])[:16],
                         "source_id": source["id"], "location": anchor, "text": piece,
                         "_terms": terms(piece+" "+hint) | field_terms,
                         "_qualifier": source_qualifier(location)})
    def walk(value, pointer="", depth=0, named_field=False):
        if len(rows) >= 2048 or depth > 16:
            return
        location = {"json_pointer": pointer}
        if named_field and not isinstance(value, (dict, list)):
            location["include_key"] = True
        rendered = render_json_fragment(value, location)
        if isinstance(value, str):
            add(rendered, location)
            return
        if (len(rendered) <= MAX_PASSAGE or depth >= 10) and not re.search(r"\b[ACDEFGHIKLMNPQRSTVWY]{60,}\b", rendered):
            hint = ""
            if any('"type":"' + kind + '"' in rendered for kind in (
                "Initiator methionine", "Signal", "Transit peptide", "Propeptide", "Chain", "Peptide")):
                hint = "processing cleavage maturation mature N-terminal signal peptide 成熟 切除 信号肽"
            add(rendered, location, hint)
        elif isinstance(value, dict):
            entries = list(value.items())
            if pointer in {"", "/data"}:
                entries.sort(key=lambda item: item[0] not in QUALIFIER_FIELDS)
            for key, child in entries:
                escaped = str(key).replace("~","~0").replace("/","~1")
                walk(child,pointer+"/"+escaped,depth+1,named_field=True)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                walk(child,pointer+"/"+str(index),depth+1)
    try:
        parsed = json.loads(original)
    except (ValueError, RecursionError):
        for offset, piece in chunks(original):
            if len(rows) >= 2048:
                break
            if piece.strip() and not re.search(r"\b[ACDEFGHIKLMNPQRSTVWY]{60,}\b", piece):
                rows.append({"id": source["id"]+":"+digest([offset,piece])[:16],
                             "source_id": source["id"], "location": {"text_offset":offset,"length":len(piece)},
                             "text":piece,"_terms":terms(piece)})
    else:
        walk(parsed)
    return rows


def selected_passages(source, query, budget):
    wanted = terms(query)
    available = candidates(source)
    ranked = sorted(enumerate(available),
                    key=lambda row: (-len(wanted & row[1]["_terms"]), row[0]))
    selected, used = [], 0
    reserved = set()
    # Context shares the same character and passage ceilings as query matches.
    # Keep these excerpts complete; never truncate a qualifier to reserve it.
    for row in available:
        size = len(row["text"])
        if row.get("_qualifier") and len(reserved) < 3 and used + size <= budget // 3:
            selected.append({key: value for key, value in row.items() if not key.startswith("_")})
            reserved.add(row["id"])
            used += size
    query_selected = 0
    for _, row in ranked:
        if row["id"] in reserved:
            continue
        size = len(row["text"])
        if used+size > budget:
            if query_selected or budget-used < 160:
                continue
            row = {**row, "text": row["text"][:budget-used],
                   "location": {**row["location"], "length": budget-used}}
            row["id"] = source["id"]+":"+digest([row["location"],row["text"]])[:16]
            size = len(row["text"])
        selected.append({key:value for key,value in row.items() if not key.startswith("_")})
        used += size
        query_selected += 1
        if len(selected) >= 8:
            break
    return {**{key: source.get(key) for key in ("service", "query_tool", "query_arguments", "sequence_sha256")},
            "id":source["id"], "title":source.get("title"), "kind":source.get("kind"),
            "evidence_level":source.get("evidence_level"), "source_sha256":source_digest(source),
            **({"query_result_scope": coverage} if (coverage := query_result_scope(source)) else {}),
            "passages":selected, "scope":"Selected excerpts; absence here is not evidence of absence."}


def resolve_passage(source, location):
    """Resolve the quote anchor against the original saved source, independently of selection."""
    original = str(source.get("text",""))
    if "text_offset" in location:
        offset = location["text_offset"]
        length = location["length"]
        if type(offset) is not int or type(length) is not int or offset<0 or length<1:
            raise ValueError("Invalid text passage anchor")
        return original[offset:offset+length]
    value = json.loads(original)
    pointer = location["json_pointer"]
    if pointer:
        if not isinstance(pointer,str) or not pointer.startswith("/"):
            raise ValueError("Invalid JSON passage anchor")
        for token in pointer[1:].split("/"):
            token=token.replace("~1","/").replace("~0","~")
            value=value[int(token)] if isinstance(value,list) else value[token]
    text = render_json_fragment(value, location)
    offset,length=location["offset"],location["length"]
    if type(offset) is not int or type(length) is not int or offset<0 or length<1:
        raise ValueError("Invalid JSON passage span")
    return text[offset:offset+length]
