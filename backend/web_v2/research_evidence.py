"""Bounded evidence excerpts preserve annotation text before long reference lists."""
from __future__ import annotations

import json
import re


_PRIORITY = {key: i for i, key in enumerate((
    "proteinDescription", "genes", "organism", "scope", "commentType", "value",
    "name", "description", "reaction", "cofactors", "texts", "comments",
    "features", "crossReferences", "evidences"))}


def _compact(value, depth=0, field=""):
    if depth > 9:
        return {"_excerpted": True}
    if isinstance(value, dict):
        result = {}
        omitted = {}
        for key in sorted(value, key=lambda key: _PRIORITY.get(key, 50)):
            content = value[key]
            if key == "sequence" and isinstance(content, str) and len(content) > 120:
                result[key] = {"length": len(content), "content_in_original_source": True}
                continue
            if isinstance(content, list):
                limit = 3 if key == "evidences" else 24 if key == "comments" else 12
                if len(content) > limit:
                    omitted[key] = len(content) - limit
                content = content[:limit]
            result[key] = _compact(content, depth + 1, key)
        if omitted:
            result["_omitted_items"] = omitted
        return result
    if isinstance(value, list):
        return [_compact(item, depth + 1, field) for item in value[:24]]
    if isinstance(value, str):
        def references(match):
            refs = re.findall(r"PubMed:\d+", match.group(1))
            return "(" + ", ".join(refs[:3]) + f"; {len(refs)-3} additional references in original)"
        text = re.sub(r"\((PubMed:\d+(?:,\s*PubMed:\d+){3,})\)", references, value)
        return text if len(text) <= 4000 else text[:4000] + " [excerpt continues in original]"
    return value


def evidence_excerpt(source, limit):
    original = str(source.get("text", ""))
    text = original
    if source.get("kind") in {"database", "structure", "calculation", "prediction", "public_service"}:
        try:
            value = json.loads(original)
            if isinstance(value, (dict, list)):
                text = json.dumps(_compact(value), ensure_ascii=False, separators=(",", ":"))
        except (ValueError, RecursionError):
            pass
    return text[:limit], bool(source.get("excerpted")) or text != original or len(text) > limit
