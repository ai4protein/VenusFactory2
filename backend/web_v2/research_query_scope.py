"""Keep recorded query coverage visible when selecting small evidence excerpts."""
from __future__ import annotations

import json


def query_result_scope(source):
    """Return bounded coverage metadata from the saved query response, not prose."""
    if not isinstance(source.get("query_tool"), str) or not source["query_tool"]:
        return {}
    text = source.get("text")
    if not isinstance(text, str) or len(text) > 1_000_000:
        return {}
    try:
        result = json.loads(text)
    except (ValueError, RecursionError):
        return {}
    if not isinstance(result, dict) or result.get("status") not in ("completed", "empty"):
        return {}
    records, count, total = result.get("records"), result.get("count"), result.get("total")
    if (not isinstance(records, list) or type(count) is not int or type(total) is not int
            or not 0 <= count <= total <= 2**53-1 or count != len(records)
            or type(result.get("has_more")) is not bool):
        return {}
    scope = {"coverage": "database_query", "returned_records": count,
             "reported_total": total, "more_records_available": result["has_more"]}
    page = result.get("page")
    if type(page) is int and 0 <= page <= 2**53-1:
        scope["page"] = page
    arguments = source.get("query_arguments")
    limit = arguments.get("limit") if isinstance(arguments, dict) else None
    if type(limit) is int and 0 < limit <= 2**53-1:
        scope["page_size"] = limit
    truncated = result.get("truncated_fields")
    if isinstance(truncated, list):
        scope["nested_list_truncation"] = bool(truncated)
    return scope
