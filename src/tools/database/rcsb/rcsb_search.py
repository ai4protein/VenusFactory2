"""RCSB Search API v2 wrapper.

POST a structured JSON query to https://search.rcsb.org/rcsbsearch/v2/query
and return matching identifiers (PDB IDs, assemblies, polymer entities, etc.).

Adapted from google-deepmind/science-skills (Apache-2.0):
- skills/pdb_database/scripts/search_pdb.py
"""
import json
import time
import urllib.parse
from typing import Any, Dict, List, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


RCSB_SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"

ALLOWED_RETURN_TYPES = {
    "entry",                  # [PDB ID]
    "assembly",               # [PDB ID]-[ASSEMBLY ID]
    "polymer_entity",         # [PDB ID]-[ENTITY ID]
    "non_polymer_entity",     # [PDB ID]-[ENTITY ID]
    "polymer_instance",       # [PDB ID].[LABEL ASYM ID]
    "mol_definition",         # [CHEMICAL COMP ID] or [BIRD ID]
}


def _session() -> requests.Session:
    s = requests.Session()
    r = Retry(total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504], allowed_methods=frozenset({"GET"}))
    s.mount("https://", HTTPAdapter(max_retries=r))
    s.mount("http://", HTTPAdapter(max_retries=r))
    return s


def _build_payload(
    query: Any,
    return_type: str,
    page_start: Optional[int],
    rows: Optional[int],
    sort_by: Optional[str],
    sort_direction: Optional[str],
    count_only: bool,
) -> Dict[str, Any]:
    """Construct the request payload from a query (block) or full payload."""
    if isinstance(query, dict) and "query" in query:
        payload: Dict[str, Any] = dict(query)
    else:
        payload = {"query": query}

    payload["return_type"] = return_type
    request_options: Dict[str, Any] = dict(payload.get("request_options") or {})

    if count_only:
        request_options["paginate"] = {"start": 0, "rows": 0}
        request_options.pop("return_all_hits", None)
    elif page_start is not None or rows is not None:
        request_options.pop("return_all_hits", None)
        paginate = dict(request_options.get("paginate") or {})
        if page_start is not None:
            paginate["start"] = page_start
        if rows is not None:
            paginate["rows"] = rows
        request_options["paginate"] = paginate
    else:
        request_options["return_all_hits"] = True

    if sort_by:
        sort_item: Dict[str, Any] = {"sort_by": sort_by}
        if sort_direction:
            sort_item["direction"] = sort_direction
        request_options["sort"] = [sort_item]

    if request_options:
        payload["request_options"] = request_options
    return payload


def search_rcsb_pdb(
    query: Any,
    return_type: str = "entry",
    page_start: Optional[int] = None,
    rows: Optional[int] = None,
    sort_by: Optional[str] = None,
    sort_direction: Optional[str] = None,
    count_only: bool = False,
    timeout: int = 60,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Execute a search and return (raw_response_json, summary_metadata).

    `query` may be:
    - a "query block" (dict with `type`, `service`, `parameters`); the function
      wraps it in `{"query": <block>, "return_type": ...}`.
    - a full request payload (dict that already contains a `query` key); used
      as-is and only `return_type`/pagination/sort are merged in.
    - a JSON string of either of the above (auto-parsed).
    """
    if return_type not in ALLOWED_RETURN_TYPES:
        raise ValueError(f"return_type must be one of {sorted(ALLOWED_RETURN_TYPES)}; got {return_type!r}")

    if isinstance(query, str):
        try:
            query_obj: Any = json.loads(query)
        except json.JSONDecodeError as e:
            raise ValueError(f"query string is not valid JSON: {e}")
    else:
        query_obj = query

    payload = _build_payload(
        query_obj, return_type, page_start, rows, sort_by, sort_direction, count_only
    )
    json_payload = json.dumps(payload, separators=(",", ":"))
    url = f"{RCSB_SEARCH_URL}?json={urllib.parse.quote(json_payload)}"

    sess = _session()
    t0 = time.monotonic()
    resp = sess.get(url, timeout=timeout)
    elapsed_ms = int((time.monotonic() - t0) * 1000)

    if resp.status_code == 204:
        # RCSB returns 204 when there are no matches
        body: Dict[str, Any] = {"total_count": 0, "result_set": []}
    elif resp.status_code != 200:
        raise RuntimeError(f"RCSB search failed [{resp.status_code}]: {resp.text[:500]}")
    else:
        try:
            body = resp.json()
        except ValueError:
            raise RuntimeError(f"RCSB search returned non-JSON body: {resp.text[:500]}")

    summary: Dict[str, Any] = {
        "return_type": return_type,
        "total_count": int(body.get("total_count", 0) or 0),
        "result_count": len(body.get("result_set") or []),
        "count_only": bool(count_only),
        "page_start": page_start,
        "rows": rows,
        "sort_by": sort_by,
        "elapsed_ms": elapsed_ms,
    }
    return body, summary


__all__ = ["RCSB_SEARCH_URL", "ALLOWED_RETURN_TYPES", "search_rcsb_pdb"]
