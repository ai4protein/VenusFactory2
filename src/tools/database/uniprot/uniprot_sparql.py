"""UniProt SPARQL endpoint wrapper (https://sparql.uniprot.org/sparql).

Submit a SPARQL query, return parsed JSON bindings. Writes the full SPARQL JSON
response to a file (results can be large) and returns a compact summary.

Adapted from google-deepmind/science-skills (Apache-2.0):
- skills/uniprot_database/scripts/uniprot_tools.py (sparql_query)
"""
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.tools.path_sanitizer import to_client_file_path


SPARQL_URL = "https://sparql.uniprot.org/sparql"

_PREVIEW_LEN = 1500
_SOURCE = "UniProt SPARQL"


def _session() -> requests.Session:
    s = requests.Session()
    r = Retry(total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504], allowed_methods=frozenset({"GET", "POST"}))
    s.mount("https://", HTTPAdapter(max_retries=r))
    s.mount("http://", HTTPAdapter(max_retries=r))
    return s


def _error_response(error_type: str, message: str, suggestion: Optional[str] = None) -> str:
    out: Dict[str, Any] = {
        "status": "error",
        "error": {"type": error_type, "message": message},
        "file_info": None,
    }
    if suggestion:
        out["error"]["suggestion"] = suggestion
    return json.dumps(out, ensure_ascii=False)


def _success_response(
    file_path: str, content_preview: str, biological_metadata: Dict[str, Any], elapsed_ms: int,
) -> str:
    path = Path(file_path)
    size = path.stat().st_size if path.exists() else 0
    out: Dict[str, Any] = {
        "status": "success",
        "file_info": {
            "file_path": to_client_file_path(path if path.exists() else file_path),
            "file_name": path.name,
            "file_size": size,
            "format": "json",
        },
        "content_preview": content_preview[:_PREVIEW_LEN],
        "biological_metadata": biological_metadata,
        "execution_context": {"elapsed_ms": elapsed_ms, "source": _SOURCE},
    }
    return json.dumps(out, ensure_ascii=False)


def _summarize_bindings(bindings: List[Dict[str, Any]], limit: int = 10) -> str:
    if not bindings:
        return "_(no bindings returned)_"
    vars_seen: List[str] = []
    for b in bindings[:1]:
        vars_seen = list(b.keys())
    lines = [
        f"SPARQL bindings — {len(bindings)} rows, vars: {vars_seen}",
        f"first {min(limit, len(bindings))}:",
    ]
    for b in bindings[:limit]:
        compact = {k: v.get("value", "") for k, v in b.items()}
        lines.append(f"  {compact}")
    return "\n".join(lines)


def download_uniprot_sparql_by_query(
    query: str,
    out_dir: str,
    timeout: int = 120,
) -> str:
    """Execute a SPARQL query against sparql.uniprot.org and save the JSON response.

    Returns the standard rich JSON envelope. The full SPARQL JSON (which may be
    large) is written to `<out_dir>/uniprot_sparql_<timestamp>.json`; the
    response includes a compact preview of the first 10 result rows.
    """
    t0 = time.perf_counter()
    if not query or not query.strip():
        return _error_response("ValidationError", "empty SPARQL query")
    out_dir = str(out_dir or "").strip().rstrip(os.sep)
    if not out_dir:
        return _error_response("ValidationError", "empty out_dir")
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    sess = _session()
    # Use POST to support long queries
    try:
        resp = sess.post(
            SPARQL_URL,
            data={"query": query, "format": "json"},
            headers={"Accept": "application/sparql-results+json"},
            timeout=timeout,
        )
    except requests.RequestException as e:
        return _error_response("NetworkError", str(e))

    if resp.status_code != 200:
        return _error_response(
            "SparqlError",
            f"sparql.uniprot.org [{resp.status_code}]: {resp.text[:500]}",
            suggestion="Validate the SPARQL syntax at https://sparql.uniprot.org/.",
        )
    try:
        body = resp.json()
    except ValueError:
        return _error_response("ParseError", f"non-JSON SPARQL response: {resp.text[:500]}")

    timestamp = int(time.time())
    out_path = os.path.join(out_dir, f"uniprot_sparql_{timestamp}.json")
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(body, f, ensure_ascii=False, indent=2)
    except OSError as e:
        return _error_response("IOError", f"failed to write response: {e}")

    bindings = (body.get("results") or {}).get("bindings") or []
    head_vars = (body.get("head") or {}).get("vars") or []
    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    meta = {
        "endpoint": SPARQL_URL,
        "row_count": len(bindings),
        "head_vars": head_vars,
    }
    preview = _summarize_bindings(bindings)
    return _success_response(out_path, preview, meta, elapsed_ms)


__all__ = ["SPARQL_URL", "download_uniprot_sparql_by_query"]
