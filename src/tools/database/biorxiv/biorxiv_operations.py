"""bioRxiv/medRxiv per-DOI fetch.

Complements VF's existing `query_biorxiv_tool` (keyword/category search by date
window) with single-paper lookup by DOI. The bioRxiv `details` endpoint returns
the canonical full record (authors, abstract, version history, JATS XML / PDF
links).

Adapted from google-deepmind/science-skills (Apache-2.0):
- skills/literature_search_biorxiv/scripts/search_by_doi.py
"""
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.tools.path_sanitizer import to_client_file_path


_PREVIEW_LEN = 1500
_SOURCE = "bioRxiv API"
_VALID_SERVERS = {"biorxiv", "medrxiv"}


def _session() -> requests.Session:
    s = requests.Session()
    r = Retry(total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504], allowed_methods=frozenset({"GET"}))
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


def _success_response(file_path: str, content_preview: str, biological_metadata: Dict[str, Any], elapsed_ms: int) -> str:
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


def download_biorxiv_by_doi(
    doi: str,
    out_dir: str,
    server: str = "biorxiv",
    include_abstract: bool = True,
    timeout: int = 30,
) -> str:
    """Fetch a single bioRxiv / medRxiv preprint by DOI and save JSON.

    `doi` may be the bare DOI (`10.1101/2023.05.01.538947`) or a full DOI URL.
    `server` is one of {"biorxiv", "medrxiv"}.
    """
    t0 = time.perf_counter()
    doi = (doi or "").strip()
    if not doi:
        return _error_response("ValidationError", "empty doi")
    server = (server or "").strip().lower()
    if server not in _VALID_SERVERS:
        return _error_response("ValidationError", f"server must be one of {sorted(_VALID_SERVERS)}; got {server!r}")
    out_dir = str(out_dir or "").strip().rstrip(os.sep)
    if not out_dir:
        return _error_response("ValidationError", "empty out_dir")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    # strip URL prefix if pasted
    for pfx in ("https://doi.org/", "http://doi.org/", "doi:"):
        if doi.lower().startswith(pfx):
            doi = doi[len(pfx):]

    sess = _session()
    url = f"https://api.biorxiv.org/details/{server}/{doi}"
    try:
        resp = sess.get(url, timeout=timeout, headers={"Accept": "application/json"})
    except requests.RequestException as e:
        return _error_response("NetworkError", str(e))
    if resp.status_code != 200:
        return _error_response("ApiError", f"biorxiv {resp.status_code}: {resp.text[:300]}")
    try:
        body = resp.json()
    except ValueError:
        return _error_response("ParseError", f"non-JSON response: {resp.text[:300]}")

    collection = body.get("collection") or []
    if not collection:
        return _error_response(
            "NotFound",
            f"no preprint found for DOI '{doi}' on {server}",
            suggestion="Verify the DOI; try the other server (medrxiv vs biorxiv).",
        )

    # Pick the most recent version (last entry typically newest)
    paper = collection[-1]
    if not include_abstract:
        paper.pop("abstract", None)

    safe_doi = doi.replace("/", "_").replace(":", "_")
    out_path = os.path.join(out_dir, f"{server}_{safe_doi}.json")
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"server": server, "doi": doi, "versions": collection, "latest": paper}, f, ensure_ascii=False, indent=2)
    except OSError as e:
        return _error_response("IOError", f"failed to write JSON: {e}")

    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    meta = {
        "server": server,
        "doi": doi,
        "versions_returned": len(collection),
        "title": paper.get("title"),
        "authors": paper.get("authors"),
        "date": paper.get("date"),
        "category": paper.get("category"),
        "license": paper.get("license"),
        "jatsxml": paper.get("jatsxml"),
    }
    preview = (
        f"bioRxiv {doi} (versions={len(collection)})\n"
        f"  title: {(paper.get('title') or '')[:200]}\n"
        f"  authors: {(paper.get('authors') or '')[:200]}\n"
        f"  date: {paper.get('date')}  category: {paper.get('category')}"
    )
    return _success_response(out_path, preview, meta, elapsed_ms)


__all__ = ["download_biorxiv_by_doi"]
