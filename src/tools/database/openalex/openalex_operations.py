"""OpenAlex REST API wrapper.

OpenAlex (https://openalex.org) indexes scholarly works, authors, sources,
institutions, topics, concepts and more — a free alternative to Scopus / WoS.

Exposes two VenusFactory tools:
- download_openalex_entries_by_query: keyword + filter + sort search
- download_openalex_entry_by_id: fetch a single entity by OpenAlex ID

Adapted from google-deepmind/science-skills (Apache-2.0):
- skills/literature_search_openalex/scripts/openalex_cli.py
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


BASE_URL = "https://api.openalex.org"

_PREVIEW_LEN = 1500
_SOURCE = "OpenAlex"
_DEFAULT_PER_PAGE = 25
_MAX_PER_PAGE = 200

ENTITY_TYPES = {
    "works", "authors", "sources", "institutions", "topics", "concepts",
    "domains", "fields", "subfields", "sdgs", "countries", "continents",
    "languages", "keywords", "publishers", "funders",
}


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


def _build_url(entity_type: str, id_or_query: str = "") -> str:
    if id_or_query:
        return f"{BASE_URL}/{entity_type}/{id_or_query}"
    return f"{BASE_URL}/{entity_type}"


def _http_get(sess: requests.Session, url: str, params: Optional[Dict[str, Any]] = None, timeout: int = 30) -> requests.Response:
    p = dict(params or {})
    # Polite pool: include mailto if available, else just `email` env
    mailto = os.environ.get("OPENALEX_EMAIL") or os.environ.get("USER_EMAIL")
    if mailto:
        p["mailto"] = mailto
    api_key = os.environ.get("OPENALEX_API_KEY")
    if api_key:
        p["api_key"] = api_key
    return sess.get(url, params=p, timeout=timeout, headers={"User-Agent": "VenusFactory2 (https://venusfactory.cn)"})


def _summarize_works(results: List[Dict[str, Any]], limit: int = 5) -> str:
    if not results:
        return "_(no results)_"
    lines = []
    for r in results[:limit]:
        title = r.get("title") or r.get("display_name") or "(no title)"
        oid = r.get("id", "?").rsplit("/", 1)[-1]
        year = r.get("publication_year") or r.get("created_date", "")
        cited = r.get("cited_by_count", "?")
        lines.append(f"  - [{oid}] ({year}) cited={cited}: {title[:120]}")
    return "\n".join(lines)


def download_openalex_entries_by_query(
    entity_type: str,
    out_dir: str,
    search: Optional[str] = None,
    filter_expr: Optional[str] = None,
    sort: Optional[str] = None,
    per_page: int = _DEFAULT_PER_PAGE,
    page: int = 1,
    timeout: int = 30,
) -> str:
    """Search OpenAlex for entities of `entity_type`, save the JSON page response to disk.

    `entity_type` examples: works, authors, sources, institutions, topics.
    `search` is a free-text keyword. `filter_expr` is OpenAlex filter syntax
    (e.g. "authorships.author.id:A123,publication_year:>2020"). `sort` is e.g.
    "cited_by_count:desc". Returns the standard rich JSON envelope; the file
    at `file_info.file_path` is the raw OpenAlex JSON response page.
    """
    t0 = time.perf_counter()
    entity_type = (entity_type or "").strip().lower()
    if entity_type not in ENTITY_TYPES:
        return _error_response(
            "ValidationError",
            f"entity_type must be one of {sorted(ENTITY_TYPES)}; got {entity_type!r}",
        )
    out_dir = str(out_dir or "").strip().rstrip(os.sep)
    if not out_dir:
        return _error_response("ValidationError", "empty out_dir")
    if per_page < 1 or per_page > _MAX_PER_PAGE:
        return _error_response("ValidationError", f"per_page must be in [1, {_MAX_PER_PAGE}], got {per_page}")
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    params: Dict[str, Any] = {"per_page": per_page, "page": max(1, page)}
    if search:
        params["search"] = search
    if filter_expr:
        params["filter"] = filter_expr
    if sort:
        params["sort"] = sort

    sess = _session()
    try:
        resp = _http_get(sess, _build_url(entity_type), params=params, timeout=timeout)
    except requests.RequestException as e:
        return _error_response("NetworkError", str(e))
    if resp.status_code != 200:
        return _error_response(
            "ApiError",
            f"openalex {resp.status_code}: {resp.text[:500]}",
            suggestion="Verify entity_type, search/filter/sort syntax at https://docs.openalex.org/",
        )
    try:
        body = resp.json()
    except ValueError:
        return _error_response("ParseError", f"non-JSON response: {resp.text[:500]}")

    timestamp = int(time.time())
    safe_q = (search or filter_expr or "all").replace("/", "_").replace(" ", "_")[:40]
    out_path = os.path.join(out_dir, f"openalex_{entity_type}_{safe_q}_p{page}_{timestamp}.json")
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(body, f, ensure_ascii=False, indent=2)
    except OSError as e:
        return _error_response("IOError", f"failed to write response: {e}")

    meta_block = body.get("meta") or {}
    results = body.get("results") or []
    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    meta = {
        "entity_type": entity_type,
        "search": search,
        "filter": filter_expr,
        "sort": sort,
        "page": page,
        "per_page": per_page,
        "result_count": len(results),
        "total_count": meta_block.get("count"),
        "next_page": (page + 1) if (meta_block.get("count") and page * per_page < meta_block.get("count")) else None,
    }
    preview_header = (
        f"OpenAlex {entity_type} — search={search!r} filter={filter_expr!r} "
        f"page={page} returned={len(results)}/{meta_block.get('count')}"
    )
    return _success_response(out_path, f"{preview_header}\n{_summarize_works(results)}", meta, elapsed_ms)


def download_openalex_entry_by_id(
    entity_type: str,
    openalex_id: str,
    out_dir: str,
    timeout: int = 30,
) -> str:
    """Fetch a single OpenAlex entity by ID, save raw JSON to disk.

    `openalex_id` may be the short ID (e.g. W2741809807, A5089215617) or the
    full OpenAlex URL. DOI / ORCID / ROR identifiers are also accepted when
    prefixed (`doi:`, `orcid:`, `ror:`).
    """
    t0 = time.perf_counter()
    entity_type = (entity_type or "").strip().lower()
    if entity_type not in ENTITY_TYPES:
        return _error_response(
            "ValidationError",
            f"entity_type must be one of {sorted(ENTITY_TYPES)}; got {entity_type!r}",
        )
    openalex_id = (openalex_id or "").strip()
    if not openalex_id:
        return _error_response("ValidationError", "empty openalex_id")
    out_dir = str(out_dir or "").strip().rstrip(os.sep)
    if not out_dir:
        return _error_response("ValidationError", "empty out_dir")
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Strip URL prefix if user pasted the full OpenAlex URL
    if openalex_id.startswith("http"):
        openalex_id = openalex_id.rsplit("/", 1)[-1]

    sess = _session()
    try:
        resp = _http_get(sess, _build_url(entity_type, openalex_id), timeout=timeout)
    except requests.RequestException as e:
        return _error_response("NetworkError", str(e))
    if resp.status_code == 404:
        return _error_response("NotFound", f"openalex {entity_type}/{openalex_id} not found")
    if resp.status_code != 200:
        return _error_response("ApiError", f"openalex {resp.status_code}: {resp.text[:500]}")
    try:
        body = resp.json()
    except ValueError:
        return _error_response("ParseError", f"non-JSON response: {resp.text[:500]}")

    safe_id = openalex_id.replace("/", "_").replace(":", "_")
    out_path = os.path.join(out_dir, f"openalex_{entity_type}_{safe_id}.json")
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(body, f, ensure_ascii=False, indent=2)
    except OSError as e:
        return _error_response("IOError", f"failed to write response: {e}")

    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    title = body.get("title") or body.get("display_name") or ""
    meta = {
        "entity_type": entity_type,
        "openalex_id": body.get("id"),
        "display_name": title,
        "cited_by_count": body.get("cited_by_count"),
        "publication_year": body.get("publication_year"),
        "doi": body.get("doi"),
    }
    preview = f"OpenAlex {entity_type} entry: {body.get('id')}\n  {title[:200]}"
    return _success_response(out_path, preview, meta, elapsed_ms)


__all__ = [
    "BASE_URL",
    "ENTITY_TYPES",
    "download_openalex_entries_by_query",
    "download_openalex_entry_by_id",
]
