"""PubMed batch-abstract fetch by PMID via NCBI efetch (XML).

Complements VF's existing `query_pubmed_tool` (keyword search): given a list of
PMIDs, fetch full title + authors + journal + structured abstract + DOI in one
efetch call, parse XML, write a normalized JSON to disk.

Adapted from google-deepmind/science-skills (Apache-2.0):
- skills/pubmed_database/scripts/pubmed_api.py (fetch_article_abstracts)
"""
import json
import os
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.tools.path_sanitizer import to_client_file_path


EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

_PREVIEW_LEN = 1500
_SOURCE = "PubMed (E-utilities)"
_MAX_PMIDS_PER_REQUEST = 200  # NCBI efetch handles up to ~10k; we cap for safety


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


def _text_or_none(elem: Optional[ET.Element]) -> Optional[str]:
    if elem is None:
        return None
    return "".join(elem.itertext()).strip() or None


def _parse_article(article: ET.Element) -> Dict[str, Any]:
    pmid = _text_or_none(article.find(".//PMID"))
    title = _text_or_none(article.find(".//ArticleTitle"))

    # Structured abstract — concatenate labeled sections.
    abstract_parts: List[str] = []
    for atxt in article.findall(".//Abstract/AbstractText"):
        label = atxt.attrib.get("Label", "").strip()
        body = "".join(atxt.itertext()).strip()
        if label:
            abstract_parts.append(f"{label}: {body}")
        elif body:
            abstract_parts.append(body)
    abstract = "\n".join(abstract_parts) or None

    journal = _text_or_none(article.find(".//Journal/Title"))
    year = _text_or_none(article.find(".//Journal/JournalIssue/PubDate/Year"))
    if year is None:
        year = _text_or_none(article.find(".//Journal/JournalIssue/PubDate/MedlineDate"))

    authors: List[str] = []
    for a in article.findall(".//AuthorList/Author"):
        last = _text_or_none(a.find("LastName")) or ""
        init = _text_or_none(a.find("Initials")) or ""
        collective = _text_or_none(a.find("CollectiveName"))
        if collective:
            authors.append(collective)
        elif last:
            authors.append(f"{last} {init}".strip())

    doi = None
    for eid in article.findall(".//ArticleIdList/ArticleId"):
        if (eid.attrib.get("IdType") or "").lower() == "doi":
            doi = (eid.text or "").strip()
            break

    return {
        "pmid": pmid,
        "title": title,
        "authors": authors,
        "journal": journal,
        "year": year,
        "doi": doi,
        "abstract": abstract,
    }


def download_pubmed_abstracts_by_pmids(
    pmids: List[str],
    out_path: str,
    timeout: int = 60,
) -> str:
    """Fetch full abstract + metadata for a batch of PubMed PMIDs.

    `out_path` is a file path (`.json`); contains a list of article dicts.
    """
    t0 = time.perf_counter()
    if not pmids:
        return _error_response("ValidationError", "empty pmids list")
    clean = [str(p).strip() for p in pmids if str(p).strip()]
    if not clean:
        return _error_response("ValidationError", "no non-empty PMIDs")
    if len(clean) > _MAX_PMIDS_PER_REQUEST:
        return _error_response(
            "ValidationError",
            f"too many PMIDs ({len(clean)}); max per call is {_MAX_PMIDS_PER_REQUEST}",
            suggestion="Split into chunks of ≤200 and call the tool repeatedly.",
        )
    out_path = str(out_path or "").strip()
    if not out_path:
        return _error_response("ValidationError", "empty out_path")
    out_dir = os.path.dirname(out_path) or "."
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    params: Dict[str, Any] = {
        "db": "pubmed",
        "id": ",".join(clean),
        "rettype": "abstract",
        "retmode": "xml",
    }
    api_key = os.environ.get("NCBI_API_KEY")
    if api_key:
        params["api_key"] = api_key
    email = os.environ.get("USER_EMAIL")
    if email:
        params["email"] = email

    sess = _session()
    try:
        resp = sess.get(f"{EUTILS_BASE}/efetch.fcgi", params=params, timeout=timeout)
    except requests.RequestException as e:
        return _error_response("NetworkError", str(e))
    if resp.status_code != 200:
        return _error_response("ApiError", f"efetch {resp.status_code}: {resp.text[:300]}")

    try:
        root = ET.fromstring(resp.text)
    except ET.ParseError as e:
        return _error_response("ParseError", f"failed to parse efetch XML: {e}")

    articles = [_parse_article(a) for a in root.iter("PubmedArticle")]
    if not articles:
        return _error_response(
            "NotFound",
            f"no PubmedArticle elements in response for {len(clean)} PMIDs",
            suggestion="Verify the PMIDs are valid via PubMed website.",
        )

    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"requested_pmids": clean, "articles": articles}, f, ensure_ascii=False, indent=2)
    except OSError as e:
        return _error_response("IOError", f"failed to write JSON: {e}")

    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    meta = {
        "requested_pmids": len(clean),
        "articles_returned": len(articles),
        "pmids_returned": [a["pmid"] for a in articles if a.get("pmid")],
    }
    preview_lines = [f"PubMed batch fetch — {len(articles)}/{len(clean)} articles returned"]
    for a in articles[:3]:
        preview_lines.append(f"  [{a.get('pmid')}] {(a.get('title') or '')[:120]}")
    return _success_response(out_path, "\n".join(preview_lines), meta, elapsed_ms)


__all__ = ["download_pubmed_abstracts_by_pmids"]
