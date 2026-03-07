"""
arXiv search via the official API (http://export.arxiv.org/api/query).
No LangChain dependency; returns list of dicts compatible with literature_search.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, List
from xml.etree import ElementTree as ET

import requests

try:
    from .response_utils import error_response, query_success_response
except ImportError:
    _dir = Path(__file__).resolve().parent
    if _dir.name == "deepsearch" and str(_dir.parents[3]) not in sys.path:
        sys.path.insert(0, str(_dir.parents[3]))
    from src.tools.search.deepsearch.response_utils import error_response, query_success_response


ATOM_NS = "http://www.w3.org/2005/Atom"


def _tag(name: str) -> str:
    return f"{{{ATOM_NS}}}{name}"


def _arxiv_search(
    query: str,
    max_results: int = 5,
    max_content_length: int = 10000,
) -> List[dict[str, Any]]:
    """
    Search arXiv using the official export API.

    Args:
        query: Search query string.
        max_results: Maximum number of results to return.
        max_content_length: Max length for abstract text (truncate if longer).

    Returns:
        List of dicts with keys: source, title, url, authors, published, abstract.
        Compatible with literature_search (used as item.metadata or as plain dict).
    """
    out: List[dict[str, Any]] = []
    try:
        resp = requests.get(
            "http://export.arxiv.org/api/query",
            params={
                "search_query": f"all:{query}",
                "start": 0,
                "max_results": max_results,
                "sortBy": "relevance",
                "sortOrder": "descending",
            },
            timeout=15,
        )
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
    except Exception:
        return out

    for entry in root.findall(_tag("entry")):
        title_el = entry.find(_tag("title"))
        summary_el = entry.find(_tag("summary"))
        published_el = entry.find(_tag("published"))
        id_el = entry.find(_tag("id"))

        title = (title_el.text or "").strip().replace("\n", " ")
        abstract = (summary_el.text or "").strip().replace("\n", " ") if summary_el is not None else ""
        published = (published_el.text or "").strip() if published_el is not None else ""
        entry_id = (id_el.text or "").strip() if id_el is not None else ""

        if max_content_length and len(abstract) > max_content_length:
            abstract = abstract[:max_content_length] + "..."

        authors: List[str] = []
        for author in entry.findall(_tag("author")):
            name_el = author.find(_tag("name"))
            if name_el is not None and name_el.text:
                authors.append(name_el.text.strip())

        pdf_url = ""
        for link in entry.findall(_tag("link")):
            if link.get("title") == "pdf":
                pdf_url = link.get("href", "")
                break
        url = pdf_url or entry_id

        if not title:
            continue
        out.append({
            "source": "arxiv",
            "title": title,
            "url": url,
            "authors": authors,
            "published": published,
            "published_date": published,
            "abstract": abstract,
        })
    return out


def query_arxiv(
    query: str,
    max_results: int = 5,
    max_content_length: int = 10000,
) -> str:
    """
    Query arXiv using the official export API.
    Returns rich JSON: status, content, execution_context.
    """
    t0 = time.perf_counter()
    try:
        results = _arxiv_search(query, max_results, max_content_length)
        payload = {"query": query, "count": len(results), "results": results}
        content = json.dumps(payload, ensure_ascii=False)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        return query_success_response(content, query_time_ms=elapsed_ms, source="arxiv")
    except Exception as e:
        return error_response("QueryError", str(e), suggestion="Check network connection or arXiv API status.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="arXiv literature search")
    parser.add_argument("--query", type=str, default="protein structure prediction", help="Search query")
    parser.add_argument("--max_results", type=int, default=5, help="Max results")
    args = parser.parse_args()
    
    print(f"=== query_arxiv(query='{args.query}', max_results={args.max_results}) ===")
    res = query_arxiv(args.query, max_results=args.max_results)
    print(res)

