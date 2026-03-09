"""
Literature search: arXiv, PubMed, bioRxiv, Semantic Scholar.
CLI: python src/tools/search/source/literature_search.py --query "..." [--max_results 5] [--source arxiv,pubmed,biorxiv,semantic_scholar]
"""
import json
import argparse
from datetime import datetime

import sys
import os

sys.path.append(os.getcwd())
try:
    from src.tools.search.deepsearch.arxiv_search import _arxiv_search
    from src.tools.search.deepsearch.biorxiv_search import _biorxiv_search
    from src.tools.search.deepsearch.pubmed_search import _pubmed_search
    from src.tools.search.deepsearch.semantic_scholar_search import _semantic_scholar_search
except ImportError:
    from ..deepsearch.arxiv_search import _arxiv_search
    from ..deepsearch.biorxiv_search import _biorxiv_search
    from ..deepsearch.pubmed_search import _pubmed_search
    from ..deepsearch.semantic_scholar_search import _semantic_scholar_search


def literature_search(
    query: str,
    max_results: int = 5,
    delay: float = 0.5,
    source: str = "pubmed",
) -> str:
    """
    Search literature using arXiv, PubMed, bioRxiv, Semantic Scholar.
    Returns JSON string of deduplicated results.
    """
    all_refs = []
    if "arxiv" in source:
        try:
            all_refs.extend(_arxiv_search(query, max_results=max_results))
        except Exception:
            pass
    if "pubmed" in source:
        try:
            all_refs.extend(_pubmed_search(query, max_results=max_results))
        except Exception:
            pass
    if "biorxiv" in source:
        try:
            all_refs.extend(_biorxiv_search(query, max_results=max_results))
        except Exception:
            pass
    if "semantic_scholar" in source:
        try:
            all_refs.extend(_semantic_scholar_search(query, max_results=max_results))
        except Exception:
            pass

    unified = []
    for item in all_refs:
        data_source = {}
        if hasattr(item, 'metadata') and isinstance(getattr(item, 'metadata'), dict):
            data_source = item.metadata
        elif isinstance(item, dict):
            data_source = item
        else:
            continue
        title = data_source.get("title") or ""
        url = data_source.get("url") or ""
        authors = data_source.get("authors") or []
        year = data_source.get("published") or data_source.get("published_date") or ""
        if isinstance(year, datetime):
            year = year.strftime('%Y-%m-%d')
        abstract = data_source.get("abstract") or ""
        src = data_source.get("source") or ""
        unified.append({
            "source": src, "title": title, "url": url,
            "authors": authors, "year": year, "abstract": abstract,
        })

    seen = set()
    final_refs = []
    for r in unified:
        key = (r.get("title", "").strip().lower(), (r.get("url", "")).strip().lower())
        if key in seen or not r.get("title"):
            continue
        seen.add(key)
        final_refs.append(r)
    return json.dumps(final_refs[:max_results], ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Literature search (arXiv, PubMed, bioRxiv, Semantic Scholar)")
    parser.add_argument("--query", type=str, default="protein structure prediction", help="Search query")
    parser.add_argument("--max_results", type=int, default=5, help="Max results per source")
    parser.add_argument("--source", type=str, default="arxiv,pubmed",
                        help="Comma-separated: arxiv,pubmed,biorxiv,semantic_scholar")
    args = parser.parse_args()
    result = literature_search(args.query, max_results=args.max_results, source=args.source)
    print(result)
