"""
Web search: DuckDuckGo, Tavily.
CLI: python src/tools/search/source/web_search.py --query "..." [--max_results 5] [--source duckduckgo,tavily]
"""
import json
import argparse
from datetime import datetime

import sys
import os
sys.path.append(os.getcwd())
from src.tools.search.deepsearch.duckduckgo_search import _duckduckgo_search
from src.tools.search.deepsearch.tavily_search import _tavily_search


def web_search(
    query: str,
    max_results: int = 5,
    source: str = "duckduckgo",
) -> str:
    """Search online using DuckDuckGo and/or Tavily. Returns JSON string of results."""
    all_refs = []
    if "duckduckgo" in source:
        try:
            duckduckgo_json = _duckduckgo_search(query, max_results=max_results)
            if duckduckgo_json:
                duckduckgo_entries = json.loads(duckduckgo_json)
                if isinstance(duckduckgo_entries, list):
                    all_refs.extend(duckduckgo_entries)
                elif isinstance(duckduckgo_entries, dict) and "results" in duckduckgo_entries:
                    all_refs.extend(duckduckgo_entries["results"])
        except Exception:
            pass
    if "tavily" in source:
        try:
            tavily_json = _tavily_search(query, max_results=max_results)
            if tavily_json:
                tavily_entries = json.loads(tavily_json)
                if isinstance(tavily_entries, list):
                    all_refs.extend(tavily_entries)
        except Exception:
            pass

    unified = []
    for item in all_refs:
        data_source = {}
        content = ""
        if hasattr(item, 'metadata') and isinstance(getattr(item, 'metadata'), dict):
            data_source = item.metadata
            if hasattr(item, 'page_content'):
                content = item.page_content
        elif isinstance(item, dict):
            if "metadata" in item and isinstance(item["metadata"], dict):
                data_source = item["metadata"]
                content = item.get("page_content", "")
            else:
                data_source = item
                content = item.get("page_content", "")
        else:
            continue
        title = data_source.get("title") or ""
        url = data_source.get("url") or ""
        authors = data_source.get("authors") or []
        year = data_source.get("published") or data_source.get("published_date") or ""
        if isinstance(year, datetime):
            year = year.strftime('%Y-%m-%d')
        abstract = data_source.get("abstract") or content
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
    parser = argparse.ArgumentParser(description="Web search (DuckDuckGo, Tavily)")
    parser.add_argument("--query", type=str, default="protein language model", help="Search query")
    parser.add_argument("--max_results", type=int, default=5, help="Max results")
    parser.add_argument("--source", type=str, default="duckduckgo", help="duckduckgo and/or tavily")
    args = parser.parse_args()
    result = web_search(args.query, max_results=args.max_results, source=args.source)
    print(result)
