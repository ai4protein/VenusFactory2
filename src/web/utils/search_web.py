import time
import requests
import json
from datetime import datetime
from web.utils.DeepSearch.duckduckgo_search import _duckduckgo_search
from web.utils.DeepSearch.tavily_search import _tavily_search

def web_search(query: str, max_results: int = 5, source: str = "duckduckgo") -> str:
    """
    Search online using duckduckgo and tavily APIs.

    Args:
        query: The query to search for.
        max_results: The maximum number of results to return from each source.
        source: The source to search for.
    Returns:
        A list of deduplicated search results.
    """
    all_refs = []
    
    if "duckduckgo" in source:
        # 1) Query duckduckgo
        try:
            duckduckgo_json = _duckduckgo_search(query, max_results=max_results)
            if duckduckgo_json:
                try:
                    duckduckgo_entries = json.loads(duckduckgo_json)
                    if isinstance(duckduckgo_entries, list):
                        all_refs.extend(duckduckgo_entries)
                    elif isinstance(duckduckgo_entries, dict) and "results" in duckduckgo_entries:
                        all_refs.extend(duckduckgo_entries["results"])
                except json.JSONDecodeError:
                    print("Error decoding DuckDuckGo JSON")
        except Exception as e:
            print(f"DuckDuckGo search error: {e}")
            pass
    
    if "tavily" in source:
        # 2) Query tavily
        try:
            tavily_json = _tavily_search(query, max_results=max_results)
            if tavily_json:
                try:
                    tavily_entries = json.loads(tavily_json)
                    if isinstance(tavily_entries, list):
                        all_refs.extend(tavily_entries)
                except json.JSONDecodeError:
                    print("Error decoding Tavily JSON")
        except Exception as e:
            print(f"Tavily search error: {e}")
            pass
    
    # Normalize all entries to unified format
    unified = []
    for item in all_refs:
        # Check if item is a Document object (has metadata attribute) or a dict
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
        
        # Extract common fields
        title = data_source.get("title") or ""
        url = data_source.get("url") or ""
        authors = data_source.get("authors") or []
        year = data_source.get("published") or data_source.get("published_date") or ""
        if isinstance(year, datetime):
            year = year.strftime('%Y-%m-%d')
            
        abstract = data_source.get("abstract") or ""
        if not abstract and content:
            abstract = content
            
        source = data_source.get("source") or ""

        unified.append({
            "source": source,
            "title": title,
            "url": url,
            "authors": authors,
            "year": year,
            "abstract": abstract,
        })
    
    # Deduplicate by (title, url)
    seen = set()
    final_refs = []
    for r in unified:
        key = (r.get("title", "").strip().lower(), (r.get("url", "")).strip().lower())
        if key in seen or not r.get("title"):
            continue
        seen.add(key)
        final_refs.append(r)
    
    # 转换成JSON格式返回
    return json.dumps(final_refs[:max_results], ensure_ascii=False)


if __name__ == "__main__":
    # Example usage:
    query = "transformer"
    results = web_search(query, max_results=2, source="duckduckgo")
    print("results: ", results)
    results = web_search(query, max_results=2, source="tavily")
    print("results: ", results)
