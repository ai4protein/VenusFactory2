import time
import requests
import json
from datetime import datetime
from web.utils.DeepSearch.arxiv_search import _arxiv_search
from web.utils.DeepSearch.biorxiv_search import _biorxiv_search
from web.utils.DeepSearch.pubmed_search import _pubmed_search
from web.utils.DeepSearch.senmatic_scholar_search import _semantic_scholar_search

def literature_search(query: str, max_results: int = 5, delay: float = 0.5, source: str = "arxiv") -> list[dict]:
    """
    Search literature using arXiv and PubMed APIs.

    Args:
        query: The query to search for.
        max_results: The maximum number of results to return from each source.
        delay: The delay between searches in seconds.
        source: The source to search for.
    Returns:
        A list of deduplicated search results.
    """
    all_refs = []
    
    if "arxiv" in source:
        # 1) Query arXiv
        try:
            arxiv_entries = _arxiv_search(query, max_results=max_results)
            all_refs.extend(arxiv_entries)
        except Exception:
            pass
    
    if "pubmed" in source:
        # 2) Query PubMed
        try:
            pubmed_entries = _pubmed_search(query, max_results=max_results)
            all_refs.extend(pubmed_entries)
        except Exception:
            pass
    
    if "biorxiv" in source:
        # 3) Query bioRxiv
        try:
            biorxiv_entries = _biorxiv_search(query, max_results=max_results)
            all_refs.extend(biorxiv_entries)
        except Exception:
            pass
    
    if "semantic_scholar" in source:
        # 4) Query Semantic Scholar
        try:
            semantic_scholar_entries = _semantic_scholar_search(query, max_results=max_results)
            all_refs.extend(semantic_scholar_entries)
        except Exception:
            pass
    
    # Normalize all entries to unified format
    unified = []
    for item in all_refs:
        # Check if item is a Document object (has metadata attribute) or a dict
        data_source = {}
        if hasattr(item, 'metadata') and isinstance(getattr(item, 'metadata'), dict):
             data_source = item.metadata
        elif isinstance(item, dict):
             data_source = item
        else:
            continue
        
        # Extract common fields
        # Extract common fields
        title = data_source.get("title") or ""
        url = data_source.get("url") or ""
        authors = data_source.get("authors") or []
        year = data_source.get("published") or data_source.get("published_date") or ""
        if isinstance(year, datetime):
            year = year.strftime('%Y-%m-%d')
            
        abstract = data_source.get("abstract") or ""
        source = data_source.get("source") or ""

        unified.append({
            "source": source,
            "title": title,
            "url": url,
            "authors": authors,
            "year": year,
            "abstract": abstract,
        })
    
    # Deduplicate by (title, url or doi)
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
    results = literature_search(query, max_results=2, source="semantic_scholar")
    print("results: ", results)

