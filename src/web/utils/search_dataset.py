import time
import requests
import json
from web.utils.DeepSearch.github_search import _github_search
from web.utils.DeepSearch.hugging_face_search import _hugging_face_search

def dataset_search(query: str, max_results: int = 5, delay: float = 0.5, source: str = "github") -> list[dict]:
    """
    Search datasets using GitHub and Hugging Face APIs.

    Args:
        query: The query to search for.
        max_results: The maximum number of results to return from each source.
        delay: The delay between searches in seconds.

    Returns:
        A list of deduplicated search results.
    """

    all_datasets = []

    if "github" in source:
        # 1) Query GitHub
        try:
            github_datasets = _github_search(query, max_results=max_results)
            all_datasets.extend(github_datasets)
        except Exception:
            pass

    if "hugging_face" in source:
        # 2) Query Hugging Face
        try:
            hugging_face_datasets = _hugging_face_search(query, max_results=max_results)
            all_datasets.extend(hugging_face_datasets)
        except Exception:
            pass

    # Normalize all entries to unified format
    unified = []
    for item in all_datasets:
        # Check if item is a Document object (has metadata attribute) or a dict
        data_source = {}
        if hasattr(item, 'metadata') and isinstance(getattr(item, 'metadata'), dict):
             data_source = item.metadata
        elif isinstance(item, dict):
             data_source = item
        else:
            continue

        # Extract common fields
        title = data_source.get("title") or ""
        url = data_source.get("url") or data_source.get("pdf_url") or ""
        abstract = data_source.get("summary") or data_source.get("abstract") or ""
        source = data_source.get("source")

        # Using page_content as abstract fallback if abstract is empty
        if not abstract and hasattr(item, 'page_content'):
            abstract = item.page_content

        unified.append({
            "source": source,
            "title": title,
            "url": url,
            "abstract": abstract,
        })

    # Deduplicate by (title, url)
    seen = set()
    final_datasets = []
    for d in unified:
        key = (d.get("title", "").strip().lower(), (d.get("url", "") or d.get("doi", "")).strip().lower())
        if key in seen or not d.get("title"):
            continue
        seen.add(key)
        final_datasets.append(d)

    #  转换成JSON格式返回
    return json.dumps(final_datasets[:max_results * 2], ensure_ascii=False)


if __name__ == "__main__":
    query = sys.argv[1] if len(sys.argv) > 1 else "genomic data"
    results = dataset_search(query, max_results=5)
    
    print(f"Search results for: {query}")
    print(results)

