"""
Dataset search: GitHub, Hugging Face.
CLI: python src/tools/search/source/dataset_search.py --query "..." [--max_results 5] [--source github,hugging_face]
"""
import sys
import os
sys.path.append(os.getcwd())
import json
import argparse

from src.tools.search.deepsearch.github_search import _github_search
from src.tools.search.deepsearch.hugging_face_search import _hugging_face_search


def dataset_search(
    query: str,
    max_results: int = 5,
    delay: float = 0.5,
    source: str = "github",
) -> str:
    """Search datasets via GitHub and/or Hugging Face. Returns JSON string of results."""
    all_datasets = []
    if "github" in source:
        try:
            all_datasets.extend(_github_search(query, max_results=max_results))
        except Exception:
            pass
    if "hugging_face" in source:
        try:
            all_datasets.extend(_hugging_face_search(query, max_results=max_results))
        except Exception:
            pass

    unified = []
    for item in all_datasets:
        data_source = {}
        if hasattr(item, 'metadata') and isinstance(getattr(item, 'metadata'), dict):
            data_source = item.metadata
        elif isinstance(item, dict):
            data_source = item
        else:
            continue
        title = data_source.get("title") or ""
        url = data_source.get("url") or data_source.get("pdf_url") or ""
        abstract = data_source.get("summary") or data_source.get("abstract") or ""
        src = data_source.get("source")
        if not abstract and hasattr(item, 'page_content'):
            abstract = item.page_content
        unified.append({"source": src, "title": title, "url": url, "abstract": abstract})

    seen = set()
    final_datasets = []
    for d in unified:
        key = (d.get("title", "").strip().lower(), (d.get("url", "") or d.get("doi", "")).strip().lower())
        if key in seen or not d.get("title"):
            continue
        seen.add(key)
        final_datasets.append(d)
    return json.dumps(final_datasets[:max_results * 2], ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset search (GitHub, Hugging Face)")
    parser.add_argument("--query", type=str, default="protein dataset", help="Search query")
    parser.add_argument("--max_results", type=int, default=5, help="Max results per source")
    parser.add_argument("--source", type=str, default="github,hugging_face", help="github and/or hugging_face")
    args = parser.parse_args()
    result = dataset_search(args.query, max_results=args.max_results, source=args.source)
    print(result)
