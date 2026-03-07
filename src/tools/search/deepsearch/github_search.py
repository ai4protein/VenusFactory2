import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import List

import requests

try:
    from langchain_core.documents import Document as BaseDocument
except ImportError:
    class BaseDocument:
        def __init__(self, page_content, metadata):
            self.page_content = page_content
            self.metadata = metadata

try:
    from .response_utils import error_response, query_success_response
except ImportError:
    _dir = Path(__file__).resolve().parent
    if _dir.name == "deepsearch" and str(_dir.parents[3]) not in sys.path:
        sys.path.insert(0, str(_dir.parents[3]))
    from src.tools.search.deepsearch.response_utils import error_response, query_success_response

def _github_search(query: str, max_results: int = 5, api_key: str = "") -> List[BaseDocument]:
    """
    Execute GitHub search query for repositories.
    
    Args:
        query (str): The search term.
        max_results (int): Number of repositories to return.
        api_key (str): Optional GitHub Personal Access Token.
        
    Returns:
        List[BaseDocument]: A list of documents representing GitHub repositories.
    """
    try:
        url = "https://api.github.com/search/repositories"
        headers = {
            "Accept": "application/vnd.github.v3+json"
        }
        if api_key:
            headers["Authorization"] = f"token {api_key}"
            
        params = {
            "q": query,
            "sort": "stars", # Default to sorting by stars as a proxy for "relative top"
            "order": "desc",
            "per_page": max_results
        }
        
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()
        
        items = data.get("items", [])
        docs = []
        
        for item in items:
            description = item.get("description") or "No description provided."
            content = (
                f"Repository: {item.get('full_name')}\n"
                f"Description: {description}\n"
                f"Language: {item.get('language')}\n"
                f"Stars: {item.get('stargazers_count')}\n"
                f"Forks: {item.get('forks_count')}\n"
                f"Updated: {item.get('updated_at')}"
            )
            
            metadata = {
                "source": "github",
                "title": item.get("full_name"),
                "url": item.get("html_url"),
                "query": query,
                "stars": item.get("stargazers_count"),
                "language": item.get("language"),
                "license": item.get("license", {}).get("name") if item.get("license") else None,
                "topics": item.get("topics", [])
            }
            
            docs.append(BaseDocument(page_content=content, metadata=metadata))
            
        return docs
        
    except requests.exceptions.RequestException as e:
        return []
    except Exception as e:
        return []


def query_github(query: str, max_results: int = 5, api_key: str = "") -> str:
    """
    Execute GitHub search query for repositories.
    Returns rich JSON: status, content, execution_context.
    """
    t0 = time.perf_counter()
    try:
        docs = _github_search(query, max_results, api_key)
        results = [doc.metadata if hasattr(doc, 'metadata') else doc for doc in docs]
        payload = {"query": query, "count": len(results), "results": results}
        content = json.dumps(payload, ensure_ascii=False, default=str)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        return query_success_response(content, query_time_ms=elapsed_ms, source="github")
    except Exception as e:
        return error_response("QueryError", str(e), suggestion="Check network connection or GitHub API keys.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="GitHub Search")
    parser.add_argument("--query", type=str, default="machine learning", help="Search query")
    parser.add_argument("--max_results", type=int, default=3, help="Max results")
    args = parser.parse_args()
    
    print(f"=== query_github(query='{args.query}', max_results={args.max_results}) ===")
    res = query_github(args.query, max_results=args.max_results)
    print(res)
