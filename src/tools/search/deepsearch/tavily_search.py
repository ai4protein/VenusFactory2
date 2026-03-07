import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import List

import dotenv
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

dotenv.load_dotenv()

logger = logging.getLogger(__name__)

def _tavily_search(query: str, max_results: int = 5, api_key: str = "") -> List[BaseDocument]:
    """
    Execute Tavily search query using direct API call.
    """
    # Try to find API key from args or env
    token = os.getenv("TAVILY_API_KEY")
    
    if not token:
        logger.error("No Tavily API key provided. Set TAVILY_API_KEY env var or pass api_key.")
        return []

    try:
        url = "https://api.tavily.com/search"
        payload = {
            "api_key": token,
            "query": query,
            "max_results": max_results,
            # We can add more params like 'search_depth': 'advanced' if needed
        }
        
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        results = data.get("results", [])
        docs = []
        
        for res in results:
            metadata = {
                "title": res.get("title"),
                "url": res.get("url"),
                "source": "tavily",
                "query": query,
                # Tavily also returns score, published_date, etc.
                "score": res.get("score", "Tavily")
            }
            content = res.get("content", "")
            docs.append(BaseDocument(page_content=content, metadata=metadata))
            
        return docs
        
    except Exception as e:
        logger.error(f"Tavily search failed: {e}")
        return []


def query_tavily(query: str, max_results: int = 5) -> str:
    """
    Execute Tavily search query.
    Returns rich JSON: status, content, execution_context.
    """
    t0 = time.perf_counter()
    try:
        docs = _tavily_search(query, max_results)
        results = [doc.metadata if hasattr(doc, 'metadata') else doc for doc in docs]
        payload = {"query": query, "count": len(results), "results": results}
        content = json.dumps(payload, ensure_ascii=False, default=str)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        return query_success_response(content, query_time_ms=elapsed_ms, source="tavily")
    except Exception as e:
        return error_response("QueryError", str(e), suggestion="Check network connection or API keys.")


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Tavily Search")
    parser.add_argument("--query", type=str, default="transformer", help="Search query")
    parser.add_argument("--max_results", type=int, default=2, help="Max results")
    args = parser.parse_args()
    
    print(f"=== query_tavily(query='{args.query}', max_results={args.max_results}) ===")
    res = query_tavily(args.query, max_results=args.max_results)
    print(res)
