import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any

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



BASE_URL = "https://api.biorxiv.org/details/biorxiv"
def _biorxiv_search(query: str, max_results: int = 5, days: int = 30) -> List[BaseDocument]:
    """
    Search for papers on bioRxiv by category within the last N days.

    Args:
        query (str): Search query string (must be in English)
        max_results (int): Maximum number of results to return
        days (int): Number of days to search within

    Returns:
        List of BaseDocument objects containing paper information
    """

    # Calculate date range: last N days
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    
    # Format category: lowercase and replace spaces with underscores
    category = query.lower().replace(' ', '_')
    
    papers = []
    cursor = 0
    while len(papers) < max_results:
        url = f"{BASE_URL}/{start_date}/{end_date}/{cursor}"
        if category:
            url += f"?category={category}"
        tries = 0
        while tries < 3:
            try:
                response = requests.get(url)
                response.raise_for_status()
                data = response.json()
                collection = data.get('collection', [])
                for item in collection:
                    try:
                        date = datetime.strptime(item['date'], '%Y-%m-%d')
                        papers.append(BaseDocument(
                            page_content=item['abstract'],
                            metadata={
                                "source": "biorxiv",
                                "title": item['title'],
                                "url": f"https://www.biorxiv.org/content/{item['doi']}v{item.get('version', '1')}",
                                "published_date": date,
                                "authors": item['authors'].split('; '),
                                "abstract": item['abstract']
                            }
                        ))
                    except Exception as e:
                        print(f"Error parsing bioRxiv entry: {e}")
                if len(collection) < 100:
                    break  # No more results
                cursor += 100
                break  # Exit retry loop on success
            except requests.exceptions.RequestException as e:
                tries += 1
                if tries == 3:
                    print(f"Failed to connect to bioRxiv API after {tries} attempts: {e}")
                    break
                print(f"Attempt {tries} failed, retrying...")
        else:
            continue
        break

    return papers[:max_results]


def query_biorxiv(query: str, max_results: int = 5, days: int = 30) -> str:
    """
    Search for papers on bioRxiv.
    Returns rich JSON: status, content, execution_context.
    """
    t0 = time.perf_counter()
    try:
        docs = _biorxiv_search(query, max_results, days)
        # Convert documents to dicts for JSON serialization
        results = [doc.metadata if hasattr(doc, 'metadata') else doc for doc in docs]
        payload = {"query": query, "count": len(results), "results": results}
        content = json.dumps(payload, ensure_ascii=False, default=str)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        return query_success_response(content, query_time_ms=elapsed_ms, source="biorxiv")
    except Exception as e:
        return error_response("QueryError", str(e), suggestion="Check network connection or bioRxiv API status.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="bioRxiv literature search")
    parser.add_argument("--query", type=str, default="bioinformatics", help="Search query (category)")
    parser.add_argument("--max_results", type=int, default=5, help="Max results")
    args = parser.parse_args()
    
    print(f"=== query_biorxiv(query='{args.query}', max_results={args.max_results}) ===")
    res = query_biorxiv(args.query, max_results=args.max_results)
    print(res)
