import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import List, Optional

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

from scholarly import scholarly, ProxyGenerator

logger = logging.getLogger(__name__)

def _google_scholar_search(query: str, max_results: int = 5, api_key: str = "") -> List[BaseDocument]:
    """
    Execute Google Scholar search using the scholarly library.
    
    Args:
        query (str): Search query.
        max_results (int, optional): Maximum number of results to return. Defaults to 5.
        api_key (str, optional): Not used for scholarly, kept for interface compatibility.
    
    Returns:
        List[BaseDocument]: List of documents retrieved from Google Scholar.
    """
    papers = []
    
    try:
        # Construct the proxy generator
        # Using FreeProxies to bypass simple IP blocks
        logger.info("Attempting to configure FreeProxies for Google Scholar...")
        try:
            pg = ProxyGenerator()
            # success = pg.FreeProxies() # FreeProxies can be unreliable, sometimes ScraperAPI/Tor is better if available
            # If FreeProxies fails or is too slow, we might default to direct or handle error
            # For now enabling FreeProxies as requested by logic flow
            pg.FreeProxies()
            scholarly.use_proxy(pg)
            logger.info("FreeProxies configured successfully.")
        except Exception as proxy_error:
            logger.warning(f"Failed to setup FreeProxies: {proxy_error}. Continuing without proxy (might fail if blocked).")

        search_query_results = scholarly.search_pubs(query)
        
        for _ in range(max_results):
            try:
                item = next(search_query_results)
                
                # Extract data from scholarly result
                bib = item.get('bib', {})
                title = bib.get('title', 'No Title')
                authors = bib.get('author', [])
                year = bib.get('pub_year', 'Unknown')
                abstract = bib.get('abstract', '')
                url = item.get('pub_url', '')
                
                metadata = {
                    "title": title,
                    "url": url,
                    "authors": authors,
                    "year": year,
                    "abstract": abstract,
                    "query": query,
                }
                
                content = f"Title: {title}\nURL: {url}\nAuthors: {', '.join(authors) if isinstance(authors, list) else authors}\nYear: {year}\nAbstract: {abstract}"
                
                papers.append(BaseDocument(page_content=content, metadata=metadata))
                
                # Be nice to Google
                time.sleep(random.uniform(1.0, 3.0))
                
            except StopIteration:
                break
                
    except Exception as e:
        logger.error(f"Error during Google Scholar search: {e}")
        # If we hit a CAPTCHA or blocking issue, scholarly might raise an exception.
        # usually MaxTriesExceededException
        pass

    return papers


def query_google_scholar(query: str, max_results: int = 5) -> str:
    """
    Execute Google Scholar search.
    Returns rich JSON: status, content, execution_context.
    """
    t0 = time.perf_counter()
    try:
        docs = _google_scholar_search(query, max_results)
        results = [doc.metadata if hasattr(doc, 'metadata') else doc for doc in docs]
        payload = {"query": query, "count": len(results), "results": results}
        content = json.dumps(payload, ensure_ascii=False, default=str)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        return query_success_response(content, query_time_ms=elapsed_ms, source="google_scholar")
    except Exception as e:
        return error_response("QueryError", str(e), suggestion="Check network connection or scholarly proxy.")


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Google Scholar Search")
    parser.add_argument("--query", type=str, default="alphafold", help="Search query")
    parser.add_argument("--max_results", type=int, default=2, help="Max results")
    args = parser.parse_args()
    
    print(f"=== query_google_scholar(query='{args.query}', max_results={args.max_results}) ===")
    res = query_google_scholar(args.query, max_results=args.max_results)
    print(res)

