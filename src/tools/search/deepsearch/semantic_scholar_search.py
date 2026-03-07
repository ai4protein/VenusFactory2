import json
import logging
import os
import random
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

import requests

try:
    from langchain_core.documents import Document as BaseDocument
except ImportError:
    class BaseDocument:
        def __init__(self, page_content, metadata=None):
            self.page_content = page_content
            self.metadata = metadata or {}

try:
    from .response_utils import error_response, query_success_response
except ImportError:
    _dir = Path(__file__).resolve().parent
    if _dir.name == "deepsearch" and str(_dir.parents[3]) not in sys.path:
        sys.path.insert(0, str(_dir.parents[3]))
    from src.tools.search.deepsearch.response_utils import error_response, query_success_response

logger = logging.getLogger(__name__)

class SemanticScholarAPIWrapper:
    """Wrapper for Semantic Scholar API."""

    SEMANTIC_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
    SEMANTIC_BASE_URL = "https://api.semanticscholar.org/graph/v1"
    BROWSERS = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
    ]

    def __init__(self):
        self._setup_session()

    def _setup_session(self):
        """Initialize session with random user agent"""
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": random.choice(self.BROWSERS),
                "Accept": "text/html,application/xhtml+xml",
                "Accept-Language": "en-US,en;q=0.9",
            }
        )

    def _parse_date(self, date_str: str) -> Optional[str]:
        """Parse date from Semantic Scholar format and return string."""
        if not date_str:
            return None
        try:
            # Check if it's already in YYYY-MM-DD
            datetime.strptime(date_str.strip(), "%Y-%m-%d")
            return date_str.strip()
        except ValueError:
            return None

    def _extract_url_from_disclaimer(self, disclaimer: str) -> str:
        """Extract URL from disclaimer text"""
        url_patterns = [
            r'https?://[^\s,)]+',  # Basic HTTP/HTTPS URL
            r'https?://arxiv\.org/abs/[^\s,)]+',  # arXiv link
            r'https?://[^\s,)]*\.pdf',  # PDF link
        ]
        
        all_urls = []
        for pattern in url_patterns:
            matches = re.findall(pattern, disclaimer)
            all_urls.extend(matches)
        
        if not all_urls:
            return ""
        
        doi_urls = [url for url in all_urls if 'doi.org' in url]
        if doi_urls:
            return doi_urls[0]
        
        non_unpaywall_urls = [url for url in all_urls if 'unpaywall.org' not in url]
        if non_unpaywall_urls:
            url = non_unpaywall_urls[0]
            if 'arxiv.org/abs/' in url:
                return url.replace('/abs/', '/pdf/')
            return url
        
        if all_urls:
            url = all_urls[0]
            if 'arxiv.org/abs/' in url:
                return url.replace('/abs/', '/pdf/')
            return url
        
        return ""

    def get_api_key(self) -> Optional[str]:
        """Get the Semantic Scholar API key from environment variables."""
        api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
        if not api_key or api_key.strip() == "":
            logger.warning("No SEMANTIC_SCHOLAR_API_KEY set or it's empty. Using unauthenticated access with lower rate limits.")
            return None
        return api_key.strip()
    
    def request_api(self, path: str, params: dict) -> Any:
        """Make a request to the Semantic Scholar API."""
        max_retries = 2
        retry_delay = 1
        max_wait = 5

        for attempt in range(max_retries):
            try:
                api_key = self.get_api_key()
                headers = {"x-api-key": api_key} if api_key else {}
                url = f"{self.SEMANTIC_BASE_URL}/{path}"
                response = self.session.get(url, params=params, headers=headers, timeout=15)

                if response.status_code == 429:
                    if attempt < max_retries - 1:
                        wait_time = min(retry_delay * (2 ** attempt), max_wait)
                        logger.warning(f"Rate limited (429). Waiting {wait_time} seconds before retry {attempt + 1}/{max_retries}")
                        time.sleep(wait_time)
                        continue
                    else:
                        logger.error(f"Rate limited (429) after {max_retries} attempts.")
                        return None

                response.raise_for_status()
                return response.json()

            except requests.exceptions.RequestException as e:
                if getattr(e.response, "status_code", None) == 429 and attempt < max_retries - 1:
                    wait_time = min(retry_delay * (2 ** attempt), max_wait)
                    logger.warning(f"Rate limited (429). Waiting {wait_time} seconds before retry {attempt + 1}/{max_retries}")
                    time.sleep(wait_time)
                    continue
                logger.error(f"Error requesting API: {e}")
                return None

        return None

    def search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Search Semantic Scholar and return raw results."""
        fields = ["title", "abstract", "year", "citationCount", "authors", "url", "publicationDate", "openAccessPdf", "externalIds", "fieldsOfStudy"]
        params = {
            "query": query,
            "limit": max_results,
            "fields": ",".join(fields),
        }
        
        data = self.request_api("paper/search", params)
        if not data or 'data' not in data:
            return []
            
        processed_results = []
        for item in data['data']:
            try:
                # Extract PDF URL
                pdf_url = ""
                if item.get('openAccessPdf'):
                    open_access_pdf = item['openAccessPdf']
                    if open_access_pdf.get('url'):
                        pdf_url = open_access_pdf['url']
                    elif open_access_pdf.get('disclaimer'):
                        pdf_url = self._extract_url_from_disclaimer(open_access_pdf['disclaimer'])

                # Format authors
                authors = [author['name'] for author in item.get('authors', [])]
                
                result = {
                    "title": item.get('title'),
                    "abstract": item.get('abstract', ''),
                    "url": item.get('url', ''),
                    "pdf_url": pdf_url,
                    "published_date": self._parse_date(item.get('publicationDate', '')),
                    "authors": authors,
                    "doi": item.get('externalIds', {}).get('DOI', ''),
                    "citation_count": item.get('citationCount', 0),
                    "categories": item.get('fieldsOfStudy', [])
                }
                processed_results.append(result)
            except Exception as e:
                logger.warning(f"Error processing paper item: {e}")
                continue
                
        return processed_results

def _semantic_scholar_search(query: str, max_results: int = 5, max_content_length: int = 10000) -> List[BaseDocument]:
    """
    Execute Semantic Scholar search query.

    Args:
        query (str): Search query.
        max_results (int, optional): Maximum number of results to return. Defaults to 5.
        max_content_length (int, optional): Maximum length of document content to include. Defaults to 10000.

    Returns:
        List[BaseDocument]: List of documents retrieved from Semantic Scholar.
    """
    try:
        wrapper = SemanticScholarAPIWrapper()
        results = wrapper.search(query, max_results=max_results)
        
        docs = []
        for result in results:
            metadata = {
                "source": "semantic_scholar",
                "title": result.get("title"),
                "url": result.get("url"),
                "published_date": result.get("published_date"),
                "authors": result.get("authors"),
                "abstract": result.get("abstract")
            }
            
            docs.append(BaseDocument(page_content=result.get("abstract", ""), metadata=metadata))
            
        return docs
    except Exception as e:
        logger.error(f"Semantic Scholar search failed: {e}")
        return []


def query_semantic_scholar(query: str, max_results: int = 5) -> str:
    """
    Execute Semantic Scholar search query.
    Returns rich JSON: status, content, execution_context.
    """
    t0 = time.perf_counter()
    try:
        docs = _semantic_scholar_search(query, max_results)
        results = [doc.metadata if hasattr(doc, 'metadata') else doc for doc in docs]
        payload = {"query": query, "count": len(results), "results": results}
        content = json.dumps(payload, ensure_ascii=False, default=str)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        return query_success_response(content, query_time_ms=elapsed_ms, source="semantic_scholar")
    except Exception as e:
        return error_response("QueryError", str(e), suggestion="Check network connection or Semantic Scholar API limits.")


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Semantic Scholar Search")
    parser.add_argument("--query", type=str, default="transformers", help="Search query")
    parser.add_argument("--max_results", type=int, default=2, help="Max results")
    args = parser.parse_args()
    
    print(f"=== query_semantic_scholar(query='{args.query}', max_results={args.max_results}) ===")
    res = query_semantic_scholar(args.query, max_results=args.max_results)
    print(res)
