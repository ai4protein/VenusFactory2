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
    from mcp_tools.tools.search.deepsearch.response_utils import error_response, query_success_response

dotenv.load_dotenv()

logger = logging.getLogger(__name__)

def _tavily_search(query: str, max_results: int = 5, api_key: str = "") -> List[BaseDocument]:
    """Compatibility adapter for callers expecting LangChain documents."""
    import asyncio
    from ..web_provider import search_web
    result = asyncio.run(search_web(query, max_results, 'tavily'))
    return [BaseDocument(page_content=row['text'], metadata={**row, 'query': query}) for row in result['results']]


def query_tavily(query: str, max_results: int = 5) -> str:
    """Search Tavily through the shared bounded web provider."""
    from ..web_provider import query_web
    return query_web(query, max_results, 'tavily')


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
