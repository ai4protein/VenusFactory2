import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Annotated
from xml.etree import ElementTree as ET

import requests
from langchain_community.utilities import PubMedAPIWrapper

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

from xml.etree import ElementTree as ET

SEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
FETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
def _pubmed_search(query: str,
                    max_results: int = 5,
                    max_content_length: int = 10000,
                    api_key: str = "") -> List[BaseDocument]:
    """
    Execute PubMed search query.

    Args:
        query: Search query string (must be in English)
        max_results: Maximum number of results to return
        max_content_length: Maximum content length per BaseDocument

    Returns:
        List of BaseDocument objects containing paper information
    """
    try:
        search_params = {
            'db': 'pubmed',
            'term': query,
            'retmax': max_results,
            'retmode': 'xml'
        }

        search_response = requests.get(SEARCH_URL, params=search_params, timeout=15)
        search_response.raise_for_status()
        search_root = ET.fromstring(search_response.content)
        # NCBI XML may use default namespace (e.g. xmlns="http://..."); support both
        def _ns(tag):
            if tag.startswith("{") and "}" in tag:
                return tag[1 : tag.index("}") + 1]
            return ""

        def _find_all(root, path):
            out = root.findall(path)
            if out:
                return out
            ns = _ns(root.tag)
            if ns:
                # e.g. .//Id -> .//{http://...}Id
                path_ns = path.replace(".//", f".//{ns}")
                return root.findall(path_ns)
            return []

        def _find(elem, path):
            if elem is None:
                return None
            e = elem.find(path)
            if e is not None:
                return e
            ns = _ns(elem.tag)
            if ns:
                path_ns = path.replace(".//", f".//{ns}") if ".//" in path else (ns + path)
                return elem.find(path_ns)
            return None

        ids = [e.text for e in _find_all(search_root, ".//Id") if e.text]
        if not ids:
            return []

        fetch_params = {
            "db": "pubmed",
            "id": ",".join(ids),
            "retmode": "xml",
        }
        fetch_response = requests.get(FETCH_URL, params=fetch_params, timeout=15)
        fetch_response.raise_for_status()
        fetch_root = ET.fromstring(fetch_response.content)

        articles = _find_all(fetch_root, ".//PubmedArticle")
        papers = []
        for article in articles:
            try:
                pmid_elem = _find(article, ".//PMID")
                pmid = pmid_elem.text if pmid_elem is not None and pmid_elem.text else ""
                if not pmid:
                    continue
                title_elem = _find(article, ".//ArticleTitle")
                title = title_elem.text if title_elem is not None and title_elem.text else "No Title"

                authors = []
                for author in _find_all(article, ".//Author"):
                    last_name = _find(author, "LastName")
                    initials = _find(author, "Initials")
                    if last_name is not None and last_name.text and initials is not None and initials.text:
                        authors.append(f"{last_name.text} {initials.text}")

                abstract_elem = _find(article, ".//AbstractText")
                abstract = abstract_elem.text if abstract_elem is not None and abstract_elem.text else ""

                pub_date_elem = _find(article, ".//PubDate/Year")
                pub_date = pub_date_elem.text if pub_date_elem is not None and pub_date_elem.text else "1900"
                try:
                    published = datetime.strptime(pub_date, "%Y")
                except ValueError:
                    published = None
                year_str = published.strftime("%Y-%m-%d") if published else pub_date

                papers.append(
                    BaseDocument(
                        page_content=abstract,
                        metadata={
                            "source": "pubmed",
                            "title": title,
                            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                            "published": year_str,
                            "published_date": year_str,
                            "authors": authors,
                            "abstract": abstract,
                        },
                    )
                )
            except Exception as e:
                print(f"Error parsing PubMed article: {e}")
        return papers
    except Exception as e:
        print(f"Error executing PubMed search: {e}")
        return []


def query_pubmed(query: str, max_results: int = 5, max_content_length: int = 10000) -> str:
    """
    Execute PubMed search query.
    Returns rich JSON: status, content, execution_context.
    """
    t0 = time.perf_counter()
    try:
        docs = _pubmed_search(query, max_results, max_content_length)
        results = [doc.metadata if hasattr(doc, 'metadata') else doc for doc in docs]
        payload = {"query": query, "count": len(results), "results": results}
        content = json.dumps(payload, ensure_ascii=False, default=str)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        return query_success_response(content, query_time_ms=elapsed_ms, source="pubmed")
    except Exception as e:
        return error_response("QueryError", str(e), suggestion="Check network connection or PubMed API status.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="PubMed Search")
    parser.add_argument("--query", type=str, default="BRCA1", help="Search query")
    parser.add_argument("--max_results", type=int, default=2, help="Max results")
    args = parser.parse_args()
    
    print(f"=== query_pubmed(query='{args.query}', max_results={args.max_results}) ===")
    res = query_pubmed(args.query, max_results=args.max_results)
    print(res)
