import logging
import random
import time
from typing import List, Optional
from langchain_core.documents import Document as BaseDocument
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


