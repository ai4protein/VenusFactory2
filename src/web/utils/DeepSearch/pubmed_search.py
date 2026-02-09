import os
import time
import requests
from typing import List, Annotated
from datetime import datetime
from langchain_community.utilities import PubMedAPIWrapper
from langchain_core.documents import Document as BaseDocument
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

        search_response = requests.get(SEARCH_URL, params=search_params)
        search_response.raise_for_status()
        search_root = ET.fromstring(search_response.content)
        ids = [id.text for id in search_root.findall('.//Id')]
        
        if not ids:
            return []

        fetch_params = {
            'db': 'pubmed',
            'id': ','.join(ids),
            'retmode': 'xml'
        }
        fetch_response = requests.get(FETCH_URL, params=fetch_params)
        fetch_response.raise_for_status()
        fetch_root = ET.fromstring(fetch_response.content)
        
        papers = []
        for article in fetch_root.findall('.//PubmedArticle'):
            try:
                pmid = article.find('.//PMID').text
                title_elem = article.find('.//ArticleTitle')
                title = title_elem.text if title_elem is not None else "No Title"
                
                authors = []
                author_list = article.findall('.//Author')
                for author in author_list:
                    last_name = author.find('LastName')
                    initials = author.find('Initials')
                    if last_name is not None and initials is not None:
                        authors.append(f"{last_name.text} {initials.text}")
                
                abstract_elem = article.find('.//AbstractText')
                abstract = abstract_elem.text if abstract_elem is not None else ''
                
                pub_date_elem = article.find('.//PubDate/Year')
                pub_date = pub_date_elem.text if pub_date_elem is not None else "1900"
                
                try:
                    published = datetime.strptime(pub_date, '%Y')
                except ValueError:
                    published = None
                
                papers.append(BaseDocument(
                    page_content=abstract,
                    metadata={
                        "source": "pubmed",
                        "title": title,
                        "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                        "published_date": published,
                        "abstract": abstract,
                        "author": authors,
                    }
                ))
            except Exception as e:
                print(f"Error parsing PubMed article: {e}")
        return papers
    except Exception as e:
        print(f"Error executing PubMed search: {e}")
        return []

