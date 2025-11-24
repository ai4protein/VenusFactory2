import time
import requests


def query_arxiv(query: str, max_results: int = 10):
    """
    Query arXiv using the official API.

    Args:
        query: The query to search for.
        max_results: The maximum number of results to return.

    Returns:
        A list of structured entries or an empty list on error.
        Each entry: {"title", "summary", "authors", "published", "pdf_url", "doi", "source"}
    """
    try:
        # arXiv API endpoint
        base_url = "http://export.arxiv.org/api/query"
        params = {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": max_results,
            "sortBy": "relevance",
            "sortOrder": "descending"
        }
        
        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status()
        
        # Parse Atom XML response
        from xml.etree import ElementTree as ET
        root = ET.fromstring(response.content)
        
        # Define namespaces
        ns = {
            'atom': 'http://www.w3.org/2005/Atom',
            'arxiv': 'http://arxiv.org/schemas/atom'
        }
        
        entries = []
        for entry in root.findall('atom:entry', ns):
            title = entry.find('atom:title', ns)
            summary = entry.find('atom:summary', ns)
            published = entry.find('atom:published', ns)
            
            # Extract authors
            authors = []
            for author in entry.findall('atom:author', ns):
                name = author.find('atom:name', ns)
                if name is not None and name.text:
                    authors.append(name.text.strip())
            
            # Extract PDF link
            pdf_url = ""
            for link in entry.findall('atom:link', ns):
                if link.get('title') == 'pdf':
                    pdf_url = link.get('href', '')
                    break
            
            # Extract DOI
            doi = entry.find('arxiv:doi', ns)
            
            entries.append({
                "title": title.text.strip() if title is not None else "",
                "summary": summary.text.strip() if summary is not None else "",
                "authors": [str(a) for a in authors],
                "published": published.text.strip() if published is not None else "",
                "pdf_url": pdf_url,
                "doi": doi.text.strip() if doi is not None else "",
                "source": "arXiv"
            })
        
        return entries
    except Exception as e:
        return []


def query_pubmed(query: str, max_papers: int = 10, max_retries: int = 3):
    """
    Query PubMed using NCBI E-utilities API.

    Args:
        query: The query to search for.
        max_papers: The maximum number of papers to return.
        max_retries: The maximum number of retries if the query fails.

    Returns:
        A list of structured entries or an empty list on error.
        Each entry: {"title", "authors", "published", "abstract", "pmid", "source", "url", "doi"}
    """
    try:
        # Step 1: Use esearch to get PMIDs
        esearch = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        params = {"db": "pubmed", "term": query, "retmax": max_papers, "retmode": "json"}
        r = requests.get(esearch, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        idlist = data.get("esearchresult", {}).get("idlist", [])
        
        if not idlist:
            return []
        
        entries = []
        
        # Step 2: Use esummary to get metadata
        esummary = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
        params = {"db": "pubmed", "id": ",".join(idlist), "retmode": "json"}
        r2 = requests.get(esummary, params=params, timeout=10)
        r2.raise_for_status()
        sumdata = r2.json().get("result", {})
        
        for pmid in idlist:
            item = sumdata.get(pmid, {})
            title = item.get("title", "") or ""
            pubdate = item.get("pubdate", "") or ""
            source = item.get("source", "") or "PubMed"
            
            # Extract authors list
            authors_list = []
            if "authors" in item and isinstance(item["authors"], list):
                for a in item["authors"]:
                    name = a.get("name") if isinstance(a, dict) else str(a)
                    if name:
                        authors_list.append(name)
            
            # Step 3: Use efetch to get abstract
            abstract = ""
            try:
                efetch = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
                p3 = {"db": "pubmed", "id": pmid, "retmode": "text", "rettype": "abstract"}
                r3 = requests.get(efetch, params=p3, timeout=10)
                if r3.status_code == 200:
                    abstract = r3.text.strip()
            except Exception:
                abstract = ""
            
            # Try to extract DOI if available
            doi = ""
            if isinstance(item.get("articleids", []), list):
                for aid in item.get("articleids", []):
                    if aid.get("idtype") == "doi":
                        doi = aid.get("value") or ""
            
            url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
            
            entries.append({
                "title": title,
                "authors": authors_list,
                "published": pubdate,
                "abstract": abstract,
                "pmid": pmid,
                "doi": doi,
                "url": url,
                "source": "PubMed"
            })
        
        return entries
    except Exception:
        return []


def literature_search(query: str, max_results: int = 5, delay: float = 0.5) -> list[dict]:
    """
    Search literature using arXiv and PubMed APIs.

    Args:
        query: The query to search for.
        max_results: The maximum number of results to return from each source.
        delay: The delay between searches in seconds.

    Returns:
        A list of deduplicated search results.
    """
    all_refs = []
    
    # 1) Query arXiv
    try:
        arxiv_entries = query_arxiv(query, max_results=max_results)
        all_refs.extend(arxiv_entries)
    except Exception:
        pass
    
    time.sleep(delay)
    
    # 2) Query PubMed
    try:
        pubmed_entries = query_pubmed(query, max_papers=max_results)
        all_refs.extend(pubmed_entries)
    except Exception:
        pass
    
    # Normalize all entries to unified format
    unified = []
    for item in all_refs:
        if not isinstance(item, dict):
            continue
        
        # Extract common fields
        title = item.get("title") or ""
        url = item.get("url") or item.get("pdf_url") or ""
        authors = item.get("authors") or []
        year = item.get("published") or item.get("pubdate") or ""
        abstract = item.get("summary") or item.get("abstract") or ""
        doi = item.get("doi") or ""
        source = item.get("source") or ""
        
        unified.append({
            "source": source,
            "title": title,
            "url": url,
            "authors": authors,
            "year": year,
            "abstract": abstract,
            "doi": doi
        })
    
    # Deduplicate by (title, url or doi)
    seen = set()
    final_refs = []
    for r in unified:
        key = (r.get("title", "").strip().lower(), (r.get("url", "") or r.get("doi", "")).strip().lower())
        if key in seen or not r.get("title"):
            continue
        seen.add(key)
        final_refs.append(r)
    
    return final_refs[:max_results * 2]  # Return up to max_results * 2 total results