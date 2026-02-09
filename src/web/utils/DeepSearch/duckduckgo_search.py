import logging
import os
import json
import requests
import urllib.parse
import concurrent.futures
from typing import List
from bs4 import BeautifulSoup
from langchain_core.documents import Document as BaseDocument

logger = logging.getLogger(__name__)

def _duckduckgo_search(query: str, max_results: int = 5, api_key: str = "") -> str:
    """
    Execute DuckDuckGo search query using scraping.
    """
    results_docs = []
    
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
        }
        
        # 1. Use DuckDuckGo HTML search
        encoded_query = urllib.parse.quote(query)
        search_url = f"https://duckduckgo.com/html/?q={encoded_query}"
        
        # Container for initial results
        search_items = []
        
        logger.info(f"Searching DuckDuckGo for: {query}")
        response = requests.get(search_url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            results_blocks = soup.select('div.result')
            logger.info(f"Found {len(results_blocks)} results on DuckDuckGo")
            
            for result in results_blocks[:max_results]:
                title_el = result.select_one('h2.result__title')
                link_el = result.select_one('a.result__a')
                desc_el = result.select_one('a.result__snippet')
                
                if title_el and link_el:
                    url = link_el.get('href')
                    if url and '//duckduckgo.com/l/?' in url:
                        try:
                            # Extract real URL
                            url_param = urllib.parse.parse_qs(urllib.parse.urlparse(url).query).get('uddg', [''])[0]
                            if url_param:
                                url = url_param
                        except:
                            pass
                        
                    if url and url.startswith('http'):
                        search_items.append({
                            "title": title_el.get_text().strip(),
                            "url": url,
                            "description": desc_el.get_text().strip() if desc_el else "",
                            "source": "Web Search"
                        })

        # 2. Fetch page content concurrently
        def fetch_page_content(item):
            try:
                page_resp = requests.get(item['url'], headers=headers, timeout=5)
                if page_resp.status_code == 200:
                    page_soup = BeautifulSoup(page_resp.text, 'html.parser')
                    
                    for script in page_soup(["script", "style", "nav", "footer", "header", "iframe", "noscript"]):
                        script.extract()
                    
                    paragraphs = page_soup.find_all(['p', 'h1', 'h2', 'h3'])
                    
                    content_parts = []
                    char_count = 0
                    max_chars = 1000
                    
                    for p in paragraphs:
                        text = p.get_text().strip()
                        if len(text) > 30: 
                            content_parts.append(text)
                            char_count += len(text)
                        if char_count >= max_chars:
                            break
                            
                    full_content = "\n".join(content_parts)
                    
                    if full_content:
                        item['content'] = full_content
                    else:
                        item['content'] = item['description']
                else:
                    item['content'] = item['description']
            except Exception:
                item['content'] = item['description']
            return item

        # 3. Concurrent execution
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            future_to_item = {executor.submit(fetch_page_content, item): item for item in search_items}
            
            for future in concurrent.futures.as_completed(future_to_item):
                try:
                    updated_item = future.result()
                    
                    metadata = {
                        "title": updated_item.get("title"),
                        "url": updated_item.get("url"),
                        "source": "duckduckgo",
                        "query": query,
                        "description": updated_item.get("description")
                    }
                    content = updated_item.get("content", "")
                    doc = BaseDocument(page_content=content, metadata=metadata)
                    results_docs.append(doc)
                    
                except Exception:
                    continue

        serializable_docs = [
            {"page_content": doc.page_content, "metadata": doc.metadata} 
            for doc in results_docs
        ]
        return json.dumps(serializable_docs, ensure_ascii=False)

    except Exception as e:
        logger.error(f"DuckDuckGo search failed: {str(e)}")
        return json.dumps([], ensure_ascii=False)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    query = "transformer"
    results = _duckduckgo_search(query, max_results=2)
    print("results: ", results)
