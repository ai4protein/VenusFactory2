import logging
import os
import json
import requests
from typing import List
from langchain_core.documents import Document as BaseDocument
import dotenv

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
                "score": "Tavily"
            }
            content = res.get("content", "")
            docs.append(BaseDocument(page_content=content, metadata=metadata))
            
        # 转换成JSON格式返回
        serializable_docs = [
            {"page_content": doc.page_content, "metadata": doc.metadata} 
            for doc in docs
        ]
        return json.dumps(serializable_docs, ensure_ascii=False)
        
    except Exception as e:
        logger.error(f"Tavily search failed: {e}")
        return []

if __name__ == "__main__":

    query = "transformer"
    results = _tavily_search(query, max_results=2)
    print("results: ", results)
