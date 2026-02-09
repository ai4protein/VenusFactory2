from typing import List
import requests
import os
import logging
from langchain_core.documents import Document as BaseDocument

def _github_search(query: str, max_results: int = 5, api_key: str = "") -> List[BaseDocument]:
    """
    Execute GitHub search query for repositories.
    
    Args:
        query (str): The search term.
        max_results (int): Number of repositories to return.
        api_key (str): Optional GitHub Personal Access Token.
        
    Returns:
        List[BaseDocument]: A list of documents representing GitHub repositories.
    """
    try:
        url = "https://api.github.com/search/repositories"
        headers = {
            "Accept": "application/vnd.github.v3+json"
        }
        if api_key:
            headers["Authorization"] = f"token {api_key}"
            
        params = {
            "q": query,
            "sort": "stars", # Default to sorting by stars as a proxy for "relative top"
            "order": "desc",
            "per_page": max_results
        }
        
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()
        
        items = data.get("items", [])
        docs = []
        
        for item in items:
            description = item.get("description") or "No description provided."
            content = (
                f"Repository: {item.get('full_name')}\n"
                f"Description: {description}\n"
                f"Language: {item.get('language')}\n"
                f"Stars: {item.get('stargazers_count')}\n"
                f"Forks: {item.get('forks_count')}\n"
                f"Updated: {item.get('updated_at')}"
            )
            
            metadata = {
                "source": "github",
                "title": item.get("full_name"),
                "url": item.get("html_url"),
                "query": query,
                "stars": item.get("stargazers_count"),
                "language": item.get("language"),
                "license": item.get("license", {}).get("name") if item.get("license") else None,
                "topics": item.get("topics", [])
            }
            
            docs.append(BaseDocument(page_content=content, metadata=metadata))
            
        return docs
        
    except requests.exceptions.RequestException as e:
        return []
    except Exception as e:
        return []

