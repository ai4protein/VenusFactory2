from typing import List
import requests
import os
from datetime import datetime, timedelta
from langchain_core.documents import Document as BaseDocument


BASE_URL = "https://api.biorxiv.org/details/biorxiv"
def _biorxiv_search(query: str, max_results: int = 5, days: int = 30) -> List[BaseDocument]:
    """
    Search for papers on bioRxiv by category within the last N days.

    Args:
        query (str): Search query string (must be in English)
        max_results (int): Maximum number of results to return
        days (int): Number of days to search within

    Returns:
        List of BaseDocument objects containing paper information
    """

    # Calculate date range: last N days
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    
    # Format category: lowercase and replace spaces with underscores
    category = query.lower().replace(' ', '_')
    
    papers = []
    cursor = 0
    while len(papers) < max_results:
        url = f"{BASE_URL}/{start_date}/{end_date}/{cursor}"
        if category:
            url += f"?category={category}"
        tries = 0
        while tries < 3:
            try:
                response = requests.get(url)
                response.raise_for_status()
                data = response.json()
                collection = data.get('collection', [])
                for item in collection:
                    try:
                        date = datetime.strptime(item['date'], '%Y-%m-%d')
                        papers.append(BaseDocument(
                            page_content=item['abstract'],
                            metadata={
                                "source": "biorxiv",
                                "title": item['title'],
                                "url": f"https://www.biorxiv.org/content/{item['doi']}v{item.get('version', '1')}",
                                "published_date": date,
                                "authors": item['authors'].split('; '),
                                "abstract": item['abstract']
                            }
                        ))
                    except Exception as e:
                        print(f"Error parsing bioRxiv entry: {e}")
                if len(collection) < 100:
                    break  # No more results
                cursor += 100
                break  # Exit retry loop on success
            except requests.exceptions.RequestException as e:
                tries += 1
                if tries == 3:
                    print(f"Failed to connect to bioRxiv API after {tries} attempts: {e}")
                    break
                print(f"Attempt {tries} failed, retrying...")
        else:
            continue
        break

    return papers[:max_results]

