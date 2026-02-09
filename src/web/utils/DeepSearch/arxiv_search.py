from typing import List
from langchain_community.utilities import ArxivAPIWrapper
from langchain_core.documents import Document as BaseDocument

def _arxiv_search(query: str, max_results: int = 5, max_content_length: int = 10000) -> List[BaseDocument]:
    """
    Execute Arxiv search query.

    Args:
        query (str): Search query.
        max_results (int, optional): Maximum number of results to return. Defaults to 5.
        max_content_length (int, optional): Maximum length of document content. Defaults to 10000.

    Returns:
        List[BaseDocument]: List of documents retrieved from Arxiv.
    """
    try:
        api_wrapper = ArxivAPIWrapper(
            top_k_results=max_results,
            doc_content_chars_max=max_content_length
        )
        
        results = api_wrapper.load(query)
        docs = []
        for doc in results:
            metadata = {
                "source": "arxiv",
                "title": doc.metadata.get("Title"),
                "url": doc.metadata.get("Entry ID"), # ArxivAPIWrapper usually puts url in Entry ID or we construct it
                "query": query,
                "published_date": doc.metadata.get("Published"),
                "authors": doc.metadata.get("Authors"),
                "abstract": doc.metadata.get("Abstract")
            }
            docs.append(BaseDocument(page_content=doc.page_content, metadata=metadata))
            
        return docs
    except Exception as e:
        return []
