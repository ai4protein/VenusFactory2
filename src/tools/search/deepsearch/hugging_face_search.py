import json
import sys
import time
from pathlib import Path
from typing import List

from huggingface_hub import HfApi

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

def _hugging_face_search(query: str, max_results: int = 5, api_key: str = "") -> List[BaseDocument]:
    """
    Execute Hugging Face search (models and datasets).
    """
    try:
        api = HfApi(token=api_key)
        
        # Search models
        models = api.list_models(search=query, limit=max_results, sort="downloads", direction=-1)
        # Search datasets
        datasets = api.list_datasets(search=query, limit=max_results, sort="downloads", direction=-1)
        
        docs = []
        for model in models:
            content = f"Model: {model.id}\nDownloads: {model.downloads}\nLikes: {model.likes}\nTask: {model.pipeline_tag}"
            metadata = {
                "title": model.id,
                "url": f"https://huggingface.co/{model.id}",
                "query": query,
                "type": "model"
            }
            docs.append(BaseDocument(page_content=content, metadata=metadata))
            
        for dataset in datasets:
            content = f"Dataset: {dataset.id}\nDownloads: {dataset.downloads}\nLikes: {dataset.likes}"
            metadata = {
                "source": "huggingface",
                "title": dataset.id,
                "url": f"https://huggingface.co/datasets/{dataset.id}",
                "query": query,
                "type": "dataset"
            }
            docs.append(BaseDocument(page_content=content, metadata=metadata))
        
        return docs
    except Exception as e:
        return []


def query_hugging_face(query: str, max_results: int = 5, api_key: str = "") -> str:
    """
    Execute Hugging Face search (models and datasets).
    Returns rich JSON: status, content, execution_context.
    """
    t0 = time.perf_counter()
    try:
        docs = _hugging_face_search(query, max_results, api_key)
        results = [doc.metadata if hasattr(doc, 'metadata') else doc for doc in docs]
        payload = {"query": query, "count": len(results), "results": results}
        content = json.dumps(payload, ensure_ascii=False, default=str)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        return query_success_response(content, query_time_ms=elapsed_ms, source="huggingface")
    except Exception as e:
        return error_response("QueryError", str(e), suggestion="Check network connection or valid huggingface keys.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Hugging Face Search")
    parser.add_argument("--query", type=str, default="machine learning", help="Search query")
    parser.add_argument("--max_results", type=int, default=2, help="Max results")
    args = parser.parse_args()
    
    print(f"=== query_hugging_face(query='{args.query}', max_results={args.max_results}) ===")
    res = query_hugging_face(args.query, max_results=args.max_results)
    print(res)

