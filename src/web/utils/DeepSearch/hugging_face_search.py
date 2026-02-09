from typing import List
from huggingface_hub import HfApi
from langchain_core.documents import Document as BaseDocument

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

# print(_hugging_face_search("machine learning"))
