# deepsearch: 各数据源检索实现（被 source 层调用）
# - arxiv_search, biorxiv_search, pubmed_search, semantic_scholar_search → literature
# - duckduckgo_search, tavily_search → web_search
# - github_search, hugging_face_search → dataset_search
# - google_scholar_search

from .arxiv_search import _arxiv_search
from .biorxiv_search import _biorxiv_search
from .pubmed_search import _pubmed_search
from .semantic_scholar_search import _semantic_scholar_search, SemanticScholarAPIWrapper
from .duckduckgo_search import _duckduckgo_search
from .tavily_search import _tavily_search
from .github_search import _github_search
from .hugging_face_search import _hugging_face_search
from .google_scholar_search import _google_scholar_search

__all__ = [
    "_arxiv_search",
    "_biorxiv_search",
    "_pubmed_search",
    "_semantic_scholar_search",
    "SemanticScholarAPIWrapper",
    "_duckduckgo_search",
    "_tavily_search",
    "_github_search",
    "_hugging_face_search",
    "_google_scholar_search",
]
