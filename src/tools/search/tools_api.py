# search API outer layer: FastAPI routes; call core (deepsearch query_*), return JSON.

import json
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from .deepsearch.arxiv_search import query_arxiv
from .deepsearch.biorxiv_search import query_biorxiv
from .deepsearch.semantic_scholar_search import query_semantic_scholar
from .deepsearch.hugging_face_search import query_hugging_face
from .deepsearch.duckduckgo_search import query_duckduckgo
from .deepsearch.github_search import query_github
from .deepsearch.tavily_search import query_tavily
from .deepsearch.pubmed_search import query_pubmed
from .deepsearch.fda_search import query_fda


router = APIRouter(prefix="/api/v1/search", tags=["search"])


def _ensure_dict(result):
    """Core returns JSON str; parse to dict for application/json response."""
    if isinstance(result, str):
        try:
            return json.loads(result)
        except json.JSONDecodeError:
            return {"raw": result}
    return result


# ---------- Request bodies ----------
class QueryArxivBody(BaseModel):
    query: str = Field(..., description="Search query for arXiv.")
    max_results: int = Field(default=5, ge=1, le=50, description="Max results. Default 5.")
    max_content_length: int = Field(default=10000, description="Max abstract length.")


class QueryBiorxivBody(BaseModel):
    query: str = Field(..., description="Search query (category) for bioRxiv.")
    max_results: int = Field(default=5, ge=1, le=50, description="Max results. Default 5.")
    days: int = Field(default=30, ge=1, description="Days to search. Default 30.")


class QueryDuckduckgoBody(BaseModel):
    query: str = Field(..., description="Search query for DuckDuckGo.")
    max_results: int = Field(default=5, ge=1, le=50, description="Max results. Default 5.")


class QueryGithubBody(BaseModel):
    query: str = Field(..., description="Search query for GitHub.")
    max_results: int = Field(default=5, ge=1, le=50, description="Max results. Default 5.")
    api_key: str = Field(default="", description="Optional GitHub API key.")


class QueryTavilyBody(BaseModel):
    query: str = Field(..., description="Search query for Tavily.")
    max_results: int = Field(default=5, ge=1, le=50, description="Max results. Default 5.")


class QueryPubmedBody(BaseModel):
    query: str = Field(..., description="Search query for PubMed.")
    max_results: int = Field(default=5, ge=1, le=50, description="Max results. Default 5.")
    max_content_length: int = Field(default=10000, description="Max abstract length.")


class QueryFdaBody(BaseModel):
    query: str = Field(..., description="Search query for openFDA drug labels.")
    max_results: int = Field(default=5, ge=1, le=50, description="Max results. Default 5.")
    max_content_length: int = Field(default=10000, description="Max content length.")


class QueryHuggingFaceBody(BaseModel):
    query: str = Field(..., description="Search query for Hugging Face.")
    max_results: int = Field(default=5, ge=1, le=50, description="Max results. Default 5.")
    api_key: str = Field(default="", description="Optional Hugging Face API key.")


class QuerySemanticScholarBody(BaseModel):
    query: str = Field(..., description="Search query for Semantic Scholar.")
    max_results: int = Field(default=5, ge=1, le=50, description="Max results. Default 5.")


# ---------- Routes ----------
@router.post("/arxiv")
def api_query_arxiv(body: QueryArxivBody):
    """Query arXiv. Returns JSON list of papers."""
    try:
        result = query_arxiv(
            query=body.query,
            max_results=body.max_results,
            max_content_length=body.max_content_length,
        )
        return _ensure_dict(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/biorxiv")
def api_query_biorxiv(body: QueryBiorxivBody):
    """Query bioRxiv. Returns JSON list of papers."""
    try:
        result = query_biorxiv(
            query=body.query,
            max_results=body.max_results,
            days=body.days,
        )
        return _ensure_dict(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/duckduckgo")
def api_query_duckduckgo(body: QueryDuckduckgoBody):
    """Query DuckDuckGo. Returns JSON list of search results."""
    try:
        result = query_duckduckgo(query=body.query, max_results=body.max_results)
        return _ensure_dict(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/github")
def api_query_github(body: QueryGithubBody):
    """Query GitHub. Returns JSON list of repositories."""
    try:
        result = query_github(
            query=body.query,
            max_results=body.max_results,
            api_key=body.api_key,
        )
        return _ensure_dict(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tavily")
def api_query_tavily(body: QueryTavilyBody):
    """Query Tavily. Returns JSON list of search results."""
    try:
        result = query_tavily(query=body.query, max_results=body.max_results)
        return _ensure_dict(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/pubmed")
def api_query_pubmed(body: QueryPubmedBody):
    """Query PubMed. Returns JSON list of papers."""
    try:
        result = query_pubmed(
            query=body.query,
            max_results=body.max_results,
            max_content_length=body.max_content_length,
        )
        return _ensure_dict(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/fda")
def api_query_fda(body: QueryFdaBody):
    """Query openFDA drug labels. Returns JSON list of drug label info."""
    try:
        result = query_fda(
            query=body.query,
            max_results=body.max_results,
            max_content_length=body.max_content_length,
        )
        return _ensure_dict(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/hugging_face")
def api_query_hugging_face(body: QueryHuggingFaceBody):
    """Query Hugging Face. Returns JSON list of models/datasets."""
    try:
        result = query_hugging_face(
            query=body.query,
            max_results=body.max_results,
            api_key=body.api_key,
        )
        return _ensure_dict(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/semantic_scholar")
def api_query_semantic_scholar(body: QuerySemanticScholarBody):
    """Query Semantic Scholar. Returns JSON list of papers."""
    try:
        result = query_semantic_scholar(
            query=body.query,
            max_results=body.max_results,
        )
        return _ensure_dict(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
