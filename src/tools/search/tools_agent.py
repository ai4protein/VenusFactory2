# search: @tool definitions for retrieval, data fetch, and external DBs (BRENDA, ChEMBL, FDA, KEGG, STRING, ClinVar, Bioservices).
# Logic in .tools_mcp, .database, .source. All tools in this file; hub imports from here only.

import json
import os
from typing import List, Optional, Literal
from langchain.tools import tool
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


class ArxivSearchInput(BaseModel):
    query: str = Field(..., description="Search query string for arXiv. Required.")
    max_results: int = Field(default=5, ge=1, le=50, description="Maximum number of results to return. Default 5.")
    max_content_length: int = Field(default=10000, description="Max length for abstract text.")

@tool("query_arxiv", args_schema=ArxivSearchInput)
def query_arxiv_tool(query: str, max_results: int = 5, max_content_length: int = 10000) -> str:
    """Query arXiv using the official export API. Returns JSON list of papers."""
    return query_arxiv(query=query, max_results=max_results, max_content_length=max_content_length)

class BiorxivSearchInput(BaseModel):
    query: str = Field(..., description="Search query string (category) for bioRxiv. Required.")
    max_results: int = Field(default=5, ge=1, le=50, description="Maximum number of results to return. Default 5.")
    days: int = Field(default=30, ge=1, description="Number of days to search within. Default 30.")

@tool("query_biorxiv", args_schema=BiorxivSearchInput)
def query_biorxiv_tool(query: str, max_results: int = 5, days: int = 30) -> str:
    """Query bioRxiv API for papers. Returns JSON list of papers."""
    return query_biorxiv(query=query, max_results=max_results, days=days)

class DuckDuckGoSearchInput(BaseModel):
    query: str = Field(..., description="Search query string for DuckDuckGo. Required.")
    max_results: int = Field(default=5, ge=1, le=50, description="Maximum number of results to return. Default 5.")

@tool("query_duckduckgo", args_schema=DuckDuckGoSearchInput)
def query_duckduckgo_tool(query: str, max_results: int = 5) -> str:
    """Query DuckDuckGo for general web search. Returns JSON list of search results."""
    return query_duckduckgo(query=query, max_results=max_results)

class FdaSearchInput(BaseModel):
    query: str = Field(..., description="Drug or device name to search (e.g. aspirin, metformin, device name). Required.")
    max_results: int = Field(default=10, ge=1, le=500, description="Maximum number of results per source. Default 10.")
    source: str = Field(default="drug_events,drug_recalls", description="Comma-separated openFDA sources: drug_events, drug_recalls, device_events.")


class GithubSearchInput(BaseModel):
    query: str = Field(..., description="Search query string for GitHub. Required.")
    max_results: int = Field(default=5, ge=1, le=50, description="Maximum number of results to return. Default 5.")
    api_key: str = Field(default="", description="Optional GitHub Personal Access Token.")

@tool("query_github", args_schema=GithubSearchInput)
def query_github_tool(query: str, max_results: int = 5, api_key: str = "") -> str:
    """Query GitHub for repositories. Returns JSON list of repositories."""
    return query_github(query=query, max_results=max_results, api_key=api_key)

class TavilySearchInput(BaseModel):
    query: str = Field(..., description="Search query string for Tavily. Required.")
    max_results: int = Field(default=5, ge=1, le=50, description="Maximum number of results to return. Default 5.")

@tool("query_tavily", args_schema=TavilySearchInput)
def query_tavily_tool(query: str, max_results: int = 5) -> str:
    """Query Tavily for web search. Returns JSON list of search results."""
    return query_tavily(query=query, max_results=max_results)

class PubmedSearchInput(BaseModel):
    query: str = Field(..., description="Search query string for PubMed. Required.")
    max_results: int = Field(default=5, ge=1, le=50, description="Maximum number of results to return. Default 5.")
    max_content_length: int = Field(default=10000, description="Max length for abstract text.")

@tool("query_pubmed", args_schema=PubmedSearchInput)
def query_pubmed_tool(query: str, max_results: int = 5, max_content_length: int = 10000) -> str:
    """Query PubMed for biomedical literature. Returns JSON list of papers."""
    return query_pubmed(query=query, max_results=max_results, max_content_length=max_content_length)

class FdaDrugLabelSearchInput(BaseModel):
    query: str = Field(..., description="Search query string for openFDA drug labels. Required.")
    max_results: int = Field(default=5, ge=1, le=50, description="Maximum number of results to return. Default 5.")
    max_content_length: int = Field(default=10000, description="Max length for abstract text.")

@tool("query_fda", args_schema=FdaDrugLabelSearchInput)
def query_fda_tool(query: str, max_results: int = 5, max_content_length: int = 10000) -> str:
    """Query openFDA drug labeling database. Returns JSON list of drug label information."""
    return query_fda(query=query, max_results=max_results, max_content_length=max_content_length)


class HuggingFaceSearchInput(BaseModel):
    query: str = Field(..., description="Search query string for Hugging Face models and datasets. Required.")
    max_results: int = Field(default=5, ge=1, le=50, description="Maximum number of results to return. Default 5.")
    api_key: str = Field(default="", description="Optional Hugging Face API key for authenticated access.")

@tool("query_hugging_face", args_schema=HuggingFaceSearchInput)
def query_hugging_face_tool(query: str, max_results: int = 5, api_key: str = "") -> str:
    """Query Hugging Face for models and datasets. Returns JSON list of resources."""
    return query_hugging_face(query=query, max_results=max_results, api_key=api_key)

class SemanticScholarSearchInput(BaseModel):
    query: str = Field(..., description="Search query string for Semantic Scholar. Required.")
    max_results: int = Field(default=5, ge=1, le=50, description="Maximum number of results to return. Default 5.")

@tool("query_semantic_scholar", args_schema=SemanticScholarSearchInput)
def query_semantic_scholar_tool(query: str, max_results: int = 5) -> str:
    """Query Semantic Scholar for papers. Returns JSON list of papers."""
    return query_semantic_scholar(query=query, max_results=max_results)

DEEPSEARCH_TOOLS = [
    query_arxiv_tool,
    query_biorxiv_tool,
    query_duckduckgo_tool,
    query_fda_tool,
    query_github_tool,
    query_hugging_face_tool,
    query_pubmed_tool,
    query_semantic_scholar_tool,
    query_tavily_tool,
]
