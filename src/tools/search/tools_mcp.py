# search MCP outer layer: FastMCP tools; call core (deepsearch query_*), return JSON string.

from typing import Optional

from fastmcp import FastMCP

from .deepsearch.arxiv_search import query_arxiv
from .deepsearch.biorxiv_search import query_biorxiv
from .deepsearch.semantic_scholar_search import query_semantic_scholar
from .deepsearch.hugging_face_search import query_hugging_face
from .deepsearch.duckduckgo_search import query_duckduckgo
from .deepsearch.github_search import query_github
from .deepsearch.tavily_search import query_tavily
from .deepsearch.pubmed_search import query_pubmed
from .deepsearch.fda_search import query_fda


mcp = FastMCP("Venus_Search_MCP")


@mcp.tool(name="query_arxiv")
def mcp_query_arxiv(
    query: str,
    max_results: int = 5,
    max_content_length: int = 10000,
) -> str:
    """Query arXiv using the official export API. Returns JSON list of papers."""
    return query_arxiv(query=query, max_results=max_results, max_content_length=max_content_length)


@mcp.tool(name="query_biorxiv")
def mcp_query_biorxiv(
    query: str,
    max_results: int = 5,
    days: int = 30,
) -> str:
    """Query bioRxiv API for papers. Returns JSON list of papers."""
    return query_biorxiv(query=query, max_results=max_results, days=days)


@mcp.tool(name="query_duckduckgo")
def mcp_query_duckduckgo(
    query: str,
    max_results: int = 5,
) -> str:
    """Query DuckDuckGo for general web search. Returns JSON list of search results."""
    return query_duckduckgo(query=query, max_results=max_results)


@mcp.tool(name="query_github")
def mcp_query_github(
    query: str,
    max_results: int = 5,
    api_key: str = "",
) -> str:
    """Query GitHub for repositories. Returns JSON list of repositories."""
    return query_github(query=query, max_results=max_results, api_key=api_key)


@mcp.tool(name="query_tavily")
def mcp_query_tavily(
    query: str,
    max_results: int = 5,
) -> str:
    """Query Tavily for web search. Returns JSON list of search results."""
    return query_tavily(query=query, max_results=max_results)


@mcp.tool(name="query_pubmed")
def mcp_query_pubmed(
    query: str,
    max_results: int = 5,
    max_content_length: int = 10000,
) -> str:
    """Query PubMed for biomedical literature. Returns JSON list of papers."""
    return query_pubmed(query=query, max_results=max_results, max_content_length=max_content_length)


@mcp.tool(name="query_fda")
def mcp_query_fda(
    query: str,
    max_results: int = 5,
    max_content_length: int = 10000,
) -> str:
    """Query openFDA drug labeling database. Returns JSON list of drug label information."""
    return query_fda(query=query, max_results=max_results, max_content_length=max_content_length)


@mcp.tool(name="query_hugging_face")
def mcp_query_hugging_face(
    query: str,
    max_results: int = 5,
    api_key: str = "",
) -> str:
    """Query Hugging Face for models and datasets. Returns JSON list of resources."""
    return query_hugging_face(query=query, max_results=max_results, api_key=api_key)


@mcp.tool(name="query_semantic_scholar")
def mcp_query_semantic_scholar(
    query: str,
    max_results: int = 5,
) -> str:
    """Query Semantic Scholar for papers. Returns JSON list of papers."""
    return query_semantic_scholar(query=query, max_results=max_results)
