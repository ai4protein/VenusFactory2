"""LangChain capability search."""

import json
from typing import Literal

from langchain.tools import tool

from .service import search


@tool("search_research_tools")
def search_research_tools_tool(
    query: str = "", kind: Literal["all", "tools", "models"] = "all",
    stage: str = "", limit: int = 10, offset: int = 0,
) -> str:
    """Discover available scientific tools and published Scientist-created protein models. Use kind='models' when asked which protein models exist; use an empty query for an overview, or search by protein ID, property or task in English/Chinese. Results are ranked and paginated. For a model, call its returned describe tool before predict_with_evolved_model; these are protein predictors, not chat LLM choices. Native tool results give their registered names and arguments."""
    return json.dumps(search(query, kind, stage, limit, offset), ensure_ascii=False)


from mcp_tools.tools.platform_manual import query_manual, TOOL_DESCRIPTION


@tool('query_platform_manual', description=TOOL_DESCRIPTION)
def query_platform_manual_tool(query: str='', section: str='', language: Literal['auto','en','zh']='auto', limit: int=5) -> str:
    return json.dumps(query_manual(query,section,language,limit),ensure_ascii=False)


DISCOVERY_TOOLS = [search_research_tools_tool, query_platform_manual_tool]
