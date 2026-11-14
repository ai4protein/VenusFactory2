"""MCP discovery bound to the server that exposes the callable tools."""

from typing import Literal

from fastmcp import FastMCP

from .service import search


def register_discovery_tools(server):
    from mcp_tools.tools.platform_manual import register_manual_tool
    register_manual_tool(server)
    # A mounted child's Context refers to the child, not the unified server.
    # Bind the actual catalog owner explicitly at registration time.
    @server.tool(name="search_research_tools")
    async def search_research_tools(
        query: str = "", kind: Literal["all", "tools", "models"] = "all",
        stage: str = "", limit: int = 10, offset: int = 0,
    ) -> dict:
        """Search scientific tools and platform protein models by protein ID, property or task (English/Chinese). For model inventory use kind='models', query=''. Results include bounded pages and current counts. For platform models, inspect the returned tool schema and selection hints; RSI case repositories are available separately in RSI Plaza for preview and download. Protein models are scientific tools, separate from the chat LLM selector."""
        mounted = await server.list_tools()
        catalog = [
            {"name": t.name, "description": t.description, "parameters": t.parameters}
            for t in mounted
        ]
        return search(query, kind, stage, limit, offset, tool_catalog=catalog)



mcp = FastMCP("Venus_Research_Discovery")
register_discovery_tools(mcp)
