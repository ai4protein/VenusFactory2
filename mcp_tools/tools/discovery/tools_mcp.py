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
        """Search scientific tools and platform, public and your reviewed private protein models by protein ID, property or task (English/Chinese). For model inventory use kind='models', query=''. Results include bounded pages and current counts. For platform models, inspect the returned tool schema and selection hints; for Scientist models, describe the selected entry and use its scope_ref and required inputs for prediction. Protein models are scientific tools, separate from the chat LLM selector."""
        mounted = await server.list_tools()
        catalog = [
            {"name": t.name, "description": t.description, "parameters": t.parameters}
            for t in mounted
        ]
        return search(query, kind, stage, limit, offset, tool_catalog=catalog)

    @server.tool(name="search_generated_capabilities")
    def search_generated_capabilities(query: str = "", limit: int = 10, offset: int = 0) -> dict:
        """Search public Scientist programs and your reviewed private deliveries by protein, property or scenario. Returns compact pages with scope summaries, author email and version; use offset to continue. Describe a selected ID for its full interface and review evidence. Private results stay owner-only."""
        from mcp_tools.tools.runtime_tool_policy import assert_tool_allowed
        from .generated_scope import search as search_capabilities
        assert_tool_allowed("search_generated_capabilities")
        return search_capabilities(query, limit, offset=offset)

    @server.tool(name="describe_generated_capability")
    def describe_generated_capability(capability_id: str) -> dict:
        """Inspect a Scientist program's input examples, scope and execution validation."""
        from mcp_tools.tools.runtime_tool_policy import assert_tool_allowed
        from .generated_scope import describe as describe_capability
        assert_tool_allowed("describe_generated_capability")
        return describe_capability(capability_id)

    @server.tool(name="run_generated_capability")
    async def run_generated_capability(capability_id: str, inputs: dict) -> dict:
        """Invoke a Scientist-created model or computation in an isolated process."""
        from mcp_tools.tools.runtime_tool_policy import assert_tool_allowed
        from .generated_scope import run as run_capability
        assert_tool_allowed("run_generated_capability")
        return await run_capability(capability_id, inputs)


mcp = FastMCP("Venus_Research_Discovery")
register_discovery_tools(mcp)
