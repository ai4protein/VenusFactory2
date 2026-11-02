"""Agent access to programs created by autonomous Scientist goals."""
import json
from langchain.tools import tool
from mcp_tools.tools.runtime_tool_policy import assert_tool_allowed

@tool('search_generated_capabilities')
def search_generated_capabilities(query: str='', limit: int=10, offset: int=0) -> str:
    """Search reusable models and computations created by autonomous Scientist goals. Results include an ID, scope, author, version and approved review status. Page with offset. Describe a match to inspect its input/output contract and limitations, then run it by ID."""
    assert_tool_allowed('search_generated_capabilities')
    from agent.harness.research_workspace import search_capabilities
    return json.dumps(search_capabilities(query,limit,offset=offset),ensure_ascii=False)

@tool('describe_generated_capability')
def describe_generated_capability(capability_id: str) -> str:
    """Inspect a Scientist-created program's scope, sample input and validation before using it."""
    assert_tool_allowed('describe_generated_capability')
    from agent.harness.research_workspace import describe_capability
    return json.dumps(describe_capability(capability_id),ensure_ascii=False)

@tool('run_generated_capability')
async def run_generated_capability(capability_id: str, inputs: dict) -> str:
    """Run a published Scientist-created model or computation with JSON inputs in an isolated process. Describe its input and scope first."""
    assert_tool_allowed('run_generated_capability')
    from agent.harness.research_workspace import run_capability
    return json.dumps(await run_capability(capability_id,inputs),ensure_ascii=False)

RESEARCH_CAPABILITY_TOOLS=[search_generated_capabilities,describe_generated_capability,run_generated_capability]
