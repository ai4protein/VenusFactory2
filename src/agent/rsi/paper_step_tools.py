"""MCP entry; copied as generated_tools.py in an audited scientific bundle."""

import os
from pathlib import Path

from fastmcp import FastMCP

from agent.rsi.paper_step_tool_service import PaperStepTools

service = PaperStepTools(Path(os.environ.get("VENUS_RSI_WORKSPACE", "/workspace")), Path(__file__).parent)
mcp = FastMCP("VenusFactoryPaperSteps")


@mcp.tool()
def inspect_paper_step_task() -> dict:
    """Inspect the frozen paper scope, reproduce(inputs) ABI and permitted development examples."""
    return service.inspect()


@mcp.tool()
def evaluate_paper_step(source_code: str, rationale: str) -> dict:
    """Execute your reproduce(inputs) Python on development examples only. Two durable attempts; every failed source consumes one. No host acceptance references are exposed."""
    return service.evaluate(source_code, rationale)


@mcp.tool()
def submit_paper_step(source_sha256: str) -> dict:
    """Submit one previously tested successful source. Host acceptance and registration happen after this session."""
    return service.submit(source_sha256)


@mcp.tool()
def discover_paper_steps() -> dict:
    """Discover a host-validated step compatible with the current numerical interface and declared scope."""
    return service.discover()


@mcp.tool()
def execute_registered_paper_step(capability_id: str) -> dict:
    """Execute the discovered immutable code on all fresh task inputs and save a numerical witness. Host verification follows this session."""
    return service.reuse(capability_id)


if __name__ == "__main__":
    mcp.run(transport="stdio", show_banner=False)
