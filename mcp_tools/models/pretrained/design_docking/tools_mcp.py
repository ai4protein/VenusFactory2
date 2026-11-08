"""MCP transport for the shared design and docking operations."""
from fastmcp import FastMCP
from .operations import ligandmpnn_design, antibmpnn_design, vina_dock, bindcraft_design, boltzgen_design, diffdock_dock
from .tools_agent import EXTERNAL_TOOLS

mcp = FastMCP("VenusFactory Design and Docking")
for _operation in (ligandmpnn_design, antibmpnn_design, vina_dock, bindcraft_design, boltzgen_design, diffdock_dock):
    mcp.tool(name=_operation.__name__)(_operation)
