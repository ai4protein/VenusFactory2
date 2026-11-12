"""
Unified MCP server: mounts all tools from mcp_tools/tools/*/tools_mcp.py
(mutation, predict, search, database, denovo, skill).
Started by webui or run directly. Uses FastMCP mount() so tools are namespaced.
"""
import os
import logging

from fastmcp import FastMCP
import uvicorn
from dotenv import load_dotenv

load_dotenv()

from mcp_tools.models.pretrained.mutation.tools_mcp import mcp as mutation_mcp
from mcp_tools.tools.predict.tools_mcp import mcp as predict_mcp
from mcp_tools.tools.search.tools_mcp import mcp as search_mcp
from mcp_tools.tools.research.tools_mcp import mcp as public_protein_mcp
from mcp_tools.tools.database.tools_mcp import mcp as database_mcp
from mcp_tools.models.pretrained.design.tools_mcp import mcp as denovo_mcp
from mcp_tools.tools.skill.tools_mcp import mcp as skill_mcp
from mcp_tools.tools.discovery.tools_mcp import register_discovery_tools
from mcp_tools.tools.runtime_tool_policy import is_local_mode

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

mcp = FastMCP("VenusFactory MCP Server")
mcp.mount(mutation_mcp)
mcp.mount(predict_mcp)
from mcp_tools.models.pretrained.admet.tools_mcp import mcp as admet_mcp
mcp.mount(admet_mcp)
from mcp_tools.models.pretrained.design_docking.tools_mcp import mcp as external_mcp
mcp.mount(external_mcp)
mcp.mount(search_mcp)
mcp.mount(public_protein_mcp)
mcp.mount(database_mcp)
mcp.mount(denovo_mcp)
mcp.mount(skill_mcp)
register_discovery_tools(mcp)
# Training uses the shared queue and scoped ESM2-8M policy online.
from mcp_tools.models.factory.training.tools_mcp import mcp as train_mcp
mcp.mount(train_mcp)
if is_local_mode():
    from mcp_tools.models.evolved.tools_mcp import mcp as evolved_mcp
    mcp.mount(evolved_mcp)



# Workers invoke the original tool implementations without submitting recursively.
from mcp_tools.scheduler.mcp_service import QueueMiddleware
mcp.add_middleware(QueueMiddleware())


def start_http_server(host=None, port=None):
    """Compatibility entry point: ensure the independent service is running."""
    if host and host != "127.0.0.1":
        raise ValueError("The tool service must bind to loopback")
    if port:
        os.environ["MCP_HTTP_PORT"] = str(port)
    from mcp_tools.scheduler.service import ensure_service
    ensure_service()
    return "127.0.0.1", int(os.getenv("MCP_HTTP_PORT", "8080"))


if __name__ == "__main__":
    from mcp_tools.scheduler.mcp_service import application
    uvicorn.run(application(mcp, scheduler_only=globals().get("_VENUS_SCHEDULER_ONLY", False)),
                host="127.0.0.1", port=int(os.getenv("MCP_HTTP_PORT", "8080")),
                fd=globals().get("_VENUS_LISTENER_FD"))
