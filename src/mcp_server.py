"""
Unified MCP server: mounts all tools from src/tools * /tools_mcp.py (mutation, predict, search, database).
Started by webui or run directly. Uses FastMCP mount() so tools are namespaced (e.g. mutation_*, predict_*).
"""
import os
import logging
import threading
import time
from typing import Optional

from fastmcp import FastMCP
import uvicorn
from dotenv import load_dotenv

load_dotenv()

from tools.mutation.tools_mcp import mcp as mutation_mcp
from tools.predict.tools_mcp import mcp as predict_mcp
from tools.search.tools_mcp import mcp as search_mcp
from tools.database.tools_mcp import mcp as database_mcp

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

mcp = FastMCP("VenusFactory MCP Server")
mcp.mount(mutation_mcp)
mcp.mount(predict_mcp)
mcp.mount(search_mcp)
mcp.mount(database_mcp)


_http_server_thread: Optional[threading.Thread] = None
_http_server_lock = threading.Lock()


def start_http_server(host: Optional[str] = None, port: Optional[int] = None) -> tuple[str, int]:
    """Start the unified MCP HTTP server in a background thread (for webui)."""
    global _http_server_thread
    host = host or os.getenv("MCP_HTTP_HOST", "0.0.0.0")
    port = port or int(os.getenv("MCP_HTTP_PORT", "8080"))

    def _serve() -> None:
        try:
            logger.info("VenusFactory2 MCP Server running on %s:%s", host, port)
            logger.info("MCP endpoint: http://%s:%s/mcp", host, port)
            app_with_route = mcp.http_app()
            uvicorn.run(app_with_route, host=host, port=port)
        except Exception as exc:
            logger.error("MCP HTTP server exited unexpectedly: %s", exc)

    with _http_server_lock:
        if _http_server_thread and _http_server_thread.is_alive():
            logger.info("MCP server thread already running.")
            return host, port
        _http_server_thread = threading.Thread(target=_serve, name="MCPHttpServer", daemon=True)
        _http_server_thread.start()
        time.sleep(2)

    return host, port


if __name__ == "__main__":
    logger.info("VenusFactory2 MCP Server starting...")
    start_http_server()
