"""Compatibility import; model execution belongs to the MCP tool service."""
import sys
from mcp_tools.models.pretrained.design_docking import service
sys.modules[__name__] = service
