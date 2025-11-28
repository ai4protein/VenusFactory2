import os
import json
import logging
import asyncio
import threading
import time
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
from uuid import uuid4
from pydantic import Field
from fastmcp import FastMCP
from mcp.types import TextContent
from web.utils.common_utils import get_save_path
from web.chat_tools import (
    PDB_sequence_extraction_tool,
    uniprot_query_tool,
    interpro_query_tool,
    pdb_structure_download_tool,
    ncbi_sequence_download_tool,
    alphafold_structure_download_tool,
    zero_shot_sequence_prediction_tool,
    zero_shot_structure_prediction_tool,
    protein_function_prediction_tool,
    functional_residue_prediction_tool,
    protein_properties_generation_tool,
    generate_training_config_tool,
    ai_code_execution_tool,
    literature_search_tool,
)

UPLOAD_DIR = get_save_path("MCP_Server", "Uploads")
OUTPUT_DIR = get_save_path("MCP_Server", "Outputs")

default_port = int(os.getenv("MCP_HTTP_PORT", "8002"))
default_host = os.getenv("MCP_HTTP_HOST", "0.0.0.0")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

mcp = FastMCP("VenusFactory MCP Server")

_http_server_thread: Optional[threading.Thread] = None
_http_server_lock = threading.Lock()


def start_http_server(host: Optional[str] = None, port: Optional[int] = None) -> tuple[str, int]:
    global _http_server_thread
    host = host or os.getenv("MCP_HTTP_HOST", "0.0.0.0")
    port = port or int(os.getenv("MCP_HTTP_PORT", "8080"))

    def _serve() -> None:
        try:
            logger.info(f"🚀 VenusFactory MCP Server running internally on {host}:{port}")
            logger.info(f"📡 SSE Endpoint: http://{host}:{port}/sse") 
            mcp.run(transport="sse", host=host, port=port)
            
        except Exception as exc:
            logger.error("MCP HTTP server exited unexpectedly: %s", exc)

    with _http_server_lock:
        if _http_server_thread and _http_server_thread.is_alive():
            logger.info("Server thread is already running.")
            return host, port
        thread = threading.Thread(target=_serve, name="MCPHttpServer", daemon=True)
        thread.start()
        _http_server_thread = thread
        time.sleep(2)

    return host, port

def format_tool_response(result: Any) -> str:
    try:
        if hasattr(result, 'content'):
            return str(result.content)
        if isinstance(result, (dict, list)):
            return json.dumps(result, ensure_ascii=False, indent=2)
        return str(result)
    except Exception as e:
        return f"Error processing result: {str(e)}"

@mcp.tool()
async def query_uniprot(uniprot_id: str) -> str:
    try:
        result = await asyncio.to_thread(uniprot_query_tool.invoke, {"uniprot_id": uniprot_id})
        return format_tool_response(result)
    except Exception as e:
        return f"Tool execution failed: {str(e)}"

@mcp.tool()
async def query_interpro(uniprot_id: str) -> str:
    try:
        result = await asyncio.to_thread(interpro_query_tool.invoke, {"uniprot_id": uniprot_id})
        return format_tool_response(result)
    except Exception as e:
        return f"Tool execution failed: {str(e)}"

@mcp.tool()
async def download_pdb_structure(pdb_id: str, output_format: str = "pdb") -> str:
    try:
        result = await asyncio.to_thread(pdb_structure_download_tool.invoke, {
            "pdb_id": pdb_id,
            "output_format": output_format
        })
        return format_tool_response(result)
    except Exception as e:
        return f"Tool execution failed: {str(e)}"

@mcp.tool()
async def download_ncbi_sequence(accession_id: str, output_format: str = "fasta") -> str:
    try:
        result = await asyncio.to_thread(ncbi_sequence_download_tool.invoke, {
            "accession_id": accession_id,
            "output_format": output_format
        })
        return format_tool_response(result)
    except Exception as e:
        return f"Tool execution failed: {str(e)}"

@mcp.tool()
async def download_alphafold_structure(uniprot_id: str, output_format: str = "pdb") -> str:
    try:
        result = await asyncio.to_thread(alphafold_structure_download_tool.invoke, {
            "uniprot_id": uniprot_id,
            "output_format": output_format
        })
        return format_tool_response(result)
    except Exception as e:
        return f"Tool execution failed: {str(e)}"

@mcp.tool()
async def extract_pdb_sequence(pdb_file_path: str) -> str:
    try:
        result = await asyncio.to_thread(PDB_sequence_extraction_tool.invoke, {"pdb_file": pdb_file_path})
        return format_tool_response(result)
    except Exception as e:
        return f"Tool execution failed: {str(e)}"

@mcp.tool()
async def predict_zero_shot_sequence(
    sequence: Optional[str] = None, 
    fasta_file: Optional[str] = None,
    model_name: str = "ESM2-650M"
) -> str:
    params = {"model_name": model_name}
    if fasta_file:
        params["fasta_file"] = fasta_file
    elif sequence:
        params["sequence"] = sequence
    else:
        return f"Tool execution failed: Either sequence or fasta_file is required"
        
    try:
        result = await asyncio.to_thread(zero_shot_sequence_prediction_tool.invoke, params)
        return format_tool_response(result)
    except Exception as e:
        return f"Tool execution failed: {str(e)}"

@mcp.tool()
async def predict_zero_shot_structure(
    structure_file_path: str,
    model_name: str = "ESM-IF1"
) -> str:
    try:
        result = await asyncio.to_thread(zero_shot_structure_prediction_tool.invoke, {
            "structure_file": structure_file_path,
            "model_name": model_name
        })
        return format_tool_response(result)
    except Exception as e:
        return f"Tool execution failed: {str(e)}"

@mcp.tool()
async def predict_protein_function(
    sequence: Optional[str] = None,
    fasta_file: Optional[str] = None,
    model_name: str = "ESM2-650M",
    task: str = "Solubility"
) -> str:
    params = {
        "model_name": model_name,
        "task": task
    }
    if fasta_file:
        params["fasta_file"] = fasta_file
    elif sequence:
        params["sequence"] = sequence
    else:
        return f"Tool execution failed: Either sequence or fasta_file is required"

    try:
        result = await asyncio.to_thread(protein_function_prediction_tool.invoke, params)
        return format_tool_response(result)
    except Exception as e:
        return f"Tool execution failed: {str(e)}"

@mcp.tool()
async def predict_functional_residue(
    sequence: Optional[str] = None,
    fasta_file: Optional[str] = None,
    model_name: str = "ESM2-650M",
    task: str = "Activity Site"
) -> str:
    params = {
        "model_name": model_name,
        "task": task
    }
    if fasta_file:
        params["fasta_file"] = fasta_file
    elif sequence:
        params["sequence"] = sequence
    else:
        return f"Tool execution failed: Either sequence or fasta_file is required"

    try:
        result = await asyncio.to_thread(functional_residue_prediction_tool.invoke, params)
        return format_tool_response(result)
    except Exception as e:
        return f"Tool execution failed: {str(e)}"

@mcp.tool()
async def predict_protein_properties(
    sequence: Optional[str] = None,
    fasta_file: Optional[str] = None,
    task_name: str = "Physical and chemical properties"
) -> str:
    params = {
        "task_name": task_name
    }
    if fasta_file:
        params["fasta_file"] = fasta_file
    elif sequence:
        params["sequence"] = sequence
    else:
        return f"Tool execution failed: Either sequence or fasta_file is required"

    try:
        result = await asyncio.to_thread(protein_properties_generation_tool.invoke, params)
        return format_tool_response(result)
    except Exception as e:
        return f"Tool execution failed: {str(e)}"

@mcp.tool()
async def generate_training_config(
    csv_file_path: str,
    test_csv_file_path: Optional[str] = None,
    output_name: str = "custom_training_config"
) -> str:
    try:
        result = await asyncio.to_thread(generate_training_config_tool.invoke, {
            "csv_file": csv_file_path,
            "test_csv_file": test_csv_file_path,
            "output_name": output_name
        })
        return format_tool_response(result)
    except Exception as e:
        return f"Tool execution failed: {str(e)}"

@mcp.tool()
async def execute_ai_code(
    task_description: str,
    input_files: List[str] = Field(default_factory=list)
) -> str:
    try:
        result = await asyncio.to_thread(ai_code_execution_tool.invoke, {
            "task_description": task_description,
            "input_files": input_files
        })
        return format_tool_response(result)
    except Exception as e:
        return f"Tool execution failed: {str(e)}"

@mcp.tool()
async def search_literature(query: str, max_results: int = 5) -> str:
    try:
        result = await asyncio.to_thread(literature_search_tool.invoke, {
            "query": query,
            "max_results": max_results
        })
        return format_tool_response(result)
    except Exception as e:
        return format_tool_response({"success": False, "error": str(e)})

if __name__ == "__main__":    
    logger.info("VenusFactory MCP Server starting...")
    mcp.run(transport="sse")
