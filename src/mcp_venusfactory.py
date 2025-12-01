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
import uvicorn
from fastapi import FastAPI
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
    literature_search_tool,
)

UPLOAD_DIR = get_save_path("MCP_Server", "Uploads")
OUTPUT_DIR = get_save_path("MCP_Server", "Outputs")

default_port = int(os.getenv("MCP_HTTP_PORT", "8080"))
default_host = os.getenv("MCP_HTTP_HOST", "0.0.0.0")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

mcp = FastMCP("VenusFactory MCP Server")

try:
    if hasattr(mcp, "_http_app"):
        mcp._http_app.add_middleware(ProxyHeadersMiddleware, trusted_hosts="*")
        logger.info("✅ ProxyHeadersMiddleware added to _http_app")
    elif hasattr(mcp, "app"):
        mcp.app.add_middleware(ProxyHeadersMiddleware, trusted_hosts="*")
        logger.info("✅ ProxyHeadersMiddleware added to app")
    else:
        os.environ["FORWARDED_ALLOW_IPS"] = "*"
        logger.info("⚠️ Middleware object not found, using env var FORWARDED_ALLOW_IPS=*")
except Exception as e:
    logger.warning(f"Failed to add ProxyHeadersMiddleware: {e}")
    os.environ["FORWARDED_ALLOW_IPS"] = "*"

_http_server_thread: Optional[threading.Thread] = None
_http_server_lock = threading.Lock()


def start_http_server(host: Optional[str] = None, port: Optional[int] = None) -> tuple[str, int]:
    global _http_server_thread
    host = host or os.getenv("MCP_HTTP_HOST", "0.0.0.0")
    port = port or int(os.getenv("MCP_HTTP_PORT", "8080"))

    def _serve() -> None:
        try:
            logger.info(f"🚀 VenusFactory MCP Server running on {host}:{port}")
            mcp_asgi = getattr(mcp, "_http_app", getattr(mcp, "app", None))
            
            if not mcp_asgi:
                logger.error("Could not find MCP ASGI app. Init failed.")
                return
            parent_app = FastAPI()
            
            mount_path = "/MCP/Protein_Tool"
            
            parent_app.mount(mount_path, mcp_asgi)
            
            logger.info(f"✅ MCP Mounted at: {mount_path}")
            logger.info(f"📡 SSE URL: http://{host}:{port}{mount_path}/sse")
            logger.info(f"📨 Message URL: http://{host}:{port}{mount_path}/message")


            uvicorn.run(
                parent_app, 
                host=host, 
                port=port, 
                log_level="info",
                access_log=False 
            )
            
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
    """
    Query UniProt for protein information.
    Args:
        uniprot_id (str): UniProt ID of the protein.
    Returns:
        str: JSON-formatted protein information.
    """
    try:
        result = await asyncio.to_thread(uniprot_query_tool.invoke, {"uniprot_id": uniprot_id})
        return format_tool_response(result)
    except Exception as e:
        return f"Tool execution failed: {str(e)}"

@mcp.tool()
async def query_interpro(uniprot_id: str) -> str:
    """
    Query InterPro for protein domain information.
    Args:
        uniprot_id (str): UniProt ID of the protein.
    Returns:
        str: JSON-formatted domain information.
    """
    try:
        result = await asyncio.to_thread(interpro_query_tool.invoke, {"uniprot_id": uniprot_id})
        return format_tool_response(result)
    except Exception as e:
        return f"Tool execution failed: {str(e)}"

@mcp.tool()
async def download_pdb_structure(pdb_id: str, output_format: str = "pdb") -> str:
    """
    Download PDB structure file.
    Args:
        pdb_id (str): PDB ID of the structure.
        output_format (str): Output the path of the structure file.
    Returns:
        str: Path to the downloaded structure file.
    """
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
    """
    Download NCBI sequence file.
    Args:
        accession_id (str): Accession ID of the sequence.
        output_format (str): Output the path of the sequence file.
    Returns:
        str: Path to the downloaded sequence file.
    """
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
    """
    Download AlphaFold structure file.
    Args:
        uniprot_id (str): UniProt ID of the protein.
        output_format (str): Output the path of the structure file.
    Returns:
        str: Path to the downloaded structure file.
    """
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
    """
    Extract sequence from PDB file.
    Args:
        pdb_file_path (str): Path to the PDB file.
    Returns:
        str: Extracted sequence.
    """
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
    """
    Predict zero-shot sequence.
    Args:
        sequence (Optional[str]): Protein sequence.
        fasta_file (Optional[str]): Path to the FASTA file.
        model_name (str): Model name for prediction. Support Model: VenusPLM, ESM2-650M, ESM-1b, ESM-1v
    Returns:
        str: Prediction result.
    """
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
    """
    Predict zero-shot structure.
    Args:
        structure_file_path (str): Path to the structure file.
        model_name (str): Model name for prediction. Supported models: VenusREM (foldseek-based), ProSST-2048, ProtSSN, ESM-IF1, SaProt, MIF-ST
    Returns:
        str: Prediction result.
    """
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
    """
    Predict protein function.
    Args:
        sequence (Optional[str]): Protein sequence.
        fasta_file (Optional[str]): Path to the FASTA file.
        model_name (str): Model name for prediction. Support Models: ESM2-650M, Ankh-large, ProtBert, ProtT5-xl-uniref50 
        task (str): Task for prediction. Support Task: Solubility, Subcellular Localization, Membrane Protein, Metal Ion Binding, Stability, Sortingsignal, Optimal Temperature, Kcat, Optimal PH, Immunogenicity Prediction - Virus, Immunogenicity Prediction - Bacteria, Immunogenicity Prediction - Tumor
    Returns:
        str: Prediction result.
    """
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
    """
    Predict functional residue.
    Args:
        sequence (Optional[str]): Protein sequence.
        fasta_file (Optional[str]): Path to the FASTA file.
        model_name (str): Model name for prediction. Support Models: ESM2-650M, Ankh-large, ProtT5-xl-uniref50 
        task (str): Task for prediction. Support Task: Activity Site, Binding Site, Conserved Site, Motif
    Returns:
        str: Prediction result.
    """
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
    """
    Predict protein properties.
    Args:
        sequence (Optional[str]): Protein sequence.
        fasta_file (Optional[str]): Path to the FASTA file.
        task_name (str): Task name for prediction. Support Task: Physical and chemical properties, Relative solvent accessible surface area (PDB only), SASA value (PDB only), Secondary structure (PDB only)
    Returns:
        str: Prediction result.
    """
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
async def search_literature(query: str, max_results: int = 5) -> str:
    """
    Search literature.
    Args:
        query (str): Search query.
        max_results (int): Maximum number of results.
    Returns:
        str: Search results.
    """
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
