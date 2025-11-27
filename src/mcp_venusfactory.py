import os
import json
import logging
import asyncio
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
from uuid import uuid4
import sys 
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
from pydantic import Field
# ------------------------------------------------------------------
# 关键修改：导入 FastMCP 而不是 Server
from mcp.server.fastmcp import FastMCP
from mcp.types import TextContent
# ------------------------------------------------------------------

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
OUTPUT_DIR = get_save_path("MCP_Server", "Temp_Outputs")

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# 关键修改：初始化 FastMCP
app = FastMCP("venusfactory-protein-analysis")
# ------------------------------------------------------------------

def format_tool_response(result: Any) -> List[TextContent]:
    """辅助函数：将结果格式化为 MCP 标准响应"""
    response_dict = {
        "success": True,
        "data": None,
        "error": None,
        "timestamp": datetime.now().isoformat(),
        "request_id": str(uuid4())
    }

    try:
        if hasattr(result, 'content'):
            parsed_data = result.content
        elif isinstance(result, str):
            try:
                parsed_data = json.loads(result)
            except json.JSONDecodeError:
                parsed_data = result
        else:
            parsed_data = result
        
        if isinstance(parsed_data, dict) and "success" in parsed_data:
            response_dict = parsed_data
        else:
            response_dict["data"] = parsed_data

    except Exception as e:
        response_dict["success"] = False
        response_dict["error"] = str(e)

    # 必须序列化为 JSON 字符串
    json_str = json.dumps(response_dict, ensure_ascii=False, indent=2, default=str)
    return [TextContent(type="text", text=json_str)]

# ============================================================================
# 工具定义 (无需修改，FastMCP 支持 @app.tool)
# ============================================================================

@app.tool()
async def query_uniprot(uniprot_id: str) -> List[TextContent]:
    try:
        result = await asyncio.to_thread(uniprot_query_tool.invoke, {"uniprot_id": uniprot_id})
        return format_tool_response(result)
    except Exception as e:
        return format_tool_response({"success": False, "error": str(e)})

@app.tool()
async def query_interpro(uniprot_id: str) -> List[TextContent]:
    try:
        result = await asyncio.to_thread(interpro_query_tool.invoke, {"uniprot_id": uniprot_id})
        return format_tool_response(result)
    except Exception as e:
        return format_tool_response({"success": False, "error": str(e)})

@app.tool()
async def download_pdb_structure(pdb_id: str, output_format: str = "pdb") -> List[TextContent]:
    try:
        result = await asyncio.to_thread(pdb_structure_download_tool.invoke, {
            "pdb_id": pdb_id,
            "output_format": output_format
        })
        return format_tool_response(result)
    except Exception as e:
        return format_tool_response({"success": False, "error": str(e)})

@app.tool()
async def download_ncbi_sequence(accession_id: str, output_format: str = "fasta") -> List[TextContent]:
    try:
        result = await asyncio.to_thread(ncbi_sequence_download_tool.invoke, {
            "accession_id": accession_id,
            "output_format": output_format
        })
        return format_tool_response(result)
    except Exception as e:
        return format_tool_response({"success": False, "error": str(e)})

@app.tool()
async def download_alphafold_structure(uniprot_id: str, output_format: str = "pdb") -> List[TextContent]:
    try:
        result = await asyncio.to_thread(alphafold_structure_download_tool.invoke, {
            "uniprot_id": uniprot_id,
            "output_format": output_format
        })
        return format_tool_response(result)
    except Exception as e:
        return format_tool_response({"success": False, "error": str(e)})

@app.tool()
async def extract_pdb_sequence(pdb_file_path: str) -> List[TextContent]:
    try:
        result = await asyncio.to_thread(PDB_sequence_extraction_tool.invoke, {"pdb_file": pdb_file_path})
        return format_tool_response(result)
    except Exception as e:
        return format_tool_response({"success": False, "error": str(e)})

@app.tool()
async def predict_zero_shot_sequence(
    sequence: Optional[str] = None, 
    fasta_file: Optional[str] = None,
    model_name: str = "ESM2-650M"
) -> List[TextContent]:
    params = {"model_name": model_name}
    if fasta_file:
        params["fasta_file"] = fasta_file
    elif sequence:
        params["sequence"] = sequence
    else:
        return format_tool_response({"success": False, "error": "Either sequence or fasta_file is required"})
        
    try:
        result = await asyncio.to_thread(zero_shot_sequence_prediction_tool.invoke, params)
        return format_tool_response(result)
    except Exception as e:
        return format_tool_response({"success": False, "error": str(e)})

@app.tool()
async def predict_zero_shot_structure(
    structure_file_path: str,
    model_name: str = "ESM-IF1"
) -> List[TextContent]:
    try:
        result = await asyncio.to_thread(zero_shot_structure_prediction_tool.invoke, {
            "structure_file": structure_file_path,
            "model_name": model_name
        })
        return format_tool_response(result)
    except Exception as e:
        return format_tool_response({"success": False, "error": str(e)})

@app.tool()
async def predict_protein_function(
    sequence: Optional[str] = None,
    fasta_file: Optional[str] = None,
    model_name: str = "ESM2-650M",
    task: str = "Solubility"
) -> List[TextContent]:
    params = {
        "model_name": model_name,
        "task": task
    }
    if fasta_file:
        params["fasta_file"] = fasta_file
    elif sequence:
        params["sequence"] = sequence
    else:
        return format_tool_response({"success": False, "error": "Either sequence or fasta_file is required"})

    try:
        result = await asyncio.to_thread(protein_function_prediction_tool.invoke, params)
        return format_tool_response(result)
    except Exception as e:
        return format_tool_response({"success": False, "error": str(e)})

@app.tool()
async def predict_functional_residue(
    sequence: Optional[str] = None,
    fasta_file: Optional[str] = None,
    model_name: str = "ESM2-650M",
    task: str = "Activity Site"
) -> List[TextContent]:
    params = {
        "model_name": model_name,
        "task": task
    }
    if fasta_file:
        params["fasta_file"] = fasta_file
    elif sequence:
        params["sequence"] = sequence
    else:
        return format_tool_response({"success": False, "error": "Either sequence or fasta_file is required"})

    try:
        result = await asyncio.to_thread(functional_residue_prediction_tool.invoke, params)
        return format_tool_response(result)
    except Exception as e:
        return format_tool_response({"success": False, "error": str(e)})

@app.tool()
async def predict_protein_properties(
    sequence: Optional[str] = None,
    fasta_file: Optional[str] = None,
    task_name: str = "Physical and chemical properties"
) -> List[TextContent]:
    params = {
        "task_name": task_name
    }
    if fasta_file:
        params["fasta_file"] = fasta_file
    elif sequence:
        params["sequence"] = sequence
    else:
        return format_tool_response({"success": False, "error": "Either sequence or fasta_file is required"})

    try:
        result = await asyncio.to_thread(protein_properties_generation_tool.invoke, params)
        return format_tool_response(result)
    except Exception as e:
        return format_tool_response({"success": False, "error": str(e)})

@app.tool()
async def generate_training_config(
    csv_file_path: str,
    test_csv_file_path: Optional[str] = None,
    output_name: str = "custom_training_config"
) -> List[TextContent]:
    try:
        result = await asyncio.to_thread(generate_training_config_tool.invoke, {
            "csv_file": csv_file_path,
            "test_csv_file": test_csv_file_path,
            "output_name": output_name
        })
        return format_tool_response(result)
    except Exception as e:
        return format_tool_response({"success": False, "error": str(e)})

@app.tool()
async def execute_ai_code(
    task_description: str,
    input_files: List[str] = Field(default_factory=list)
) -> List[TextContent]:
    try:
        result = await asyncio.to_thread(ai_code_execution_tool.invoke, {
            "task_description": task_description,
            "input_files": input_files
        })
        return format_tool_response(result)
    except Exception as e:
        return format_tool_response({"success": False, "error": str(e)})

@app.tool()
async def search_literature(query: str, max_results: int = 5) -> List[TextContent]:
    try:
        result = await asyncio.to_thread(literature_search_tool.invoke, {
            "query": query,
            "max_results": max_results
        })
        return format_tool_response(result)
    except Exception as e:
        return format_tool_response({"success": False, "error": str(e)})

# ============================================================================
# Main Loop (FastMCP 写法更简洁)
# ============================================================================

if __name__ == "__main__":
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    logger.info("VenusFactory MCP Server starting (FastMCP mode)...")
    
    # FastMCP 自带运行逻辑，不需要手写 stdio 循环
    app.run(transport="stdio")