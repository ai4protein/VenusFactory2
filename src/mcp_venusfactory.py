import os
import json
import logging
import asyncio
import threading
import time
import requests
import uuid
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
from uuid import uuid4
from urllib.parse import urlparse
from pydantic import BaseModel, Field, validator, field_validator
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
    dataset_search_tool,
    web_search_tool,
    protein_structure_prediction_ESMFold_tool,
    foldseek_search_tool,
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

# ==================== Pydantic Models for Input Validation ====================

class UniProtQueryInput(BaseModel):
    """Input validation for UniProt query."""
    uniprot_id: str = Field(..., min_length=1, description="UniProt ID of the protein")
    
    @field_validator('uniprot_id')
    @classmethod
    def validate_uniprot_id(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("UniProt ID cannot be empty")
        return v.strip()

class InterProQueryInput(BaseModel):
    """Input validation for InterPro query."""
    uniprot_id: str = Field(..., min_length=1, description="UniProt ID of the protein")
    
    @field_validator('uniprot_id')
    @classmethod
    def validate_uniprot_id(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("UniProt ID cannot be empty")
        return v.strip()

class PDBStructureDownloadInput(BaseModel):
    """Input validation for PDB structure download."""
    pdb_id: str = Field(..., min_length=4, max_length=4, description="PDB ID (4 characters)")
    output_format: str = Field(default="pdb", description="Output format (pdb, cif, etc.)")
    
    @field_validator('pdb_id')
    @classmethod
    def validate_pdb_id(cls, v: str) -> str:
        v = v.strip().upper()
        if len(v) != 4:
            raise ValueError("PDB ID must be exactly 4 characters")
        return v
    
    @field_validator('output_format')
    @classmethod
    def validate_format(cls, v: str) -> str:
        allowed_formats = ['pdb', 'cif', 'mmcif']
        if v.lower() not in allowed_formats:
            raise ValueError(f"Format must be one of: {', '.join(allowed_formats)}")
        return v.lower()

class NCBISequenceDownloadInput(BaseModel):
    """Input validation for NCBI sequence download."""
    accession_id: str = Field(..., min_length=1, description="NCBI accession ID")
    output_format: str = Field(default="fasta", description="Output format")
    
    @field_validator('accession_id')
    @classmethod
    def validate_accession_id(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Accession ID cannot be empty")
        return v.strip()

class AlphaFoldStructureDownloadInput(BaseModel):
    """Input validation for AlphaFold structure download."""
    uniprot_id: str = Field(..., min_length=1, description="UniProt ID")
    output_format: str = Field(default="pdb", description="Output format")
    
    @field_validator('uniprot_id')
    @classmethod
    def validate_uniprot_id(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("UniProt ID cannot be empty")
        return v.strip()

class PDBSequenceExtractionInput(BaseModel):
    """Input validation for PDB sequence extraction."""
    pdb_file_path: str = Field(..., min_length=1, description="Path to PDB file")
    
    @field_validator('pdb_file_path')
    @classmethod
    def validate_file_path(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("File path cannot be empty")
        path = Path(v.strip())
        if not path.suffix.lower() in ['.pdb', '.cif', '.ent']:
            raise ValueError("File must be a PDB structure file (.pdb, .cif, or .ent)")
        return str(path)

class ProteinStructurePredictionInput(BaseModel):
    """Input validation for protein structure prediction using ESMFold."""
    sequence: str = Field(..., min_length=1, description="Protein sequence")
    save_path: Optional[str] = Field(None, description="Path to save the predicted structure")
    verbose: Optional[bool] = Field(default=True, description="Whether to print detailed information")
    
    @field_validator('sequence')
    @classmethod
    def validate_sequence(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Sequence cannot be empty")
        return v.strip()

class FoldSeekSearchInput(BaseModel):
    """Input for FoldSeek search"""
    pdb_file_path: str = Field(..., min_length=1, description="Path to PDB file")
    protect_start: int = Field(..., description="Start position of the protected region")
    protect_end: int = Field(..., description="End position of the protected region")

    @field_validator('pdb_file_path')
    @classmethod
    def validate_file_path(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("File path cannot be empty")
        path = Path(v.strip())
        if not path.suffix.lower() in ['.pdb', '.cif', '.ent']:
            raise ValueError("File must be a PDB structure file (.pdb, .cif, or .ent)")
        return str(path)
    

class ZeroShotSequencePredictionInput(BaseModel):
    """Input validation for zero-shot sequence prediction."""
    sequence: Optional[str] = Field(None, description="Protein sequence")
    fasta_file: Optional[str] = Field(None, description="Path to FASTA file")
    model_name: str = Field(default="ESM2-650M", description="Model name")
    
    @field_validator('model_name')
    @classmethod
    def validate_model_name(cls, v: str) -> str:
        allowed_models = ["VenusPLM", "ESM2-650M", "ESM-1b", "ESM-1v"]
        if v not in allowed_models:
            raise ValueError(f"Model must be one of: {', '.join(allowed_models)}")
        return v
    
    def model_post_init(self, __context: Any) -> None:
        """Validate that either sequence or fasta_file is provided."""
        if not self.sequence and not self.fasta_file:
            raise ValueError("Either sequence or fasta_file must be provided")
        if self.sequence and self.fasta_file:
            raise ValueError("Provide either sequence or fasta_file, not both")

class ZeroShotStructurePredictionInput(BaseModel):
    """Input validation for zero-shot structure prediction."""
    structure_file_path: str = Field(..., min_length=1, description="Path to structure file")
    model_name: str = Field(default="ESM-IF1", description="Model name")
    
    @field_validator('model_name')
    @classmethod
    def validate_model_name(cls, v: str) -> str:
        allowed_models = ["VenusREM (foldseek-based)", "ProSST-2048", "ProtSSN", "ESM-IF1", "SaProt", "MIF-ST"]
        if v not in allowed_models:
            raise ValueError(f"Model must be one of: {', '.join(allowed_models)}")
        return v
    
    @field_validator('structure_file_path')
    @classmethod
    def validate_file_path(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("File path cannot be empty")
        return v.strip()

class ProteinFunctionPredictionInput(BaseModel):
    """Input validation for protein function prediction."""
    sequence: Optional[str] = Field(None, description="Protein sequence")
    fasta_file: Optional[str] = Field(None, description="Path to FASTA file")
    model_name: str = Field(default="ESM2-650M", description="Model name")
    task: str = Field(default="Solubility", description="Prediction task")
    
    @field_validator('model_name')
    @classmethod
    def validate_model_name(cls, v: str) -> str:
        allowed_models = ["ESM2-650M", "Ankh-large", "ProtBert", "ProtT5-xl-uniref50"]
        if v not in allowed_models:
            raise ValueError(f"Model must be one of: {', '.join(allowed_models)}")
        return v
    
    @field_validator('task')
    @classmethod
    def validate_task(cls, v: str) -> str:
        allowed_tasks = [
            "Solubility", "Subcellular Localization", "Membrane Protein", 
            "Metal Ion Binding", "Stability", "Sortingsignal", "Optimal Temperature",
            "Kcat", "Optimal PH", "Immunogenicity Prediction - Virus",
            "Immunogenicity Prediction - Bacteria", "Immunogenicity Prediction - Tumor"
        ]
        if v not in allowed_tasks:
            raise ValueError(f"Task must be one of: {', '.join(allowed_tasks)}")
        return v
    
    def model_post_init(self, __context: Any) -> None:
        """Validate that either sequence or fasta_file is provided."""
        if not self.sequence and not self.fasta_file:
            raise ValueError("Either sequence or fasta_file must be provided")
        if self.sequence and self.fasta_file:
            raise ValueError("Provide either sequence or fasta_file, not both")

class FunctionalResiduePredictionInput(BaseModel):
    """Input validation for functional residue prediction."""
    sequence: Optional[str] = Field(None, description="Protein sequence")
    fasta_file: Optional[str] = Field(None, description="Path to FASTA file")
    model_name: str = Field(default="ESM2-650M", description="Model name")
    task: str = Field(default="Activity Site", description="Prediction task")
    
    @field_validator('model_name')
    @classmethod
    def validate_model_name(cls, v: str) -> str:
        allowed_models = ["ESM2-650M", "Ankh-large", "ProtT5-xl-uniref50"]
        if v not in allowed_models:
            raise ValueError(f"Model must be one of: {', '.join(allowed_models)}")
        return v
    
    @field_validator('task')
    @classmethod
    def validate_task(cls, v: str) -> str:
        allowed_tasks = ["Activity Site", "Binding Site", "Conserved Site", "Motif"]
        if v not in allowed_tasks:
            raise ValueError(f"Task must be one of: {', '.join(allowed_tasks)}")
        return v
    
    def model_post_init(self, __context: Any) -> None:
        """Validate that either sequence or fasta_file is provided."""
        if not self.sequence and not self.fasta_file:
            raise ValueError("Either sequence or fasta_file must be provided")
        if self.sequence and self.fasta_file:
            raise ValueError("Provide either sequence or fasta_file, not both")

class ProteinPropertiesPredictionInput(BaseModel):
    """Input validation for protein properties prediction."""
    sequence: Optional[str] = Field(None, description="Protein sequence")
    fasta_file: Optional[str] = Field(None, description="Path to FASTA file")
    task_name: str = Field(default="Physical and chemical properties", description="Task name")
    
    @field_validator('task_name')
    @classmethod
    def validate_task_name(cls, v: str) -> str:
        allowed_tasks = [
            "Physical and chemical properties",
            "Relative solvent accessible surface area (PDB only)",
            "SASA value (PDB only)",
            "Secondary structure (PDB only)"
        ]
        if v not in allowed_tasks:
            raise ValueError(f"Task must be one of: {', '.join(allowed_tasks)}")
        return v
    
    def model_post_init(self, __context: Any) -> None:
        """Validate that either sequence or fasta_file is provided."""
        if not self.sequence and not self.fasta_file:
            raise ValueError("Either sequence or fasta_file must be provided")
        if self.sequence and self.fasta_file:
            raise ValueError("Provide either sequence or fasta_file, not both")

class LiteratureSearchInput(BaseModel):
    """Input validation for literature search."""
    query: str = Field(..., min_length=1, description="Search query")
    max_results: int = Field(default=5, ge=1, le=100, description="Maximum number of results")
    source: str = Field(..., min_length=1, description="Source")

    @field_validator('query')
    @classmethod
    def validate_query(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Query cannot be empty")
        return v.strip()

class DatasetSearchInput(BaseModel):
    """Input validation for dataset search."""
    query: str = Field(..., min_length=1, description="Search query")
    max_results: int = Field(default=5, ge=1, le=100, description="Maximum number of results")
    source: str = Field(..., min_length=1, description="Source")

    @field_validator('query')
    @classmethod
    def validate_query(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Query cannot be empty")
        return v.strip()
    
class WebSearchInput(BaseModel):
    """Input validation for dataset search."""
    query: str = Field(..., min_length=1, description="Search query")
    max_results: int = Field(default=5, ge=1, le=100, description="Maximum number of results")
    source: str = Field(..., min_length=1, description="Source")
        
    @field_validator('query')
    @classmethod
    def validate_query(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Query cannot be empty")
        return v.strip()

class MCPError(BaseModel):
    """Error information in MCP response."""
    code: str = Field(..., description="Error code (e.g., VALIDATION_ERROR, EXECUTION_ERROR)")
    message: str = Field(..., description="Human-readable error message")
    detail: Optional[Dict[str, Any]] = Field(None, description="Additional error details")

class MCPResponse(BaseModel):
    """Unified MCP response format."""
    success: bool = Field(..., description="Whether the operation was successful")
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z", description="Response timestamp in ISO format")
    request_id: str = Field(default_factory=lambda: str(uuid4()), description="Unique request identifier")
    data: Optional[Any] = Field(None, description="Response data (dict, list, or string)")
    error: Optional[MCPError] = Field(None, description="Error information if success is False")
    
    def to_json(self) -> str:
        """Convert response to JSON string."""
        return json.dumps(self.model_dump(exclude_none=True), ensure_ascii=False, indent=2)
        
def build_success_response(data: Any, request_id: Optional[str] = None) -> MCPResponse:
    """
    Build a successful MCP response.
    
    Args:
        data: Response data (can be dict, list, string, or any JSON-serializable object)
        request_id: Optional request ID (auto-generated if not provided)
    
    Returns:
        MCPResponse with success=True
    """
    return MCPResponse(
        success=True,
        request_id=request_id or str(uuid4()),
        data=data,
        error=None
    )

def build_error_response(
    message: str,
    code: str = "ERROR",
    detail: Optional[Dict[str, Any]] = None,
    request_id: Optional[str] = None
) -> MCPResponse:
    """
    Build an error MCP response.
    
    Args:
        message: Error message
        code: Error code (e.g., VALIDATION_ERROR, EXECUTION_ERROR)
        detail: Additional error details
        request_id: Optional request ID (auto-generated if not provided)
    
    Returns:
        MCPResponse with success=False
    """
    return MCPResponse(
        success=False,
        request_id=request_id or str(uuid4()),
        data=None,
        error=MCPError(code=code, message=message, detail=detail)
    )

def download_file_from_url(url: str, target_dir: str = "Downloads") -> str:
    """
    Download file from URL to specified directory.
    
    Args:
        url: URL to download from
        target_dir: Target directory name (will be created under temp_outputs)
    
    Returns:
        Local file path of downloaded file
    
    Raises:
        Exception: If download fails
    """
    try:
        # Parse URL to get filename
        parsed_url = urlparse(url)
        filename = os.path.basename(parsed_url.path)
        
        # If no filename in URL, generate one with .txt extension
        if not filename or '.' not in filename:
            filename = f"downloaded_file_{uuid.uuid4().hex[:8]}.txt"
        
        # Create target directory using get_save_path
        save_dir = get_save_path("MCP_Server", target_dir)
        file_path = save_dir / filename
        
        # Download file
        logger.info(f"Downloading file from URL: {url}")
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # Save file
        with open(file_path, 'wb') as f:
            f.write(response.content)
        
        logger.info(f"File downloaded successfully to: {file_path}")
        return str(file_path)
        
    except Exception as e:
        logger.error(f"Failed to download file from URL {url}: {str(e)}")
        raise Exception(f"Failed to download file from URL: {str(e)}")

def is_url(path: str) -> bool:
    """Check if a string is a URL."""
    try:
        result = urlparse(path)
        return all([result.scheme, result.netloc])
    except:
        return False

def process_file_path(file_path: str, target_dir: str = "Downloads") -> str:
    """
    Process file path - if it's a URL, download it; otherwise return as is.
    
    Args:
        file_path: File path or URL
        target_dir: Target directory for downloads
    
    Returns:
        Local file path
    """
    if is_url(file_path):
        return download_file_from_url(file_path, target_dir)
    return file_path

def format_tool_response(result: Any, error: Optional[Exception] = None) -> str:
    """
    Format tool response into unified JSON structure.
    
    Args:
        result: Tool execution result
        error: Optional exception if tool execution failed
    
    Returns:
        JSON string with unified MCP response format
    """
    try:
        if error:
            # Handle error case
            if isinstance(error, ValueError):
                return build_error_response(
                    message=str(error),
                    code="VALIDATION_ERROR"
                ).to_json()
            else:
                return build_error_response(
                    message=str(error),
                    code="EXECUTION_ERROR"
                ).to_json()
        
        # Handle success case
        if hasattr(result, 'content'):
            # LangChain tool result
            data = str(result.content)
        elif isinstance(result, (dict, list)):
            data = result
        else:
            data = str(result)
        
        return build_success_response(data=data).to_json()
        
    except Exception as e:
        # Fallback error handling
        return build_error_response(
            message=f"Error formatting response: {str(e)}",
            code="FORMATTING_ERROR"
        ).to_json()


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
            app_with_route = mcp.sse_app()
            uvicorn.run(
                app_with_route,
                host=host,
                port=port
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


@mcp.tool()
async def query_uniprot(uniprot_id: str) -> str:
    """
    Retrieve protein amino acid sequence and metadata from UniProt database by accession ID.
    Args:
        uniprot_id (str): UniProt ID of the protein.
    Returns:
        str: JSON-formatted protein information.
    """
    try:
        # Validate input using Pydantic
        validated_input = UniProtQueryInput(uniprot_id=uniprot_id)
        result = await asyncio.to_thread(uniprot_query_tool.invoke, {"uniprot_id": validated_input.uniprot_id})
        return format_tool_response(result)
    except ValueError as e:
        return format_tool_response(None, error=e)
    except Exception as e:
        return format_tool_response(None, error=e)

@mcp.tool()
async def query_interpro(uniprot_id: str) -> str:
    """
    Retrieve protein domain annotations, GO terms, and functional information from InterPro database.
    Args:
        uniprot_id (str): UniProt ID of the protein.
    Returns:
        str: JSON-formatted domain information.
    """
    try:
        # Validate input using Pydantic
        validated_input = InterProQueryInput(uniprot_id=uniprot_id)
        result = await asyncio.to_thread(interpro_query_tool.invoke, {"uniprot_id": validated_input.uniprot_id})
        return format_tool_response(result)
    except ValueError as e:
        return format_tool_response(None, error=e)
    except Exception as e:
        return format_tool_response(None, error=e)

@mcp.tool()
async def download_pdb_structure(pdb_id: str, output_format: str = "pdb") -> str:
    """
    Download experimental 3D protein structure from RCSB PDB database and save to local file.
    Args:
        pdb_id (str): PDB ID of the structure.
        output_format (str): Output the path of the structure file.
    Returns:
        str: Path to the downloaded structure file.
    """
    try:
        # Validate input using Pydantic
        validated_input = PDBStructureDownloadInput(pdb_id=pdb_id, output_format=output_format)
        result = await asyncio.to_thread(pdb_structure_download_tool.invoke, {
            "pdb_id": validated_input.pdb_id,
            "output_format": validated_input.output_format
        })
        return format_tool_response(result)
    except ValueError as e:
        return format_tool_response(None, error=e)
    except Exception as e:
        return format_tool_response(None, error=e)

@mcp.tool()
async def download_ncbi_sequence(accession_id: str, output_format: str = "fasta") -> str:
    """
    Download protein or nucleotide sequence from NCBI database and save as FASTA file.
    Args:
        accession_id (str): Accession ID of the sequence.
        output_format (str): Output the path of the sequence file.
    Returns:
        str: Path to the downloaded sequence file.
    """
    try:
        # Validate input using Pydantic
        validated_input = NCBISequenceDownloadInput(accession_id=accession_id, output_format=output_format)
        result = await asyncio.to_thread(ncbi_sequence_download_tool.invoke, {
            "accession_id": validated_input.accession_id,
            "output_format": validated_input.output_format
        })
        return format_tool_response(result)
    except ValueError as e:
        return format_tool_response(None, error=e)
    except Exception as e:
        return format_tool_response(None, error=e)

@mcp.tool()
async def download_alphafold_structure(uniprot_id: str, output_format: str = "pdb") -> str:
    """
    Download AI-predicted 3D protein structure from AlphaFold database with confidence scores.
    Args:
        uniprot_id (str): UniProt ID of the protein.
        output_format (str): Output the path of the structure file.
    Returns:
        str: Path to the downloaded structure file.
    """
    try:
        # Validate input using Pydantic
        validated_input = AlphaFoldStructureDownloadInput(uniprot_id=uniprot_id, output_format=output_format)
        result = await asyncio.to_thread(alphafold_structure_download_tool.invoke, {
            "uniprot_id": validated_input.uniprot_id,
            "output_format": validated_input.output_format
        })
        return format_tool_response(result)
    except ValueError as e:
        return format_tool_response(None, error=e)
    except Exception as e:
        return format_tool_response(None, error=e)

@mcp.tool()
async def extract_pdb_sequence(pdb_file_path: str) -> str:
    """
    Extract amino acid sequences from PDB structure file for all protein chains.
    Args:
        pdb_file_path (str): Path to the PDB file or URL.
    Returns:
        str: Extracted sequence.
    """
    try:
        # Process file path (download if URL)
        processed_file_path = process_file_path(pdb_file_path, "PDB_Files")
        
        # Validate input using Pydantic
        validated_input = PDBSequenceExtractionInput(pdb_file_path=processed_file_path)
        result = await asyncio.to_thread(PDB_sequence_extraction_tool.invoke, {"pdb_file": validated_input.pdb_file_path})
        return format_tool_response(result)
    except ValueError as e:
        return format_tool_response(None, error=e)
    except Exception as e:
        return format_tool_response(None, error=e)


@mcp.tool()
async def foldseek_search(
    pdb_file_path: str,
    protect_start: int,
    protect_end: int
) -> str:
    """
    Search for protein structures using FoldSeek.
    Args:
        pdb_file_path (str): Path to the PDB file or URL.
        protect_start (int): Start position of the protected region.
        protect_end (int): End position of the protected region.
    Returns:
        str: FASTA file containing the sequences of the protected region.
    """
    try:
        # Process file path (download if URL)
        processed_file_path = process_file_path(pdb_file_path, "PDB_Files")
        
        # Validate input using Pydantic
        validated_input = FoldSeekSearchInput(pdb_file_path=processed_file_path, protect_start=protect_start, protect_end=protect_end)
        params = {
            "pdb_file_path": validated_input.pdb_file_path,
            "protect_start": validated_input.protect_start,
            "protect_end": validated_input.protect_end
        }
        result = await asyncio.to_thread(foldseek_search_tool.invoke, params)
        return format_tool_response(result)
    except ValueError as e:
        return format_tool_response(None, error=e)
    except Exception as e:
        return format_tool_response(None, error=e)

@mcp.tool()
async def Protein_structure_prediction_ESMFold(sequence: str) -> str:
    """
    Predict protein structure using ESMFold.
    Args:
        sequence (str): Protein sequence.
    Returns:
        str: Path to the predicted structure file.
    """
    try:
        # Validate input using Pydantic
        validated_input = ProteinStructurePredictionInput(sequence=sequence)
        result = await asyncio.to_thread(protein_structure_prediction_ESMFold_tool.invoke, {"sequence": validated_input.sequence})
        return format_tool_response(result)
    except ValueError as e:
        return format_tool_response(None, error=e)
    except Exception as e:
        return format_tool_response(None, error=e)

@mcp.tool()
async def zero_shot_sequence_prediction(
    sequence: Optional[str] = None, 
    fasta_file: Optional[str] = None,
    model_name: str = "ESM2-650M"
) -> str:
    """
    Predict beneficial single-point mutations from protein sequence using pre-trained language models (no training data required).
    Args:
        sequence (Optional[str]): Protein sequence.
        fasta_file (Optional[str]): Path to the FASTA file.
        model_name (str): Model name for prediction. Support Model: VenusPLM, ESM2-650M, ESM-1b, ESM-1v
    Returns:
        str: Prediction result.
    """
    try:
        # Process fasta_file if it's a URL
        processed_fasta_file = None
        if fasta_file:
            processed_fasta_file = process_file_path(fasta_file, "FASTA_Files")
        
        # Validate input using Pydantic
        validated_input = ZeroShotSequencePredictionInput(
            sequence=sequence,
            fasta_file=processed_fasta_file,
            model_name=model_name
        )
        
        params = {"model_name": validated_input.model_name}
        if validated_input.fasta_file:
            params["fasta_file"] = validated_input.fasta_file
        elif validated_input.sequence:
            params["sequence"] = validated_input.sequence
            
        result = await asyncio.to_thread(zero_shot_sequence_prediction_tool.invoke, params)
        return format_tool_response(result)
    except ValueError as e:
        return format_tool_response(None, error=e)
    except Exception as e:
        return format_tool_response(None, error=e)


@mcp.tool()
async def predict_zero_shot_structure(
    structure_file_path: str,
    model_name: str = "ESM-IF1"
) -> str:
    """
    Predict beneficial single-point mutations from 3D protein structure using pre-trained models (no training data required).
    Args:
        structure_file_path (str): Path to the structure file.
        model_name (str): Model name for prediction. Supported models: VenusREM (foldseek-based), ProSST-2048, ProtSSN, ESM-IF1, SaProt, MIF-ST
    Returns:
        str: Prediction result.
    """
    try:
        # Process structure file path (download if URL)
        processed_structure_file = process_file_path(structure_file_path, "Structure_Files")
        
        # Validate input using Pydantic
        validated_input = ZeroShotStructurePredictionInput(
            structure_file_path=processed_structure_file,
            model_name=model_name
        )
        result = await asyncio.to_thread(zero_shot_structure_prediction_tool.invoke, {
            "structure_file": validated_input.structure_file_path,
            "model_name": validated_input.model_name
        })
        return format_tool_response(result)
    except ValueError as e:
        return format_tool_response(None, error=e)
    except Exception as e:
        return format_tool_response(None, error=e)

@mcp.tool()
async def predict_protein_function(
    sequence: Optional[str] = None,
    fasta_file: Optional[str] = None,
    model_name: str = "ESM2-650M",
    task: str = "Solubility"
) -> str:
    """
    Predict protein functional properties (solubility, localization, stability, etc.) from amino acid sequence.
    Args:
        sequence (Optional[str]): Protein sequence.
        fasta_file (Optional[str]): Path to the FASTA file.
        model_name (str): Model name for prediction. Support Models: ESM2-650M, Ankh-large, ProtBert, ProtT5-xl-uniref50 
        task (str): Task for prediction. Support Task: Solubility, Subcellular Localization, Membrane Protein, Metal Ion Binding, Stability, Sortingsignal, Optimal Temperature, Kcat, Optimal PH, Immunogenicity Prediction - Virus, Immunogenicity Prediction - Bacteria, Immunogenicity Prediction - Tumor
    Returns:
        str: Prediction result.
    """
    try:
        # Process fasta_file if it's a URL
        processed_fasta_file = None
        if fasta_file:
            processed_fasta_file = process_file_path(fasta_file, "FASTA_Files")
        
        # Validate input using Pydantic
        validated_input = ProteinFunctionPredictionInput(
            sequence=sequence,
            fasta_file=processed_fasta_file,
            model_name=model_name,
            task=task
        )
        
        params = {
            "model_name": validated_input.model_name,
            "task": validated_input.task
        }
        if validated_input.fasta_file:
            params["fasta_file"] = validated_input.fasta_file
        elif validated_input.sequence:
            params["sequence"] = validated_input.sequence

        result = await asyncio.to_thread(protein_function_prediction_tool.invoke, params)
        return format_tool_response(result)
    except ValueError as e:
        return format_tool_response(None, error=e)
    except Exception as e:
        return format_tool_response(None, error=e)

@mcp.tool()
async def predict_functional_residue(
    sequence: Optional[str] = None,
    fasta_file: Optional[str] = None,
    model_name: str = "ESM2-650M",
    task: str = "Activity Site"
) -> str:
    """
    Identify functional residues (active sites, binding sites, conserved sites) in protein sequence.
    Args:
        sequence (Optional[str]): Protein sequence.
        fasta_file (Optional[str]): Path to the FASTA file.
        model_name (str): Model name for prediction. Support Models: ESM2-650M, Ankh-large, ProtT5-xl-uniref50 
        task (str): Task for prediction. Support Task: Activity Site, Binding Site, Conserved Site, Motif
    Returns:
        str: Prediction result.
    """
    try:
        # Process fasta_file if it's a URL
        processed_fasta_file = None
        if fasta_file:
            processed_fasta_file = process_file_path(fasta_file, "FASTA_Files")
        print("processed_fasta_file", processed_fasta_file)
        # Validate input using Pydantic
        validated_input = FunctionalResiduePredictionInput(
            sequence=sequence,
            fasta_file=processed_fasta_file,
            model_name=model_name,
            task=task
        )
        
        params = {
            "model_name": validated_input.model_name,
            "task": validated_input.task
        }
        if validated_input.fasta_file:
            params["fasta_file"] = validated_input.fasta_file
        elif validated_input.sequence:
            params["sequence"] = validated_input.sequence

        result = await asyncio.to_thread(functional_residue_prediction_tool.invoke, params)
        return format_tool_response(result)
    except ValueError as e:
        return format_tool_response(None, error=e)
    except Exception as e:
        return format_tool_response(None, error=e)

@mcp.tool()
async def predict_protein_properties(
    sequence: Optional[str] = None,
    fasta_file: Optional[str] = None,
    task_name: str = "Physical and chemical properties"
) -> str:
    """
    Calculate physical and chemical properties (molecular weight, pI, SASA, secondary structure) from protein sequence or structure.
    Args:
        sequence (Optional[str]): Protein sequence.
        fasta_file (Optional[str]): Path to the PDB file.
        task_name (str): Task name for prediction. Support Task: Physical and chemical properties, Relative solvent accessible surface area (PDB only), SASA value (PDB only), Secondary structure (PDB only)
    Returns:
        str: Prediction result.
    """
    try:
        # Process fasta_file if it's a URL
        processed_fasta_file = None
        if fasta_file:
            processed_fasta_file = process_file_path(fasta_file, "Structure_Files")
        
        # Validate input using Pydantic
        validated_input = ProteinPropertiesPredictionInput(
            sequence=sequence,
            fasta_file=processed_fasta_file,
            task_name=task_name
        )
        
        params = {
            "task_name": validated_input.task_name
        }
        if validated_input.fasta_file:
            params["fasta_file"] = validated_input.fasta_file
        elif validated_input.sequence:
            params["sequence"] = validated_input.sequence

        result = await asyncio.to_thread(protein_properties_generation_tool.invoke, params)
        return format_tool_response(result)
    except ValueError as e:
        return format_tool_response(None, error=e)
    except Exception as e:
        return format_tool_response(None, error=e)


@mcp.tool()
async def search_literature(query: str, max_results: int = 5, source: str = "arxiv") -> str:
    """
    Search for academic literature using arXiv and PubMed.
    Use this tool for finding scientific papers, articles, and research publications.
    
    Args:
        query: The search query for academic literature (protein names, scientific concepts, etc.)
        max_results: The maximum number of results to return.
        source: The source to search: arxiv, pubmed, biorxiv, semantic_scholar 
        
    Returns:
        A JSON string containing the academic literature search results.
    """
    try:
        # Validate input using Pydantic
        validated_input = LiteratureSearchInput(query=query, max_results=max_results, source=source)
        result = await asyncio.to_thread(literature_search_tool.invoke, {
            "query": validated_input.query,
            "max_results": validated_input.max_results,
            "source": validated_input.source
        })
        return format_tool_response(result)
    except ValueError as e:
        return format_tool_response(None, error=e)
    except Exception as e:
        return format_tool_response(None, error=e)

@mcp.tool()
async def search_dataset(query: str, max_results: int = 5, source: str = "github") -> str:
    """
    Search for datasets using GitHub and Hugging Face.
    Use this tool for finding datasets for model training.
    
    Args:
        query: The search query for datasets (protein names, scientific concepts, etc.)
        max_results: The maximum number of results to return.
        source: The source to search: github, hugging_face, or all
        
    Returns:
        A JSON string containing the dataset search results.
    """
    try:
        # Validate input using Pydantic
        validated_input = DatasetSearchInput(query=query, max_results=max_results, source=source)
        result = await asyncio.to_thread(dataset_search_tool.invoke, {
            "query": validated_input.query,
            "max_results": validated_input.max_results,
            "source": validated_input.source
        })
        return format_tool_response(result)
    except ValueError as e:
        return format_tool_response(None, error=e)
    except Exception as e:
        return format_tool_response(None, error=e)

@mcp.tool()
async def search_web(query: str, max_results: int = 5, source: str = "duckduckgo") -> str:
    """
    Search the web using DuckDuckGo and Tavily.
    Use this tool for finding information on the web.
    
    Args:
        query: The search query for the web (protein names, scientific concepts, etc.)
        max_results: The maximum number of results to return.
        source: The source to search: duckduckgo, tavily, or all
        
    Returns:
        A JSON string containing the web search results.
    """
    try:
        # Validate input using Pydantic
        validated_input = WebSearchInput(query=query, max_results=max_results, source=source)
        result = await asyncio.to_thread(web_search_tool.invoke, {
            "query": validated_input.query,
            "max_results": validated_input.max_results,
            "source": validated_input.source
        })
        return format_tool_response(result)
    except ValueError as e:
        return format_tool_response(None, error=e)
    except Exception as e:
        return format_tool_response(None, error=e)

if __name__ == "__main__":    
    logger.info("VenusFactory MCP Server starting...")
    mcp.run(transport="sse")