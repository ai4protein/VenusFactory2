import os
import json
import shutil
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
from uuid import uuid4

from fastapi import FastAPI, UploadFile, File, HTTPException, Form, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import uvicorn

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

UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("temp_outputs")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# File constraints
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB
ALLOWED_EXTENSIONS = {
    "pdb": [".pdb"],
    "fasta": [".fasta", ".fa", ".faa"],
    "csv": [".csv"],
    "structure": [".pdb", ".cif", ".mmcif"],
}

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="VenusFactory API",
    description="API for protein sequence and structure analysis tools",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class StandardResponse(BaseModel):
    """Standard API response format"""
    success: bool
    data: Optional[Any] = None
    message: Optional[str] = None
    error: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    request_id: str = Field(default_factory=lambda: str(uuid4()))

class ExtractPDBSequenceRequest(BaseModel):
    """Request model for PDB sequence extraction"""
    pdb_file: str = Field(..., description="Path to PDB file")

class UniProtRequest(BaseModel):
    """Request model for UniProt query"""
    uniprot_id: str = Field(..., description="UniProt accession ID")


class InterProRequest(BaseModel):
    """Request model for InterPro query"""
    uniprot_id: str = Field(..., description="UniProt accession ID")


class PDBDownloadRequest(BaseModel):
    """Request model for PDB structure download"""
    pdb_id: str = Field(..., description="PDB ID (e.g., 1ABC)")
    output_format: str = Field(default="pdb", description="Output format: pdb or mmcif")


class NCBIDownloadRequest(BaseModel):
    """Request model for NCBI sequence download"""
    accession_id: str = Field(..., description="NCBI accession ID (e.g., NP_001234)")
    output_format: str = Field(default="fasta", description="Output format")


class AlphaFoldDownloadRequest(BaseModel):
    """Request model for AlphaFold structure download"""
    uniprot_id: str = Field(..., description="UniProt ID")
    output_format: str = Field(default="pdb", description="Output format: pdb or mmcif")


class ZeroShotSequenceRequest(BaseModel):
    """Request model for zero-shot sequence prediction"""
    sequence: Optional[str] = Field(None, description="Protein sequence in single letter amino acid code")
    fasta_file: Optional[str] = Field(None, description="Path to FASTA file")
    model_name: str = Field(default="ESM2-650M", description="Model name: ESM-1v, ESM2-650M, ESM-1b, VenusPLM")


class ZeroShotStructureRequest(BaseModel):
    """Request model for zero-shot structure prediction"""
    structure_file: str = Field(..., description="Path to PDB structure file")
    model_name: str = Field(default="ESM-IF1", description="Model name: SaProt, ProtSSN, ESM-IF1, MIF-ST, ProSST-2048")


class FunctionPredictionRequest(BaseModel):
    """Request model for protein function prediction"""
    sequence: Optional[str] = Field(None, description="Protein sequence in single letter amino acid code")
    fasta_file: Optional[str] = Field(None, description="Path to FASTA file")
    model_name: str = Field(default="ESM2-650M", description="Model name for function prediction")
    task: str = Field(default="Solubility", description="Task: Solubility, Subcellular Localization, Membrane Protein, Metal ion binding, Stability, Sortingsignal, Optimum temperature, Kcat, Optimal PH, Immunogenicity Prediction - Virus, Immunogenicity Prediction - Bacteria, Immunogenicity Prediction - Tumor")


class ResidueFunctionRequest(BaseModel):
    """Request model for functional residue prediction"""
    sequence: Optional[str] = Field(None, description="Protein sequence in single letter amino acid code")
    fasta_file: Optional[str] = Field(None, description="Path to FASTA file")
    model_name: str = Field(default="ESM2-650M", description="Model name for function prediction")
    task: str = Field(default="Activity Site", description="Task: Activity Site, Binding Site, Conserved Site, Motif")


class ProteinPropertiesRequest(BaseModel):
    """Request model for protein properties prediction"""
    sequence: Optional[str] = Field(None, description="Protein sequence in single letter amino acid code")
    fasta_file: Optional[str] = Field(None, description="Path to PDB structure file or fasta file")
    task_name: str = Field(default="Physical and chemical properties", description="Task name: Physical and chemical properties, Relative solvent accessible surface area (PDB only), SASA value (PDB only), Secondary structure (PDB only)")


class TrainingConfigRequest(BaseModel):
    """Request model for training config generation"""
    csv_file: str = Field(..., description="Path to CSV file with training data")
    test_csv_file: Optional[str] = Field(None, description="Optional path to test CSV file")
    output_name: str = Field(default="custom_training_config", description="Name for the generated config")


class CodeExecutionRequest(BaseModel):
    """Request model for AI code execution"""
    task_description: str = Field(..., description="Description of the task")
    input_files: List[str] = Field(default=[], description="List of input file paths")


class LiteratureSearchRequest(BaseModel):
    """Request model for literature search"""
    query: str = Field(..., description="Search query")
    max_results: int = Field(default=5, description="Maximum number of results")



def create_response(
    success: bool,
    data: Any = None,
    message: str = None,
    error: str = None
) -> Dict[str, Any]:
    """Create standardized API response"""
    return {
        "success": success,
        "data": data,
        "message": message,
        "error": error,
        "timestamp": datetime.now().isoformat(),
        "request_id": str(uuid4())
    }


def save_upload_file(upload_file: UploadFile, subdir: str = "") -> str:
    """
    Save uploaded file to temporary directory
    
    Args:
        upload_file: FastAPI UploadFile object
        subdir: Optional subdirectory within UPLOAD_DIR
        
    Returns:
        Absolute path to saved file
    """
    try:
        # Create subdirectory if specified
        if subdir:
            save_dir = UPLOAD_DIR / subdir
            save_dir.mkdir(exist_ok=True)
        else:
            save_dir = UPLOAD_DIR
        
        # Generate unique filename
        file_ext = Path(upload_file.filename).suffix
        unique_filename = f"{uuid4()}{file_ext}"
        file_path = save_dir / unique_filename
        
        # Save file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)
        
        logger.info(f"File saved: {file_path}")
        return str(file_path.absolute())
    
    except Exception as e:
        logger.error(f"Error saving file: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")


def validate_file(
    upload_file: UploadFile,
    allowed_extensions: List[str],
    max_size: int = MAX_FILE_SIZE
) -> None:
    """
    Validate uploaded file
    
    Args:
        upload_file: FastAPI UploadFile object
        allowed_extensions: List of allowed file extensions (e.g., ['.pdb', '.fasta'])
        max_size: Maximum file size in bytes
        
    Raises:
        HTTPException: If validation fails
    """
    # Check extension
    file_ext = Path(upload_file.filename).suffix.lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file extension. Allowed: {', '.join(allowed_extensions)}"
        )
    
    # Check file size (if possible)
    if hasattr(upload_file, 'size') and upload_file.size:
        if upload_file.size > max_size:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum size: {max_size / 1024 / 1024:.2f} MB"
            )


def parse_tool_result(result: str) -> Dict[str, Any]:
    """
    Parse tool function result (JSON string or plain text)
    
    Args:
        result: Result string from tool function
        
    Returns:
        Parsed dictionary with success, data, and error fields
    """
    try:
        # Try to parse as JSON
        if isinstance(result, str):
            parsed = json.loads(result)
        else:
            parsed = result
        
        # Check if it's a dict with success field
        if isinstance(parsed, dict):
            return {
                "success": parsed.get("success", True),
                "data": parsed,
                "error": parsed.get("error") or parsed.get("error_message")
            }
        else:
            # If not a dict, treat entire result as data
            return {
                "success": True,
                "data": parsed,
                "error": None
            }
    
    except json.JSONDecodeError:
        # If not JSON, treat as plain text
        if "error" in result.lower() or "failed" in result.lower():
            return {
                "success": False,
                "data": None,
                "error": result
            }
        else:
            return {
                "success": True,
                "data": {"result": result},
                "error": None
            }
    except Exception as e:
        return {
            "success": False,
            "data": None,
            "error": f"Failed to parse result: {str(e)}"
        }


def cleanup_file(file_path: str) -> None:
    """
    Delete temporary file
    
    Args:
        file_path: Path to file to delete
    """
    try:
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Cleaned up file: {file_path}")
    except Exception as e:
        logger.warning(f"Failed to cleanup file {file_path}: {e}")


def cleanup_files_background(file_paths: List[str]) -> None:
    """
    Background task to cleanup multiple files
    
    Args:
        file_paths: List of file paths to delete
    """
    for file_path in file_paths:
        cleanup_file(file_path)


@app.get("/health", response_model=StandardResponse)
async def health_check():
    """
    Health check endpoint to verify service status
    """
    try:
        # Check if required directories exist
        dirs_ok = UPLOAD_DIR.exists() and OUTPUT_DIR.exists()
        
        return create_response(
            success=True,
            data={
                "status": "healthy",
                "directories": {
                    "uploads": str(UPLOAD_DIR),
                    "outputs": str(OUTPUT_DIR),
                    "accessible": dirs_ok
                }
            },
            message="Service is running"
        )
    except Exception as e:
        return create_response(
            success=False,
            error=f"Health check failed: {str(e)}"
        )


@app.post("/api/uniprot/query", response_model=StandardResponse)
async def query_uniprot(request: UniProtRequest):
    """
    Query UniProt database for protein sequence
    
    Args:
        request: UniProtRequest with uniprot_id
        
    Returns:
        Protein sequence and metadata
    """
    try:
        logger.info(f"UniProt query: {request.uniprot_id}")
        
        # Call tool
        result = uniprot_query_tool.invoke({"uniprot_id": request.uniprot_id})
        
        # Parse result
        parsed = parse_tool_result(result)
        
        return create_response(
            success=parsed["success"],
            data=parsed["data"],
            error=parsed["error"],
            message=f"UniProt query completed for {request.uniprot_id}"
        )
    
    except Exception as e:
        logger.error(f"UniProt query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/interpro/query", response_model=StandardResponse)
async def query_interpro(request: InterProRequest):
    """
    Query InterPro database for protein function annotations
    
    Args:
        request: InterProRequest with uniprot_id
        
    Returns:
        Function annotations, GO terms, and metadata
    """
    try:
        logger.info(f"InterPro query: {request.uniprot_id}")
        
        # Call tool
        result = interpro_query_tool.invoke({"uniprot_id": request.uniprot_id})
        
        # Parse result
        parsed = parse_tool_result(result)
        
        return create_response(
            success=parsed["success"],
            data=parsed["data"],
            error=parsed["error"],
            message=f"InterPro query completed for {request.uniprot_id}"
        )
    
    except Exception as e:
        logger.error(f"InterPro query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/pdb/download", response_model=StandardResponse)
async def download_pdb_structure(request: PDBDownloadRequest):
    """
    Download protein structure from PDB database
    
    Args:
        request: PDBDownloadRequest with pdb_id and format
        
    Returns:
        Path to downloaded PDB file and sequence information
    """
    try:
        logger.info(f"PDB download: {request.pdb_id}")
        
        # Call tool
        result = pdb_structure_download_tool.invoke({
            "pdb_id": request.pdb_id,
            "output_format": request.output_format
        })
        
        # Parse result
        parsed = parse_tool_result(result)
        
        return create_response(
            success=parsed["success"],
            data=parsed["data"],
            error=parsed["error"],
            message=f"PDB structure downloaded: {request.pdb_id}"
        )
    
    except Exception as e:
        logger.error(f"PDB download error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ncbi/download", response_model=StandardResponse)
async def download_ncbi_sequence(request: NCBIDownloadRequest):
    """
    Download sequence from NCBI database
    
    Args:
        request: NCBIDownloadRequest with accession_id
        
    Returns:
        Path to downloaded sequence file
    """
    try:
        logger.info(f"NCBI download: {request.accession_id}")
        
        # Call tool
        result = ncbi_sequence_download_tool.invoke({
            "accession_id": request.accession_id,
            "output_format": request.output_format
        })
        
        # Parse result
        parsed = parse_tool_result(result)
        
        return create_response(
            success=parsed["success"],
            data=parsed["data"],
            error=parsed["error"],
            message=f"NCBI sequence downloaded: {request.accession_id}"
        )
    
    except Exception as e:
        logger.error(f"NCBI download error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/alphafold/download", response_model=StandardResponse)
async def download_alphafold_structure(request: AlphaFoldDownloadRequest):
    """
    Download protein structure from AlphaFold database
    
    Args:
        request: AlphaFoldDownloadRequest with uniprot_id
        
    Returns:
        Path to downloaded structure file with confidence information
    """
    try:
        logger.info(f"AlphaFold download: {request.uniprot_id}")
        
        # Call tool
        result = alphafold_structure_download_tool.invoke({
            "uniprot_id": request.uniprot_id,
            "output_format": request.output_format
        })
        
        # Parse result
        parsed = parse_tool_result(result)
        
        return create_response(
            success=parsed["success"],
            data=parsed["data"],
            error=parsed["error"],
            message=f"AlphaFold structure downloaded: {request.uniprot_id}"
        )
    
    except Exception as e:
        logger.error(f"AlphaFold download error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/sequence/extract-from-pdb", response_model=StandardResponse)
async def extract_pdb_sequence(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    """
    Extract protein sequence from uploaded PDB file
    
    Args:
        file: Uploaded PDB file
        
    Returns:
        Extracted sequences for each chain
    """
    temp_file = None
    try:
        logger.info(f"PDB sequence extraction: {file.filename}")
        
        # Validate file
        validate_file(file, ALLOWED_EXTENSIONS["pdb"])
        
        # Save uploaded file
        temp_file = save_upload_file(file, subdir="pdb")
        
        # Call tool
        result = PDB_sequence_extraction_tool.invoke({"pdb_file": temp_file})
        
        # Parse result
        parsed = parse_tool_result(result)
        
        # Schedule cleanup
        if background_tasks:
            background_tasks.add_task(cleanup_file, temp_file)
        
        return create_response(
            success=parsed["success"],
            data=parsed["data"],
            error=parsed["error"],
            message="PDB sequence extraction completed"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"PDB extraction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Immediate cleanup if no background tasks
        if not background_tasks and temp_file:
            cleanup_file(temp_file)


@app.post("/api/prediction/zero-shot-sequence", response_model=StandardResponse)
async def predict_zero_shot_sequence(
    model_name: str = Form(default="ESM2-650M"),
    sequence: Optional[str] = Form(default=None),
    file: Optional[UploadFile] = File(default=None),
    background_tasks: BackgroundTasks = None
):
    """
    Predict beneficial mutations using sequence-based zero-shot models
    
    Args:
        model_name: Model name (ESM-1v, ESM2-650M, ESM-1b, VenusPLM)
        sequence: Protein sequence (if not uploading file)
        file: FASTA file (if not providing sequence)
        
    Returns:
        Mutation predictions with scores
    """
    temp_file = None
    try:
        logger.info(f"Zero-shot sequence prediction with {model_name}")
        
        # Validate input
        if not sequence and not file:
            raise HTTPException(
                status_code=400,
                detail="Either sequence or file must be provided"
            )
        
        # Handle file upload
        if file:
            validate_file(file, ALLOWED_EXTENSIONS["fasta"])
            temp_file = save_upload_file(file, subdir="fasta")
            
            # Call tool with file
            result = zero_shot_sequence_prediction_tool.invoke({
                "fasta_file": temp_file,
                "model_name": model_name
            })
        else:
            # Call tool with sequence
            result = zero_shot_sequence_prediction_tool.invoke({
                "sequence": sequence,
                "model_name": model_name
            })
        
        # Parse result
        parsed = parse_tool_result(result)
        
        # Schedule cleanup
        if background_tasks and temp_file:
            background_tasks.add_task(cleanup_file, temp_file)
        
        return create_response(
            success=parsed["success"],
            data=parsed["data"],
            error=parsed["error"],
            message="Zero-shot sequence prediction completed"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Zero-shot sequence prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if not background_tasks and temp_file:
            cleanup_file(temp_file)


@app.post("/api/prediction/zero-shot-structure", response_model=StandardResponse)
async def predict_zero_shot_structure(
    file: UploadFile = File(...),
    model_name: str = Form(default="ESM-IF1"),
    background_tasks: BackgroundTasks = None
):
    """
    Predict beneficial mutations using structure-based zero-shot models
    
    Args:
        file: PDB structure file
        model_name: Model name (SaProt, ProtSSN, ESM-IF1, MIF-ST, ProSST-2048)
        
    Returns:
        Mutation predictions with scores
    """
    temp_file = None
    try:
        logger.info(f"Zero-shot structure prediction with {model_name}")
        
        # Validate file
        validate_file(file, ALLOWED_EXTENSIONS["structure"])
        
        # Save uploaded file
        temp_file = save_upload_file(file, subdir="structures")
        
        # Call tool
        result = zero_shot_structure_prediction_tool.invoke({
            "structure_file": temp_file,
            "model_name": model_name
        })
        
        # Parse result
        parsed = parse_tool_result(result)
        
        # Schedule cleanup
        if background_tasks:
            background_tasks.add_task(cleanup_file, temp_file)
        
        return create_response(
            success=parsed["success"],
            data=parsed["data"],
            error=parsed["error"],
            message="Zero-shot structure prediction completed"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Zero-shot structure prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if not background_tasks and temp_file:
            cleanup_file(temp_file)


@app.post("/api/prediction/function", response_model=StandardResponse)
async def predict_protein_function(
    model_name: str = Form(default="ESM2-650M"),
    task: str = Form(default="Solubility"),
    sequence: Optional[str] = Form(default=None),
    file: Optional[UploadFile] = File(default=None),
    background_tasks: BackgroundTasks = None
):
    """
    Predict protein functions (solubility, localization, etc.)
    
    Args:
        model_name: Model name
        task: Prediction task (Solubility, Subcellular Localization, etc.)
        sequence: Protein sequence (if not uploading file)
        file: FASTA file (if not providing sequence)
        
    Returns:
        Function prediction results
    """
    temp_file = None
    try:
        logger.info(f"Function prediction: {task} with {model_name}")
        
        # Validate input
        if not sequence and not file:
            raise HTTPException(
                status_code=400,
                detail="Either sequence or file must be provided"
            )
        
        # Handle file upload
        if file:
            validate_file(file, ALLOWED_EXTENSIONS["fasta"])
            temp_file = save_upload_file(file, subdir="fasta")
            
            result = protein_function_prediction_tool.invoke({
                "fasta_file": temp_file,
                "model_name": model_name,
                "task": task
            })
        else:
            result = protein_function_prediction_tool.invoke({
                "sequence": sequence,
                "model_name": model_name,
                "task": task
            })
        
        # Parse result
        parsed = parse_tool_result(result)
        
        # Schedule cleanup
        if background_tasks and temp_file:
            background_tasks.add_task(cleanup_file, temp_file)
        
        return create_response(
            success=parsed["success"],
            data=parsed["data"],
            error=parsed["error"],
            message=f"Function prediction completed: {task}"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Function prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if not background_tasks and temp_file:
            cleanup_file(temp_file)


@app.post("/api/prediction/residue-function", response_model=StandardResponse)
async def predict_residue_function(
    model_name: str = Form(default="ESM2-650M"),
    task: str = Form(default="Activity Site"),
    sequence: Optional[str] = Form(default=None),
    file: Optional[UploadFile] = File(default=None),
    background_tasks: BackgroundTasks = None
):
    """
    Predict functional residues (active sites, binding sites, etc.)
    
    Args:
        model_name: Model name
        task: Prediction task (Activity Site, Binding Site, etc.)
        sequence: Protein sequence (if not uploading file)
        file: FASTA file (if not providing sequence)
        
    Returns:
        Functional residue predictions
    """
    temp_file = None
    try:
        logger.info(f"Residue function prediction: {task} with {model_name}")
        
        # Validate input
        if not sequence and not file:
            raise HTTPException(
                status_code=400,
                detail="Either sequence or file must be provided"
            )
        
        # Handle file upload
        if file:
            validate_file(file, ALLOWED_EXTENSIONS["fasta"])
            temp_file = save_upload_file(file, subdir="fasta")
            
            result = functional_residue_prediction_tool.invoke({
                "fasta_file": temp_file,
                "model_name": model_name,
                "task": task
            })
        else:
            result = functional_residue_prediction_tool.invoke({
                "sequence": sequence,
                "model_name": model_name,
                "task": task
            })
        
        # Parse result
        parsed = parse_tool_result(result)
        
        # Schedule cleanup
        if background_tasks and temp_file:
            background_tasks.add_task(cleanup_file, temp_file)
        
        return create_response(
            success=parsed["success"],
            data=parsed["data"],
            error=parsed["error"],
            message=f"Residue function prediction completed: {task}"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Residue function prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if not background_tasks and temp_file:
            cleanup_file(temp_file)


@app.post("/api/prediction/properties", response_model=StandardResponse)
async def predict_protein_properties(
    task_name: str = Form(default="Physical and chemical properties"),
    sequence: Optional[str] = Form(default=None),
    file: Optional[UploadFile] = File(default=None),
    background_tasks: BackgroundTasks = None
):
    """
    Predict protein physical and chemical properties
    
    Args:
        task_name: Task name (Physical and chemical properties, SASA, etc.)
        sequence: Protein sequence (if not uploading file)
        file: FASTA or PDB file (if not providing sequence)
        
    Returns:
        Protein property predictions
    """
    temp_file = None
    try:
        logger.info(f"Protein properties prediction: {task_name}")
        
        # Validate input
        if not sequence and not file:
            raise HTTPException(
                status_code=400,
                detail="Either sequence or file must be provided"
            )
        
        # Handle file upload
        if file:
            validate_file(file, ALLOWED_EXTENSIONS["fasta"] + ALLOWED_EXTENSIONS["pdb"])
            temp_file = save_upload_file(file, subdir="fasta")
            
            result = protein_properties_generation_tool.invoke({
                "fasta_file": temp_file,
                "task_name": task_name
            })
        else:
            result = protein_properties_generation_tool.invoke({
                "sequence": sequence,
                "task_name": task_name
            })
        
        # Parse result
        parsed = parse_tool_result(result)
        
        # Schedule cleanup
        if background_tasks and temp_file:
            background_tasks.add_task(cleanup_file, temp_file)
        
        return create_response(
            success=parsed["success"],
            data=parsed["data"],
            error=parsed["error"],
            message=f"Protein properties prediction completed: {task_name}"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Protein properties prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if not background_tasks and temp_file:
            cleanup_file(temp_file)


@app.post("/api/training/generate-config", response_model=StandardResponse)
async def generate_training_config(
    csv_file: UploadFile = File(...),
    test_csv_file: Optional[UploadFile] = File(default=None),
    output_name: str = Form(default="custom_training_config"),
    background_tasks: BackgroundTasks = None
):
    """
    Generate training configuration from CSV files
    
    Args:
        csv_file: Training CSV file with 'aa_seq' and 'label' columns
        test_csv_file: Optional test CSV file
        output_name: Name for generated config file
        
    Returns:
        Path to generated training configuration JSON file
    """
    temp_files = []
    try:
        logger.info(f"Generating training config: {output_name}")
        
        # Validate and save training CSV
        validate_file(csv_file, ALLOWED_EXTENSIONS["csv"])
        train_csv_path = save_upload_file(csv_file, subdir="training")
        temp_files.append(train_csv_path)
        
        # Handle optional test CSV
        test_csv_path = None
        if test_csv_file:
            validate_file(test_csv_file, ALLOWED_EXTENSIONS["csv"])
            test_csv_path = save_upload_file(test_csv_file, subdir="training")
            temp_files.append(test_csv_path)
            
        # Call tool
        result = generate_training_config_tool.invoke({
            "csv_file": train_csv_path,
            "test_csv_file": test_csv_path,
            "output_name": output_name
        })
        
        # Parse result
        parsed = parse_tool_result(result)
        
        # Schedule cleanup
        if background_tasks:
            background_tasks.add_task(cleanup_files_background, temp_files)
        
        return create_response(
            success=parsed["success"],
            data=parsed["data"],
            error=parsed["error"],
            message="Training configuration generated successfully"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Training config generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if not background_tasks:
            cleanup_files_background(temp_files)


@app.post("/api/code/execute", response_model=StandardResponse)
async def execute_ai_code(request: CodeExecutionRequest):
    """
    Generate and execute Python code based on task description
    
    Args:
        request: CodeExecutionRequest with task description and input files
        
    Returns:
        Execution results and generated output files
    """
    try:
        logger.info(f"AI code execution: {request.task_description[:100]}...")
        
        # Validate input files exist
        for file_path in request.input_files:
            if not os.path.exists(file_path):
                raise HTTPException(
                    status_code=400,
                    detail=f"Input file not found: {file_path}"
                )
        
        # Call tool
        result = ai_code_execution_tool.invoke({
            "task_description": request.task_description,
            "input_files": request.input_files
        })
        
        # Parse result
        parsed = parse_tool_result(result)
        
        return create_response(
            success=parsed["success"],
            data=parsed["data"],
            error=parsed["error"],
            message="Code execution completed"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Code execution error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/literature/search", response_model=StandardResponse)
async def search_literature(request: LiteratureSearchRequest):
    """
    Search for scientific literature
    
    Args:
        request: LiteratureSearchRequest with query and max_results
        
    Returns:
        Search results from arXiv, PubMed, and Google Scholar
    """
    try:
        logger.info(f"Literature search: {request.query}")
        
        # Call tool
        result = literature_search_tool.invoke({
            "query": request.query,
            "max_results": request.max_results
        })
        
        # Parse result
        parsed = parse_tool_result(result)
        
        return create_response(
            success=parsed["success"],
            data=parsed["data"],
            error=parsed["error"],
            message=f"Literature search completed for: {request.query}"
        )

    except Exception as e:
        logger.error(f"Literature search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/download/{file_path:path}")
async def download_file(file_path: str):
    """
    Download generated output files
    
    Args:
        file_path: Relative path to file in temp_outputs directory
        
    Returns:
        File download response
    """
    try:
        # Construct full path
        full_path = OUTPUT_DIR / file_path
        
        # Security check: ensure file is within OUTPUT_DIR
        if not str(full_path.resolve()).startswith(str(OUTPUT_DIR.resolve())):
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Check if file exists
        if not full_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        
        return FileResponse(
            path=str(full_path),
            filename=full_path.name,
            media_type='application/octet-stream'
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"File download error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions with standard response format"""
    return JSONResponse(
        status_code=exc.status_code,
        content=create_response(
            success=False,
            error=exc.detail
        )
    )

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Handle all other exceptions"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=create_response(
            success=False,
            error=f"Internal server error: {type(exc).__name__}"
        )
    )


@app.on_event("startup")
async def startup_event():
    """Initialize service on startup"""
    logger.info("VenusFactory API starting up...")
    logger.info(f"Upload directory: {UPLOAD_DIR}")
    logger.info(f"Output directory: {OUTPUT_DIR}")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("VenusFactory API shutting down...")


if __name__ == "__main__":
    uvicorn.run(
        "fast_api:app",
        host="0.0.0.0",
        port=5000,
        reload=True,  # Enable auto-reload during development
        log_level="info"
    )