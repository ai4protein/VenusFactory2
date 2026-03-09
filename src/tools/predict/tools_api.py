"""
Predict API: FastAPI routes calling features_operations, finetuned_operations, strcutrue_operations.
Returns core status JSON (with file_info, etc.) directly.
"""
import json
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from .features.features_operations import (
    calculate_physchem_from_fasta,
    calculate_rsa_from_pdb,
    calculate_sasa_from_pdb,
    calculate_ss_from_pdb,
    calculate_all_properties,
)
from .finetuned.fintuned_operations import (
    predict_protein_function,
    predict_residue_function,
)
from .structure.strcutrue_operations import predict_structure_esmfold


router = APIRouter(prefix="/api/v1/predict", tags=["predict"])


def _ensure_dict(result):  # core may return JSON str
    if isinstance(result, str):
        try:
            return json.loads(result)
        except json.JSONDecodeError:
            return {"status": "error", "error": {"type": "ParseError", "message": result}, "file_info": None}
    return result


# ---------- Request bodies ----------
class CalculatePhyschemBody(BaseModel):
    fasta_file: str = Field(..., description="Path to FASTA file.")
    output_file: Optional[str] = Field(default=None, description="Output JSON path.")
    out_dir: Optional[str] = Field(default=None, description="Output directory.")


class CalculateRsaBody(BaseModel):
    pdb_file: str = Field(..., description="Path to PDB file.")
    chain_id: str = Field(default="A", description="Chain ID.")
    output_file: Optional[str] = Field(default=None, description="Output JSON path.")
    out_dir: Optional[str] = Field(default=None, description="Output directory.")


class CalculateSasaBody(BaseModel):
    pdb_file: str = Field(..., description="Path to PDB file.")
    output_file: Optional[str] = Field(default=None, description="Output JSON path.")
    out_dir: Optional[str] = Field(default=None, description="Output directory.")


class CalculateSsBody(BaseModel):
    pdb_file: str = Field(..., description="Path to PDB file.")
    chain_id: str = Field(default="A", description="Chain ID.")
    output_file: Optional[str] = Field(default=None, description="Output JSON path.")
    out_dir: Optional[str] = Field(default=None, description="Output directory.")


class CalculateAllPropertiesBody(BaseModel):
    input_file: str = Field(..., description="Path to FASTA or PDB file.")
    file_type: str = Field(default="auto", description="fasta, pdb, or auto.")
    chain_id: str = Field(default="A", description="Chain ID for PDB.")
    output_file: Optional[str] = Field(default=None, description="Output path.")
    out_dir: Optional[str] = Field(default=None, description="Output directory.")


class PredictProteinFunctionBody(BaseModel):
    fasta_file: str = Field(..., description="Path to FASTA file.")
    task: str = Field(..., description="Task name (e.g. Solubility, Optimal Temperature).")
    model_name: str = Field(default="Ankh-large", description="Model name.")
    adapter_path: Optional[str] = Field(default=None, description="Override adapter path.")
    ckpt_base: str = Field(default="ckpt", description="Checkpoint base directory.")
    output_file: Optional[str] = Field(default=None, description="Output path.")
    out_dir: Optional[str] = Field(default=None, description="Output directory.")


class PredictResidueFunctionBody(BaseModel):
    fasta_file: str = Field(..., description="Path to FASTA file.")
    task: str = Field(..., description="Residue task (Activity Site, Binding Site, etc.).")
    model_name: str = Field(default="ESM2-650M", description="Model name.")
    adapter_path: Optional[str] = Field(default=None, description="Override adapter path.")
    ckpt_base: str = Field(default="ckpt", description="Checkpoint base directory.")
    output_file: Optional[str] = Field(default=None, description="Output path.")
    out_dir: Optional[str] = Field(default=None, description="Output directory.")


class PredictStructureEsmfoldBody(BaseModel):
    sequence: str = Field(..., description="Protein sequence (one-letter).")
    output_dir: Optional[str] = Field(default=None, description="Output directory for PDB.")
    output_file: Optional[str] = Field(default=None, description="Output PDB filename.")
    verbose: bool = Field(default=True, description="Verbose output.")


# ---------- Features ----------
@router.post("/features/physchem")
def api_calculate_physchem_from_fasta(body: CalculatePhyschemBody):
    """Calculate physicochemical properties from FASTA. Returns status JSON with file_info."""
    try:
        result = calculate_physchem_from_fasta(
            body.fasta_file,
            output_file=body.output_file,
            out_dir=body.out_dir,
        )
        return _ensure_dict(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/features/rsa")
def api_calculate_rsa_from_pdb(body: CalculateRsaBody):
    """Calculate RSA per residue from PDB. Returns status JSON with file_info."""
    try:
        result = calculate_rsa_from_pdb(
            body.pdb_file,
            chain_id=body.chain_id,
            output_file=body.output_file,
            out_dir=body.out_dir,
        )
        return _ensure_dict(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/features/sasa")
def api_calculate_sasa_from_pdb(body: CalculateSasaBody):
    """Calculate SASA per residue from PDB. Returns status JSON with file_info."""
    try:
        result = calculate_sasa_from_pdb(
            body.pdb_file,
            output_file=body.output_file,
            out_dir=body.out_dir,
        )
        return _ensure_dict(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/features/secondary-structure")
def api_calculate_ss_from_pdb(body: CalculateSsBody):
    """Calculate secondary structure per residue from PDB. Returns status JSON with file_info."""
    try:
        result = calculate_ss_from_pdb(
            body.pdb_file,
            chain_id=body.chain_id,
            output_file=body.output_file,
            out_dir=body.out_dir,
        )
        return _ensure_dict(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/features/all-properties")
def api_calculate_all_properties(body: CalculateAllPropertiesBody):
    """Calculate all properties (physchem, RSA, SASA, SS) from FASTA or PDB. Returns status JSON."""
    try:
        result = calculate_all_properties(
            body.input_file,
            file_type=body.file_type,
            chain_id=body.chain_id,
            output_file=body.output_file,
            out_dir=body.out_dir,
        )
        return _ensure_dict(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------- Finetuned ----------
@router.post("/finetuned/protein-function")
def api_predict_protein_function(body: PredictProteinFunctionBody):
    """Run finetuned protein function prediction (e.g. Solubility, Optimal Temperature). Returns status JSON."""
    try:
        result = predict_protein_function(
            fasta_file=body.fasta_file,
            task=body.task,
            model_name=body.model_name,
            adapter_path=body.adapter_path,
            ckpt_base=body.ckpt_base,
            output_file=body.output_file,
            out_dir=body.out_dir,
        )
        return _ensure_dict(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/finetuned/residue-function")
def api_predict_residue_function(body: PredictResidueFunctionBody):
    """Run finetuned residue prediction (Activity Site, Binding Site, etc.). Returns status JSON."""
    try:
        result = predict_residue_function(
            fasta_file=body.fasta_file,
            task=body.task,
            model_name=body.model_name,
            adapter_path=body.adapter_path,
            ckpt_base=body.ckpt_base,
            output_file=body.output_file,
            out_dir=body.out_dir,
        )
        return _ensure_dict(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------- Structure ----------
@router.post("/structure/esmfold")
def api_predict_structure_esmfold(body: PredictStructureEsmfoldBody):
    """Predict protein structure with ESMFold (local). Returns status JSON with file_info (PDB path)."""
    try:
        result = predict_structure_esmfold(
            sequence=body.sequence,
            output_dir=body.output_dir,
            output_file=body.output_file,
            verbose=body.verbose,
        )
        return _ensure_dict(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
