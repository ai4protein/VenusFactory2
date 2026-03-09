# predict: LangChain @tool layer; calls features_operations, finetuned_operations, strcutrue_operations; returns status JSON.

import json
import os
from typing import Optional

from langchain.tools import tool
from pydantic import BaseModel, Field

# Operations (features, finetuned, structure)
from .features.features_operations import (
    calculate_physchem_from_fasta,
    calculate_rsa_from_pdb,
    calculate_sasa_from_pdb,
    calculate_ss_from_pdb,
)
from .finetuned.fintuned_operations import (
    predict_protein_function as do_predict_protein_function,
    predict_residue_function as do_predict_residue_function,
)
from .structure.strcutrue_operations import predict_structure_esmfold as do_predict_structure_esmfold


# ---------- Features (features_operations) ----------
class CalculatePhyschemInput(BaseModel):
    fasta_file: str = Field(..., description="Path to FASTA file.")
    output_file: Optional[str] = Field(default=None, description="Output JSON path. If omitted, written to out_dir.")
    out_dir: Optional[str] = Field(default=None, description="Output directory.")


class CalculateRsaInput(BaseModel):
    pdb_file: str = Field(..., description="Path to PDB file.")
    chain_id: str = Field(default="A", description="Chain ID. Default A.")
    output_file: Optional[str] = Field(default=None, description="Output JSON path.")
    out_dir: Optional[str] = Field(default=None, description="Output directory.")


class CalculateSasaInput(BaseModel):
    pdb_file: str = Field(..., description="Path to PDB file.")
    output_file: Optional[str] = Field(default=None, description="Output JSON path.")
    out_dir: Optional[str] = Field(default=None, description="Output directory.")


class CalculateSsInput(BaseModel):
    pdb_file: str = Field(..., description="Path to PDB file.")
    chain_id: str = Field(default="A", description="Chain ID. Default A.")
    output_file: Optional[str] = Field(default=None, description="Output JSON path.")
    out_dir: Optional[str] = Field(default=None, description="Output directory.")


class CalculateAllPropertiesInput(BaseModel):
    input_file: str = Field(..., description="Path to FASTA or PDB file.")
    file_type: str = Field(default="auto", description="fasta, pdb, or auto.")
    chain_id: str = Field(default="A", description="Chain ID for PDB. Default A.")
    output_file: Optional[str] = Field(default=None, description="Output JSON path.")
    out_dir: Optional[str] = Field(default=None, description="Output directory.")


@tool("calculate_physchem_from_fasta", args_schema=CalculatePhyschemInput)
def calculate_physchem_from_fasta_tool(
    fasta_file: str,
    output_file: Optional[str] = None,
    out_dir: Optional[str] = None,
) -> str:
    """Calculate physicochemical properties from FASTA. Returns status JSON with file_info."""
    try:
        if not os.path.exists(fasta_file):
            return json.dumps({"status": "error", "error": {"type": "ValidationError", "message": f"File not found: {fasta_file}"}, "file_info": None})
        return calculate_physchem_from_fasta(fasta_file, output_file=output_file, out_dir=out_dir)
    except Exception as e:
        return json.dumps({"status": "error", "error": {"type": "ToolError", "message": str(e)}, "file_info": None}, ensure_ascii=False)


@tool("calculate_rsa_from_pdb", args_schema=CalculateRsaInput)
def calculate_rsa_from_pdb_tool(
    pdb_file: str,
    chain_id: str = "A",
    output_file: Optional[str] = None,
    out_dir: Optional[str] = None,
) -> str:
    """Calculate RSA per residue from PDB. Returns status JSON with file_info."""
    try:
        if not os.path.exists(pdb_file):
            return json.dumps({"status": "error", "error": {"type": "ValidationError", "message": f"File not found: {pdb_file}"}, "file_info": None})
        return calculate_rsa_from_pdb(pdb_file, chain_id=chain_id, output_file=output_file, out_dir=out_dir)
    except Exception as e:
        return json.dumps({"status": "error", "error": {"type": "ToolError", "message": str(e)}, "file_info": None}, ensure_ascii=False)


@tool("calculate_sasa_from_pdb", args_schema=CalculateSasaInput)
def calculate_sasa_from_pdb_tool(
    pdb_file: str,
    output_file: Optional[str] = None,
    out_dir: Optional[str] = None,
) -> str:
    """Calculate SASA per residue from PDB. Returns status JSON with file_info."""
    try:
        if not os.path.exists(pdb_file):
            return json.dumps({"status": "error", "error": {"type": "ValidationError", "message": f"File not found: {pdb_file}"}, "file_info": None})
        return calculate_sasa_from_pdb(pdb_file, output_file=output_file, out_dir=out_dir)
    except Exception as e:
        return json.dumps({"status": "error", "error": {"type": "ToolError", "message": str(e)}, "file_info": None}, ensure_ascii=False)


@tool("calculate_ss_from_pdb", args_schema=CalculateSsInput)
def calculate_ss_from_pdb_tool(
    pdb_file: str,
    chain_id: str = "A",
    output_file: Optional[str] = None,
    out_dir: Optional[str] = None,
) -> str:
    """Calculate secondary structure per residue from PDB. Returns status JSON with file_info."""
    try:
        if not os.path.exists(pdb_file):
            return json.dumps({"status": "error", "error": {"type": "ValidationError", "message": f"File not found: {pdb_file}"}, "file_info": None})
        return calculate_ss_from_pdb(pdb_file, chain_id=chain_id, output_file=output_file, out_dir=out_dir)
    except Exception as e:
        return json.dumps({"status": "error", "error": {"type": "ToolError", "message": str(e)}, "file_info": None}, ensure_ascii=False)




# ---------- Finetuned (fintuned_operations) ----------
class PredictProteinFunctionInput(BaseModel):
    fasta_file: str = Field(..., description="Path to FASTA file.")
    task: str = Field(..., description="Task name (e.g. Solubility, Optimal Temperature). From constant.json dataset_mapping_function.")
    model_name: str = Field(default="Ankh-large", description="Model: Ankh-large, ESM2-650M, ProtBert, ProtT5-xl-uniref50.")
    adapter_path: Optional[str] = Field(default=None, description="Adapter dir. If omitted, ckpt/{dataset}/{model}.")
    ckpt_base: str = Field(default="ckpt", description="Base dir for adapter. Default ckpt.")
    output_file: Optional[str] = Field(default=None, description="Output CSV path.")
    out_dir: Optional[str] = Field(default=None, description="Output directory.")


class PredictResidueFunctionInput(BaseModel):
    fasta_file: str = Field(..., description="Path to FASTA file.")
    task: str = Field(..., description="Residue task: Activity Site, Binding Site, Conserved Site, Motif.")
    model_name: str = Field(default="ESM2-650M", description="Model: ESM2-650M, Ankh-large, ProtT5-xl-uniref50.")
    adapter_path: Optional[str] = Field(default=None, description="Adapter dir. If omitted, ckpt/{residue_dataset}/{model}.")
    ckpt_base: str = Field(default="ckpt", description="Base dir. Default ckpt.")
    output_file: Optional[str] = Field(default=None, description="Output CSV path.")
    out_dir: Optional[str] = Field(default=None, description="Output directory.")


@tool("predict_protein_function", args_schema=PredictProteinFunctionInput)
def predict_protein_function_tool(
    fasta_file: str,
    task: str,
    model_name: str = "Ankh-large",
    adapter_path: Optional[str] = None,
    ckpt_base: str = "ckpt",
    output_file: Optional[str] = None,
    out_dir: Optional[str] = None,
) -> str:
    """Run finetuned protein function prediction (task = e.g. Solubility, Optimal Temperature). Returns status JSON with file_info."""
    try:
        if not os.path.exists(fasta_file):
            return json.dumps({"status": "error", "error": {"type": "ValidationError", "message": f"File not found: {fasta_file}"}, "file_info": None})
        return do_predict_protein_function(
            fasta_file=fasta_file,
            task=task,
            model_name=model_name,
            adapter_path=adapter_path,
            ckpt_base=ckpt_base,
            output_file=output_file,
            out_dir=out_dir,
        )
    except Exception as e:
        return json.dumps({"status": "error", "error": {"type": "ToolError", "message": str(e)}, "file_info": None}, ensure_ascii=False)


@tool("predict_residue_function", args_schema=PredictResidueFunctionInput)
def predict_residue_function_tool(
    fasta_file: str,
    task: str,
    model_name: str = "ESM2-650M",
    adapter_path: Optional[str] = None,
    ckpt_base: str = "ckpt",
    output_file: Optional[str] = None,
    out_dir: Optional[str] = None,
) -> str:
    """Run finetuned residue prediction (Activity Site, Binding Site, Conserved Site, Motif). Returns status JSON with file_info."""
    try:
        if not os.path.exists(fasta_file):
            return json.dumps({"status": "error", "error": {"type": "ValidationError", "message": f"File not found: {fasta_file}"}, "file_info": None})
        return do_predict_residue_function(
            fasta_file=fasta_file,
            task=task,
            model_name=model_name,
            adapter_path=adapter_path,
            ckpt_base=ckpt_base,
            output_file=output_file,
            out_dir=out_dir,
        )
    except Exception as e:
        return json.dumps({"status": "error", "error": {"type": "ToolError", "message": str(e)}, "file_info": None}, ensure_ascii=False)


# ---------- Structure (strcutrue_operations) ----------
class PredictStructureEsmfoldInput(BaseModel):
    sequence: str = Field(..., description="Protein sequence (one-letter). ESMFold local prediction.")
    output_dir: Optional[str] = Field(default=None, description="Output directory for PDB.")
    output_file: Optional[str] = Field(default=None, description="Output PDB filename.")
    verbose: bool = Field(default=True, description="Verbose output.")


@tool("predict_structure_esmfold", args_schema=PredictStructureEsmfoldInput)
def predict_structure_esmfold_tool(
    sequence: str,
    output_dir: Optional[str] = None,
    output_file: Optional[str] = None,
    verbose: bool = True,
) -> str:
    """Predict protein structure with ESMFold (local). Returns status JSON with file_info (PDB path)."""
    try:
        if not sequence or not sequence.strip():
            return json.dumps({"status": "error", "error": {"type": "ValidationError", "message": "Sequence is required."}, "file_info": None})
        return do_predict_structure_esmfold(
            sequence=sequence.strip(),
            output_dir=output_dir,
            output_file=output_file,
            verbose=verbose,
        )
    except Exception as e:
        return json.dumps({"status": "error", "error": {"type": "ToolError", "message": str(e)}, "file_info": None}, ensure_ascii=False)


PREDICT_TOOLS = [
    calculate_physchem_from_fasta_tool,
    calculate_rsa_from_pdb_tool,
    calculate_sasa_from_pdb_tool,
    calculate_ss_from_pdb_tool,
    predict_protein_function_tool,
    predict_residue_function_tool,
    predict_structure_esmfold_tool,
]
