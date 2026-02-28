# predict: @tool definitions for physchem/structure property prediction; call_* in .tools_mcp

import json
import os
from typing import Optional

from langchain.tools import tool
from pydantic import BaseModel, Field
from web.utils.common_utils import get_save_path
from web.utils.file_handlers import extract_first_chain_from_pdb_file, extract_first_sequence_from_fasta_file
from tools.search.tools_mcp import DEFAULT_BACKEND

from .tools_mcp import (
    call_protein_properties_prediction,
    call_protein_function_prediction,
    call_functional_residue_prediction,
    upload_file_to_oss_sync,
)
from src.tools.predict.structure.esmfold import (
    predict_structure_sync,
    _get_default_backend as get_esmfold_default_backend,
)


class ProteinPropertiesInput(BaseModel):
    sequence: Optional[str] = Field(None, description="Protein sequence in single letter amino acid code")
    fasta_file: Optional[str] = Field(None, description="Path to PDB structure file or fasta file")
    task_name: str = Field(default="Physical and chemical properties", description="Task name: Physical and chemical properties, Relative solvent accessible surface area (PDB only), SASA value (PDB only), Secondary structure (PDB only)")
    backend: str = Field(default="local", description="local: local Gradio; pjlab: pjlab Gradio")


class ProteinFunctionPredictionInput(BaseModel):
    sequence: Optional[str] = Field(None, description="Protein sequence in single letter amino acid code")
    fasta_file: Optional[str] = Field(None, description="Path to FASTA file")
    model_name: str = Field(default="ESM2-650M", description="Model name for function prediction")
    task: str = Field(default="Solubility", description="Task: Solubility, Subcellular Localization, Membrane Protein, Metal ion binding, Stability, Sortingsignal, Optimum temperature, Kcat, Optimal PH, Immunogenicity Prediction - Virus, Immunogenicity Prediction - Bacteria, Immunogenicity Prediction - Tumor")
    backend: str = Field(default=DEFAULT_BACKEND, description="local: local Gradio; pjlab: pjlab Gradio + SCP upload")


class FunctionalResiduePredictionInput(BaseModel):
    sequence: Optional[str] = Field(None, description="Protein sequence in single letter amino acid code")
    fasta_file: Optional[str] = Field(None, description="Path to FASTA file")
    model_name: str = Field(default="ESM2-650M", description="Model name for function prediction")
    task: str = Field(default="Activity Site", description="Task: Activity Site, Binding Site, Conserved Site, Motif")
    backend: str = Field(default=DEFAULT_BACKEND, description="local: local Gradio; pjlab: pjlab Gradio + SCP upload")


class ProteinStructurePredictionInput(BaseModel):
    sequence: str = Field(..., description="Protein sequence in single letter amino acid code")
    save_path: Optional[str] = Field(None, description="Path to save the predicted structure")
    verbose: Optional[bool] = Field(default=True, description="Whether to print detailed information")
    backend: Optional[str] = Field(default=None, description="local: GPU; pjlab: DrugSDA remote. Default from AGENT_ESMFOLD_DEFAULT_BACKEND")


@tool("protein_property_prediction", args_schema=ProteinPropertiesInput)
def protein_property_prediction_tool(sequence: Optional[str] = None, fasta_file: Optional[str] = None, task_name: str = "Physical and chemical properties", api_key: Optional[str] = None, backend: str = None) -> str:
    """Predict the protein physical, chemical, and structure properties."""
    try:
        if fasta_file:
            if not os.path.exists(fasta_file):
                return f"Error: FASTA file not found at path: {fasta_file}"
            if fasta_file.lower().endswith('.pdb'):
                fasta_file = extract_first_chain_from_pdb_file(fasta_file)
            elif fasta_file.lower().endswith(('.fasta', '.fa')):
                fasta_file = extract_first_sequence_from_fasta_file(fasta_file)
            return call_protein_properties_prediction(fasta_file=fasta_file, task_name=task_name, api_key=api_key, backend=backend or DEFAULT_BACKEND)
        elif sequence:
            return call_protein_properties_prediction(sequence=sequence, task_name=task_name, api_key=api_key, backend=backend or DEFAULT_BACKEND)
        else:
            return "Error: Either sequence or fasta_file must be provided"
    except Exception as e:
        return f"Protein properties prediction error: {str(e)}"


@tool("protein_function_prediction", args_schema=ProteinFunctionPredictionInput)
def protein_function_prediction_tool(
    sequence: Optional[str] = None,
    fasta_file: Optional[str] = None,
    model_name: str = "ESM2-650M",
    task: str = "Solubility",
    api_key: Optional[str] = None,
    backend: str = None,
) -> str:
    """Predict protein functions like solubility, localization, metal ion binding, stability, sorting signal, and optimum temperature."""
    try:
        if fasta_file and os.path.exists(fasta_file):
            fasta_file = extract_first_sequence_from_fasta_file(fasta_file)
            return call_protein_function_prediction(
                fasta_file=fasta_file,
                model_name=model_name,
                task=task,
                api_key=api_key,
                backend=backend or DEFAULT_BACKEND,
            )
        elif sequence:
            return call_protein_function_prediction(
                sequence=sequence,
                model_name=model_name,
                task=task,
                api_key=api_key,
                backend=backend or DEFAULT_BACKEND,
            )
        else:
            return "Error: Either sequence or fasta_file must be provided"
    except Exception as e:
        return f"Protein function prediction error: {str(e)}"


@tool("functional_residue_prediction", args_schema=FunctionalResiduePredictionInput)
def functional_residue_prediction_tool(
    sequence: Optional[str] = None,
    fasta_file: Optional[str] = None,
    model_name: str = "ESM2-650M",
    task: str = "Activity Site",
    api_key: Optional[str] = None,
    backend: str = None,
) -> str:
    """Predict functional residues (activity/binding/conserved site, motif)."""
    try:
        if fasta_file and os.path.exists(fasta_file):
            fasta_file = extract_first_sequence_from_fasta_file(fasta_file)
            return call_functional_residue_prediction(
                fasta_file=fasta_file,
                model_name=model_name,
                task=task,
                api_key=api_key,
                backend=backend or DEFAULT_BACKEND,
            )
        elif sequence:
            return call_functional_residue_prediction(
                sequence=sequence,
                model_name=model_name,
                task=task,
                api_key=api_key,
                backend=backend or DEFAULT_BACKEND,
            )
        else:
            return "Error: Either sequence or fasta_file must be provided"
    except Exception as e:
        return f"Functional residue prediction error: {str(e)}"


@tool("esmfold_structure_prediction", args_schema=ProteinStructurePredictionInput)
def esmfold_structure_prediction_tool(sequence: str, save_path: Optional[str] = None, verbose: Optional[bool] = True, backend: Optional[str] = None) -> str:
    """Predict protein structure using ESMFold."""
    try:
        if not save_path:
            output_dir = str(get_save_path("Agent", "ESMFold"))
            output_file = None
        elif save_path.lower().endswith((".pdb", ".cif")):
            output_dir = str(get_save_path("Agent", "ESMFold"))
            output_file = os.path.basename(save_path)
        else:
            output_dir = save_path
            output_file = None

        effective_backend = backend if backend is not None else get_esmfold_default_backend()
        pdb_path, result_info = predict_structure_sync(
            sequence, output_dir=output_dir, verbose=verbose, backend=effective_backend, output_file=output_file
        )
        pdb_path_oss_url = upload_file_to_oss_sync(pdb_path, backend=effective_backend)
        return json.dumps({
            "success": True,
            "pdb_path": pdb_path,
            "pdb_path_oss_url": pdb_path_oss_url,
            "result_info": result_info,
        }, indent=2)
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})
