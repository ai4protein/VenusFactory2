# tools/mutation: @tool definitions for zero-shot mutation prediction only; call_* in .tools_mcp

import json
import os
from typing import Optional

from langchain.tools import tool
from pydantic import BaseModel, Field
from web.utils.file_handlers import extract_first_sequence_from_fasta_file

from tools.search.tools_mcp import DEFAULT_BACKEND
from .tools_mcp import (
    call_zero_shot_mutation_sequence_prediction,
    call_zero_shot_mutation_structure_prediction_from_file,
)


class ZeroShotSequenceInput(BaseModel):
    sequence: Optional[str] = Field(None, description="Protein sequence in single letter amino acid code")
    fasta_file: Optional[str] = Field(None, description="Path to FASTA file")
    model_name: str = Field(default="ESM2-650M", description="Model name: ESM-1v, ESM2-650M, ESM-1b, VenusPLM")
    backend: str = Field(default=DEFAULT_BACKEND, description="local: local Gradio; pjlab: pjlab Gradio + SCP upload")


class ZeroShotStructureInput(BaseModel):
    structure_file: str = Field(..., description="Path to PDB structure file")
    model_name: str = Field(default="ESM-IF1", description="Model name: SaProt, ProtSSN, ESM-IF1, MIF-ST, ProSST-2048, VenusREM (foldseek-base)")
    backend: str = Field(default=DEFAULT_BACKEND, description="local: local Gradio; pjlab: pjlab Gradio + SCP upload")


@tool("zero_shot_mutation_sequence_prediction", args_schema=ZeroShotSequenceInput)
def zero_shot_mutation_sequence_prediction_tool(sequence: Optional[str] = None, fasta_file: Optional[str] = None, model_name: str = "ESM2-650M", api_key: Optional[str] = None, backend: str = None) -> str:
    """Predict beneficial mutations using sequence-based zero-shot AI models. Outputs are model-predicted scores (not experimental ΔΔG). Use for mutation prediction with protein sequences."""
    try:
        if fasta_file:
            if not os.path.exists(fasta_file):
                return f"Error: FASTA file not found at path: {fasta_file}"
            fasta_file = extract_first_sequence_from_fasta_file(fasta_file)
            return call_zero_shot_mutation_sequence_prediction(fasta_file=fasta_file, model_name=model_name, api_key=api_key, backend=backend or DEFAULT_BACKEND)
        elif sequence:
            return call_zero_shot_mutation_sequence_prediction(sequence=sequence, model_name=model_name, api_key=api_key, backend=backend or DEFAULT_BACKEND)
        else:
            return "Error: Either sequence or fasta_file must be provided"
    except Exception as e:
        return f"Zero-shot sequence prediction error: {str(e)}"


@tool("zero_shot_mutation_structure_prediction", args_schema=ZeroShotStructureInput)
def zero_shot_mutation_structure_prediction_tool(structure_file: str, model_name: str = "ESM-IF1", api_key: Optional[str] = None, backend: str = None) -> str:
    """Predict beneficial mutations using structure-based zero-shot AI models. Outputs are model-predicted scores (not experimental ΔΔG). Use for mutation prediction with PDB structure files."""
    try:
        actual_file_path = structure_file
        try:
            if structure_file.startswith('{') and structure_file.endswith('}'):
                file_info = json.loads(structure_file)
                if isinstance(file_info, dict) and 'file_path' in file_info:
                    actual_file_path = file_info['file_path']
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
        if not os.path.exists(actual_file_path):
            return f"Error: Structure file not found at path: {actual_file_path}"
        return call_zero_shot_mutation_structure_prediction_from_file(actual_file_path, model_name, api_key, backend=backend or DEFAULT_BACKEND)
    except Exception as e:
        return json.dumps({"success": False, "error": f"Zero-shot structure prediction error: {str(e)}"}, ensure_ascii=False)
