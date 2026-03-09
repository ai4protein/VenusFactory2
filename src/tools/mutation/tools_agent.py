# mutation: LangChain @tool layer; calls core (mutation_operations), returns status JSON.

import json
import os
from typing import Optional

from langchain.tools import tool
from pydantic import BaseModel, Field, model_validator

from web.utils.file_handlers import extract_first_sequence_from_fasta_file

from .models.mutation_operations import (
    zero_shot_mutation_sequence_prediction,
    zero_shot_mutation_structure_prediction,
    DEFAULT_BACKEND,
)


class ZeroShotSequenceInput(BaseModel):
    sequence: Optional[str] = Field(default=None, description="Protein sequence (one-letter). Required unless fasta_file provided.")
    fasta_file: Optional[str] = Field(default=None, description="Path to FASTA file. First sequence used. Required unless sequence provided.")
    model_name: str = Field(default="ESM2-650M", description="Model: ESM-1v, ESM2-650M, ESM-1b, VenusPLM. Default ESM2-650M.")
    backend: str = Field(default=DEFAULT_BACKEND, description="Backend: local or pjlab.")
    api_key: Optional[str] = Field(default=None, description="Optional API key for prediction service.")

    @model_validator(mode="after")
    def require_sequence_or_fasta(self):
        if not self.sequence and not self.fasta_file:
            raise ValueError("Either sequence or fasta_file must be provided.")
        return self


class ZeroShotStructureInput(BaseModel):
    structure_file: str = Field(..., description="Path to PDB structure file. First chain used.")
    model_name: str = Field(default="ESM-IF1", description="Model: SaProt, ProtSSN, ESM-IF1, MIF-ST, etc. Default ESM-IF1.")
    backend: str = Field(default=DEFAULT_BACKEND, description="Backend: local or pjlab.")
    api_key: Optional[str] = Field(default=None, description="Optional API key for prediction service.")


@tool("zero_shot_mutation_sequence_prediction", args_schema=ZeroShotSequenceInput)
def zero_shot_mutation_sequence_prediction_tool(
    sequence: Optional[str] = None,
    fasta_file: Optional[str] = None,
    model_name: str = "ESM2-650M",
    api_key: Optional[str] = None,
    backend: Optional[str] = None,
) -> str:
    """Predict beneficial mutations from sequence (zero-shot). Returns status JSON with data/file_info or error."""
    try:
        if fasta_file:
            if not os.path.exists(fasta_file):
                return json.dumps({"status": "error", "error": {"type": "ValidationError", "message": f"FASTA file not found: {fasta_file}"}, "file_info": None})
            fasta_path = extract_first_sequence_from_fasta_file(fasta_file)
            result = zero_shot_mutation_sequence_prediction(fasta_file=fasta_path, model_name=model_name, api_key=api_key, backend=backend or DEFAULT_BACKEND)
        elif sequence:
            result = zero_shot_mutation_sequence_prediction(sequence=sequence, model_name=model_name, api_key=api_key, backend=backend or DEFAULT_BACKEND)
        else:
            result = {"status": "error", "error": {"type": "ValidationError", "message": "Either sequence or fasta_file must be provided."}, "file_info": None}
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({"status": "error", "error": {"type": "ToolError", "message": str(e)}, "file_info": None}, ensure_ascii=False)


@tool("zero_shot_mutation_structure_prediction", args_schema=ZeroShotStructureInput)
def zero_shot_mutation_structure_prediction_tool(
    structure_file: str,
    model_name: str = "ESM-IF1",
    api_key: Optional[str] = None,
    backend: Optional[str] = None,
) -> str:
    """Predict beneficial mutations from PDB structure (zero-shot). Returns status JSON with data/file_info or error."""
    try:
        actual_path = structure_file
        if structure_file.strip().startswith("{") and structure_file.strip().endswith("}"):
            try:
                obj = json.loads(structure_file)
                if isinstance(obj, dict) and "file_path" in obj:
                    actual_path = obj["file_path"]
            except (json.JSONDecodeError, KeyError, TypeError):
                pass
        if not os.path.exists(actual_path):
            return json.dumps({"status": "error", "error": {"type": "ValidationError", "message": f"Structure file not found: {actual_path}"}, "file_info": None})
        result = zero_shot_mutation_structure_prediction(actual_path, model_name=model_name, api_key=api_key, backend=backend or DEFAULT_BACKEND)
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({"status": "error", "error": {"type": "ToolError", "message": str(e)}, "file_info": None}, ensure_ascii=False)


MUTATION_TOOLS = [
    zero_shot_mutation_sequence_prediction_tool,
    zero_shot_mutation_structure_prediction_tool,
]