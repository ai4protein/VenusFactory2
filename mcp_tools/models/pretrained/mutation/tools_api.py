
from mcp_tools.tools.capability_policy import guard_operation
# mutation API outer layer: FastAPI routes; call core (mutation_operations), return status JSON.

from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from mcp_tools.models.pretrained.mutation.models.mutation_operations import zero_shot_mutation_sequence_prediction, zero_shot_mutation_structure_prediction, DEFAULT_BACKEND


router = APIRouter(prefix="/api/v1/mutation", tags=["mutation"])


class ZeroShotSequenceBody(BaseModel):
    sequence: Optional[str] = Field(default=None, description="Protein sequence (one-letter). One of sequence or fasta_file required.")
    fasta_file: Optional[str] = Field(default=None, description="Path to FASTA file. One of sequence or fasta_file required.")
    model_name: str = Field(default="ESM2-650M", description="Model name. Default ESM2-650M.")
    backend: str = Field(default=DEFAULT_BACKEND, description="Backend: local or pjlab.")
    api_key: Optional[str] = Field(default=None, description="Optional API key.")


class ZeroShotStructureBody(BaseModel):
    structure_file: str = Field(..., description="Path to PDB structure file.")
    model_name: str = Field(default="ESM-IF1", description="Model name. Default ESM-IF1.")
    backend: str = Field(default=DEFAULT_BACKEND, description="Backend: local or pjlab.")
    api_key: Optional[str] = Field(default=None, description="Optional API key.")


@router.post("/zero-shot/sequence")
@guard_operation('zero_shot_mutation_sequence_prediction')
def api_zero_shot_mutation_sequence(body: ZeroShotSequenceBody):
    """Zero-shot mutation prediction from sequence. Returns status JSON."""
    if not body.sequence and not body.fasta_file:
        raise HTTPException(status_code=400, detail="Either sequence or fasta_file must be provided.")
    result = zero_shot_mutation_sequence_prediction(
        sequence=body.sequence,
        fasta_file=body.fasta_file,
        model_name=body.model_name,
        api_key=body.api_key,
        backend=body.backend,
    )
    return result


@router.post("/zero-shot/structure")
@guard_operation('zero_shot_mutation_structure_prediction')
def api_zero_shot_mutation_structure(body: ZeroShotStructureBody):
    """Zero-shot mutation prediction from PDB structure. Returns status JSON."""
    result = zero_shot_mutation_structure_prediction(
        structure_file=body.structure_file,
        model_name=body.model_name,
        api_key=body.api_key,
        backend=body.backend,
    )
    return result
