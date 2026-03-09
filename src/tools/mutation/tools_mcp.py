# mutation MCP outer layer: FastMCP tools; call core (mutation_operations), return status JSON.

import json
from typing import Optional

from fastmcp import FastMCP

from .models.mutation_operations import (
    zero_shot_mutation_sequence_prediction,
    zero_shot_mutation_structure_prediction,
    DEFAULT_BACKEND,
)


mcp = FastMCP("Venus_Mutation_MCP")


@mcp.tool(name="zero_shot_mutation_sequence_prediction")
def mcp_zero_shot_mutation_sequence_prediction(
    sequence: Optional[str] = None,
    fasta_file: Optional[str] = None,
    model_name: str = "ESM2-650M",
    api_key: Optional[str] = None,
    backend: Optional[str] = None,
) -> str:
    """Predict beneficial mutations from protein sequence (zero-shot). Returns status JSON."""
    result = zero_shot_mutation_sequence_prediction(
        sequence=sequence,
        fasta_file=fasta_file,
        model_name=model_name,
        api_key=api_key,
        backend=backend or DEFAULT_BACKEND,
    )
    return json.dumps(result, ensure_ascii=False, indent=2)


@mcp.tool(name="zero_shot_mutation_structure_prediction")
def mcp_zero_shot_mutation_structure_prediction(
    structure_file: str,
    model_name: str = "ESM-IF1",
    api_key: Optional[str] = None,
    backend: Optional[str] = None,
) -> str:
    """Predict beneficial mutations from PDB structure (zero-shot). Returns status JSON."""
    result = zero_shot_mutation_structure_prediction(
        structure_file=structure_file,
        model_name=model_name,
        api_key=api_key,
        backend=backend or DEFAULT_BACKEND,
    )
    return json.dumps(result, ensure_ascii=False, indent=2)
