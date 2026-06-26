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
    out_dir: Optional[str] = None,
) -> str:
    """Predict beneficial mutations from a protein SEQUENCE (zero-shot).
    Returns status JSON with the CSV path and a preview.

    `model_name` must be one of the sequence-based models:
        - "ESM2-650M"           (default; general-purpose pLM)
        - "ESM-1b"              (older ESM)
        - "ESM-1v"              (variant-effect specialist)
        - "SaProt"              (structure-aware)
        - "VenusPLM"            (in-house)
    For structure-based scoring (using a PDB) use
    `zero_shot_mutation_structure_prediction` instead — that's where
    "ProSST-2048", "ProtSSN", "MIF-ST", "ESM-IF1", "VenusREM" live.

    `out_dir` (optional): output directory for the CSV/heatmap. When omitted,
    falls back to the global temp_outputs Zero_shot/HeatMap path.
    """
    result = zero_shot_mutation_sequence_prediction(
        sequence=sequence,
        fasta_file=fasta_file,
        model_name=model_name,
        api_key=api_key,
        backend=backend or DEFAULT_BACKEND,
        out_dir=out_dir,
    )
    return json.dumps(result, ensure_ascii=False, indent=2)


@mcp.tool(name="zero_shot_mutation_structure_prediction")
def mcp_zero_shot_mutation_structure_prediction(
    structure_file: str,
    model_name: str = "ESM-IF1",
    api_key: Optional[str] = None,
    backend: Optional[str] = None,
    out_dir: Optional[str] = None,
) -> str:
    """Predict beneficial mutations from a PDB STRUCTURE (zero-shot).
    Returns status JSON with the CSV path and a preview.

    `model_name` must be one of the structure-based models:
        - "ESM-IF1"             (default; ESM inverse-folding)
        - "ProSST-2048"         (structure-aware sequence pLM — pick this
                                 when the user asks for ProSST)
        - "ProtSSN"             (structure-sequence supervised)
        - "MIF-ST"              (masked inverse folding)
        - "SaProt"              (structure-aware)
        - "VenusREM (foldseek-based)"  (Foldseek-augmented ProSST)
    For sequence-only scoring use `zero_shot_mutation_sequence_prediction`
    instead.

    `out_dir` (optional): output directory for the CSV/heatmap. When omitted,
    falls back to the global temp_outputs Zero_shot/HeatMap path.
    """
    result = zero_shot_mutation_structure_prediction(
        structure_file=structure_file,
        model_name=model_name,
        api_key=api_key,
        backend=backend or DEFAULT_BACKEND,
        out_dir=out_dir,
    )
    return json.dumps(result, ensure_ascii=False, indent=2)
