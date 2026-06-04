"""
tools/denovo/tools_agent.py — ProteinMPNN LangChain @tool 定义

两个工具：
    proteinmpnn_design_tool  – 序列设计（单链/多链/界面/同源多聚体/部分固定）
    proteinmpnn_score_tool   – 序列打分
"""
import json
from typing import Dict, List, Optional

from langchain.tools import tool
from pydantic import BaseModel, Field

from .tools_mcp import call_proteinmpnn_design, call_proteinmpnn_score


class ProteinMPNNSequenceDesignFromStructureInput(BaseModel):
    pdb_path: str = Field(..., description="Path to input PDB file")
    designed_chains: Optional[List[str]] = Field(
        None,
        description="Chain IDs to redesign, e.g. ['A']. None = all chains (single-chain design)"
    )
    fixed_chains: Optional[List[str]] = Field(
        None,
        description="Chain IDs to fix as structural context (interface design). Used together with designed_chains."
    )
    fixed_residues_json: Optional[str] = Field(
        None,
        description='JSON string: chain → 1-indexed residue positions to keep native, e.g. \'{"A": [1, 5, 10]}\''
    )
    homomer: bool = Field(
        False,
        description="True for homomeric symmetry: all designed_chains share the same sequence"
    )
    num_sequences: int = Field(8, description="Number of sequences to generate")
    temperatures: Optional[List[float]] = Field(None, description="Sampling temperatures, e.g. [0.1, 0.2]")
    omit_aas: str = Field("X", description="Amino acids to exclude from design")
    model_name: str = Field("v_48_020", description="Model weights: v_48_002/010/020/030")
    backbone_noise: float = Field(0.0, description="Backbone coord Gaussian noise std")
    ca_only: bool = Field(False, description="Use CA-only model and coordinates")
    out_dir: Optional[str] = Field(
        None,
        description="Output directory for designed sequences (FASTA). When omitted, falls back to the global temp_outputs ProteinMPNN/Design path (backward-compat).",
    )


class ProteinMPNNSequenceScoringFromStructureInput(BaseModel):
    pdb_path: str = Field(..., description="Backbone PDB file path")
    fasta_path: Optional[str] = Field(None, description="FASTA to score; None = score native PDB sequence")
    designed_chains: Optional[List[str]] = Field(None, description="Chains to evaluate; None = all chains")
    num_batches: int = Field(1, description="Stochastic forward passes to average over")
    model_name: str = Field("v_48_020", description="Model weights name")
    backbone_noise: float = Field(0.0, description="Backbone coord Gaussian noise std")
    out_dir: Optional[str] = Field(
        None,
        description="Output directory for scored sequences (FASTA). When omitted, falls back to the global temp_outputs ProteinMPNN/Score path (backward-compat).",
    )


@tool("proteinmpnn_sequence_design_from_structure", args_schema=ProteinMPNNSequenceDesignFromStructureInput)
def proteinmpnn_sequence_design_from_structure_tool(
    pdb_path: str,
    designed_chains: Optional[List[str]] = None,
    fixed_chains: Optional[List[str]] = None,
    fixed_residues_json: Optional[str] = None,
    homomer: bool = False,
    num_sequences: int = 8,
    temperatures: Optional[List[float]] = None,
    omit_aas: str = "X",
    model_name: str = "v_48_020",
    backbone_noise: float = 0.0,
    ca_only: bool = False,
    out_dir: Optional[str] = None,
) -> str:
    """
    Design protein sequences using ProteinMPNN given a PDB backbone structure.
    Covers multiple scenarios via optional parameters:
    - Single-chain design: provide only pdb_path
    - Multi-chain partial design: set designed_chains=['A']
    - Interface / binder design: set designed_chains=['B'], fixed_chains=['A']
    - Homomeric symmetric design: set designed_chains=['A','B','C'], homomer=True
    - Partial fixed design: set fixed_residues_json='{"A": [1, 5, 10]}'
    Returns the path to the output FASTA file.
    """
    fixed_residues = None
    if fixed_residues_json:
        try:
            fixed_residues = json.loads(fixed_residues_json)
        except json.JSONDecodeError as e:
            return json.dumps({"success": False, "error": f"Invalid fixed_residues_json: {e}"})
        # Validate + clamp residue indices to the actual chain length.
        # ProteinMPNN expects 1-based residue numbers; passing an index
        # equal to or beyond the chain length crashes with a numpy
        # ``index N is out of bounds for axis 0 with size N`` deep inside
        # the runner. Catch this at the boundary instead.
        if isinstance(fixed_residues, dict):
            try:
                import os as _os
                from tools.denovo.proteinmpnn.protein_mpnn_utils import parse_PDB as _parse_PDB
                if _os.path.exists(pdb_path):
                    pdb_dict_list = _parse_PDB(pdb_path)
                    chain_lengths: dict[str, int] = {}
                    for k, v in (pdb_dict_list[0].items() if pdb_dict_list else []):
                        if k.startswith("seq_chain_"):
                            chain_lengths[k[-1]] = len(v)
                    cleaned: dict[str, list[int]] = {}
                    warnings: list[str] = []
                    for chain, idxs in fixed_residues.items():
                        if not isinstance(idxs, list):
                            continue
                        n = chain_lengths.get(chain, 0)
                        valid = []
                        for i in idxs:
                            try:
                                ii = int(i)
                            except Exception:
                                continue
                            if n > 0 and (ii < 1 or ii > n):
                                warnings.append(f"chain {chain}: residue index {ii} out of range [1, {n}]; dropped")
                                continue
                            valid.append(ii)
                        if valid:
                            cleaned[chain] = sorted(set(valid))
                    fixed_residues = cleaned or fixed_residues
                    if warnings:
                        # Continue with cleaned set; surface the clamp in tool output
                        print(f"⚠️ proteinmpnn: clamped {len(warnings)} out-of-bound fixed residues: {warnings[:5]}")
            except Exception:
                # If validation itself fails, fall through with the original
                # residues — the runner will still error but with the same
                # message as before.
                pass
    return call_proteinmpnn_design(
        pdb_path=pdb_path,
        designed_chains=designed_chains,
        fixed_chains=fixed_chains,
        fixed_residues=fixed_residues,
        homomer=homomer,
        num_sequences=num_sequences,
        temperatures=temperatures,
        omit_aas=omit_aas,
        model_name=model_name,
        backbone_noise=backbone_noise,
        ca_only=ca_only,
        out_dir=out_dir,
    )


@tool("proteinmpnn_sequence_scoring_from_structure", args_schema=ProteinMPNNSequenceScoringFromStructureInput)
def proteinmpnn_sequence_scoring_from_structure_tool(
    pdb_path: str,
    fasta_path: Optional[str] = None,
    designed_chains: Optional[List[str]] = None,
    num_batches: int = 1,
    model_name: str = "v_48_020",
    backbone_noise: float = 0.0,
    out_dir: Optional[str] = None,
) -> str:
    """
    Score sequences against a backbone structure using ProteinMPNN log-probabilities
    (NLL = -log_prob; lower is better). Outputs a FASTA file with each sequence and its
    score/global_score in the header. If no fasta_path is provided, scores the native
    sequence from the PDB. Returns the path to the output FASTA file.
    """
    return call_proteinmpnn_score(
        pdb_path=pdb_path,
        fasta_path=fasta_path,
        designed_chains=designed_chains,
        num_batches=num_batches,
        model_name=model_name,
        backbone_noise=backbone_noise,
        out_dir=out_dir,
    )
