"""
tools/denovo/tools_mcp.py — ProteinMPNN local call layer + FastMCP wrapper

Provides:
  - call_proteinmpnn_design / call_proteinmpnn_score: plain helpers used by
    tools_agent.py (LangChain @tool path, for the legacy graph engine).
  - `mcp = FastMCP("Venus_Denovo_MCP")` exposing the same two operations
    over the MCP protocol so kimi-code (and any other MCP client) can see
    them. Mounted by src/mcp_server.py alongside mutation/predict/...
"""
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

from fastmcp import FastMCP

_HERE = Path(__file__).resolve().parent
_MPNN_DIR = _HERE / "proteinmpnn"
if str(_MPNN_DIR) not in sys.path:
    sys.path.insert(0, str(_MPNN_DIR))

from protein_mpnn_function import proteinmpnn_design, proteinmpnn_score


def _read_fasta_preview(fasta_path: str, max_records: int = 10) -> List[Dict]:
    """Parse up to max_records entries from a FASTA file.

    Returns a list of dicts: [{"header": "...", "sequence": "..."}, ...]
    """
    records = []
    try:
        with open(fasta_path, "r") as f:
            header = None
            seq_lines: List[str] = []
            for line in f:
                line = line.rstrip()
                if line.startswith(">"):
                    if header is not None:
                        records.append({"header": header, "sequence": "".join(seq_lines)})
                        if len(records) >= max_records:
                            break
                    header = line[1:]  # strip leading '>'
                    seq_lines = []
                else:
                    seq_lines.append(line)
            # flush last record
            if header is not None and len(records) < max_records:
                records.append({"header": header, "sequence": "".join(seq_lines)})
    except Exception:
        pass  # preview is best-effort; don't fail the whole call
    return records


def call_proteinmpnn_design(
    pdb_path: str,
    designed_chains: Optional[List[str]] = None,
    fixed_chains: Optional[List[str]] = None,
    fixed_residues: Optional[Dict[str, List[int]]] = None,
    homomer: bool = False,
    num_sequences: int = 8,
    temperatures: Optional[List[float]] = None,
    omit_aas: str = "X",
    model_name: str = "v_48_020",
    backbone_noise: float = 0.0,
    ca_only: bool = False,
    out_dir: Optional[str] = None,
) -> str:
    try:
        fasta_path = proteinmpnn_design(
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
        preview = _read_fasta_preview(fasta_path, max_records=10)
        return json.dumps({
            "success": True,
            "fasta_path": fasta_path,
            "total_sequences_preview": len(preview),
            "sequences_preview": preview,
        })
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})


def call_proteinmpnn_score(
    pdb_path: str,
    fasta_path: Optional[str] = None,
    designed_chains: Optional[List[str]] = None,
    num_batches: int = 1,
    model_name: str = "v_48_020",
    backbone_noise: float = 0.0,
    out_dir: Optional[str] = None,
) -> str:
    try:
        out_fasta = proteinmpnn_score(
            pdb_path=pdb_path,
            fasta_path=fasta_path,
            designed_chains=designed_chains,
            num_batches=num_batches,
            model_name=model_name,
            backbone_noise=backbone_noise,
            out_dir=out_dir,
        )
        preview = _read_fasta_preview(out_fasta, max_records=10)
        return json.dumps({
            "success": True,
            "fasta_path": out_fasta,
            "total_sequences_preview": len(preview),
            "sequences_preview": preview,
        })
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})


# ── FastMCP registration ──────────────────────────────────────────────────
mcp = FastMCP("Venus_Denovo_MCP")


@mcp.tool(name="proteinmpnn_design")
def mcp_proteinmpnn_design(
    pdb_path: str,
    designed_chains: Optional[List[str]] = None,
    fixed_chains: Optional[List[str]] = None,
    fixed_residues: Optional[Dict[str, List[int]]] = None,
    homomer: bool = False,
    num_sequences: int = 8,
    temperatures: Optional[List[float]] = None,
    omit_aas: str = "X",
    model_name: str = "v_48_020",
    backbone_noise: float = 0.0,
    ca_only: bool = False,
    out_dir: Optional[str] = None,
) -> str:
    """Run ProteinMPNN sequence design on a PDB structure. Returns status JSON
    with `fasta_path` and a `sequences_preview` of the first 10 designs.

    Args mirror `protein_mpnn_run.py`: pass `designed_chains` (e.g. ["A"]) to
    design only those chains; `fixed_chains` to freeze others; `fixed_residues`
    as {chain: [residue_indices]} to lock specific positions (e.g. active-site
    or catalytic triad); `homomer=True` for symmetric multimer; `temperatures`
    is a list of sampling T (default [0.1]); `model_name` picks among
    v_48_002 / v_48_010 / v_48_020 / v_48_030 (higher = noisier training, more
    diverse designs); `ca_only=True` uses the Cα-only model variant.
    """
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


@mcp.tool(name="proteinmpnn_score")
def mcp_proteinmpnn_score(
    pdb_path: str,
    fasta_path: Optional[str] = None,
    designed_chains: Optional[List[str]] = None,
    num_batches: int = 1,
    model_name: str = "v_48_020",
    backbone_noise: float = 0.0,
    out_dir: Optional[str] = None,
) -> str:
    """Score sequences against a backbone with ProteinMPNN. Returns status
    JSON with the per-sequence log-likelihood file path and a preview.

    Pass `fasta_path=None` to score the native sequence of `pdb_path`;
    otherwise scores each record in the FASTA against the backbone.
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
