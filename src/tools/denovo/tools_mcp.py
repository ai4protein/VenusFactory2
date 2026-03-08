"""
tools/denovo/tools_mcp.py — ProteinMPNN local call layer

Two call_* functions corresponding to protein_mpnn_function.py,
returning JSON strings. Each result also includes a "sequences_preview"
field with the first N FASTA records so the agent can immediately see
the designed/scored sequences without having to open the file.
"""
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

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
) -> str:
    try:
        out_fasta = proteinmpnn_score(
            pdb_path=pdb_path,
            fasta_path=fasta_path,
            designed_chains=designed_chains,
            num_batches=num_batches,
            model_name=model_name,
            backbone_noise=backbone_noise,
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
