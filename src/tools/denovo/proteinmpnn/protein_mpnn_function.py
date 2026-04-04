"""
protein_mpnn_function.py — ProteinMPNN unified Python function interface.

Two public functions:
    proteinmpnn_design  – covers all design scenarios: single-chain, multi-chain,
                          interface, homomer, and partial-fixed design
    proteinmpnn_score   – sequence scoring (NLL), no sampling

Output paths are managed automatically via _get_save_path; callers do not pass out_folder.
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

_SRC = _HERE.parents[2]   # proteinmpnn -> denovo -> tools -> src
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from protein_mpnn_run import proteinmpnn_run


def _get_save_path(subdir1: str, subdir2: str = None) -> Path:
    """Mirror of web.utils.common_utils.get_save_path without the gradio import."""
    base = os.getenv("TEMP_OUTPUTS_DIR", "temp_outputs")
    now = datetime.now()
    date_path = Path(base) / f"{now.year}/{now.month:02d}/{now.day:02d}"
    path = date_path / subdir1 / subdir2 if subdir2 else date_path / subdir1
    path.mkdir(parents=True, exist_ok=True)
    return path


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _default_args(pdb_path: str, out_folder: str, **kwargs) -> argparse.Namespace:
    defaults = dict(
        suppress_print=0,
        ca_only=False,
        use_soluble_model=False,
        path_to_model_weights="",
        model_name="v_48_020",
        seed=0,
        save_score=0,
        save_probs=0,
        score_only=0,
        path_to_fasta="",
        conditional_probs_only=0,
        conditional_probs_only_backbone=0,
        unconditional_probs_only=0,
        backbone_noise=0.0,
        num_seq_per_target=8,
        batch_size=1,
        max_length=200000,
        sampling_temp="0.1",
        out_folder=out_folder,
        pdb_path=pdb_path,
        pdb_path_chains="",
        jsonl_path="",
        chain_id_jsonl="",
        fixed_positions_jsonl="",
        omit_AAs="X",
        bias_AA_jsonl="",
        bias_by_res_jsonl="",
        omit_AA_jsonl="",
        pssm_jsonl="",
        pssm_multi=0.0,
        pssm_threshold=0.0,
        pssm_log_odds_flag=0,
        pssm_bias_flag=0,
        tied_positions_jsonl="",
    )
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def _write_jsonl(data: dict, tmpdir: str, name: str) -> str:
    path = os.path.join(tmpdir, name)
    with open(path, "w") as f:
        f.write(json.dumps(data) + "\n")
    return path


def proteinmpnn_design(
    pdb_path: str,
    designed_chains: Optional[List[str]] = None,
    fixed_chains: Optional[List[str]] = None,
    fixed_residues: Optional[Dict[str, List[int]]] = None,
    homomer: bool = False,
    num_sequences: int = 8,
    temperatures: List[float] = None,
    omit_aas: str = "X",
    model_name: str = "v_48_020",
    backbone_noise: float = 0.0,
    ca_only: bool = False,
    use_soluble_model: bool = False,
) -> str:
    """
    ProteinMPNN sequence design covering all scenarios. Returns the output FASTA path.

    Scenario selection (determined by parameter combination)
    ---------------------------------------------------------
    Single-chain design   designed_chains=None  (default)
    Multi-chain design    designed_chains=['A']
    Interface design      designed_chains=['B'], fixed_chains=['A']
    Homomeric design      designed_chains=['A','B','C'], homomer=True
    Partial-fixed design  fixed_residues={'A': [1, 5, 10]}
    ---------------------------------------------------------

    Parameters
    ----------
    pdb_path        : Input PDB file (single file, supports multiple chains)
    designed_chains : Chain IDs to redesign; None = all chains
    fixed_chains    : Chains held fixed as structural context (interface design);
                      requires designed_chains to be set; written as chain_id_jsonl
    fixed_residues  : Residue positions to keep at native identity, e.g. {'A': [1, 5, 10]} (1-indexed)
    homomer         : If True, build tied-position constraints across designed_chains (homomeric symmetry)
    num_sequences   : Number of sequences to generate
    temperatures    : Sampling temperature list, default [0.1]
    omit_aas        : Amino acid letters to exclude from design
    model_name      : Model weights name, e.g. 'v_48_020'
    backbone_noise  : Gaussian noise std added to backbone coordinates
    ca_only         : Use the CA-only model
    use_soluble_model : Use soluble ProteinMPNN weights
    """
    if temperatures is None:
        temperatures = [0.1]
    out_folder = str(_get_save_path("ProteinMPNN", "Design"))
    pdb_name = Path(pdb_path).stem
    sampling_temp = " ".join(str(t) for t in temperatures)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        kwargs: dict = dict(
            pdb_path=pdb_path,
            out_folder=out_folder,
            num_seq_per_target=num_sequences,
            sampling_temp=sampling_temp,
            omit_AAs=omit_aas,
            model_name=model_name,
            backbone_noise=backbone_noise,
            ca_only=ca_only,
            use_soluble_model=use_soluble_model,
        )

        # interface design: binder (designed) + target (fixed) -> chain_id_jsonl
        if designed_chains and fixed_chains:
            chain_id_jsonl = _write_jsonl(
                {pdb_name: [designed_chains, fixed_chains]}, tmpdir, "chain_id.jsonl"
            )
            kwargs["chain_id_jsonl"] = chain_id_jsonl

        # multi-chain or single-chain: limit to designed_chains only, no fixed context
        elif designed_chains:
            kwargs["pdb_path_chains"] = " ".join(designed_chains)

        # partial-fixed: write fixed_positions_jsonl
        if fixed_residues:
            fixed_jsonl = _write_jsonl({pdb_name: fixed_residues}, tmpdir, "fixed_positions.jsonl")
            kwargs["fixed_positions_jsonl"] = fixed_jsonl

        # homomeric: build tied_positions from designed_chains
        if homomer:
            chains = designed_chains or []
            if not chains:
                from protein_mpnn_utils import parse_PDB
                pdb_dict_list = parse_PDB(pdb_path)
                chains = sorted([k[-1:] for k in pdb_dict_list[0] if k.startswith("seq_chain_")])
            from protein_mpnn_utils import parse_PDB
            pdb_dict_list = parse_PDB(pdb_path, input_chain_list=chains)
            chain_length = len(pdb_dict_list[0][f"seq_chain_{chains[0]}"])
            tied_positions = [{c: [i] for c in chains} for i in range(1, chain_length + 1)]
            tied_jsonl = _write_jsonl({pdb_name: tied_positions}, tmpdir, "tied_positions.jsonl")
            kwargs["tied_positions_jsonl"] = tied_jsonl

        args = _default_args(**kwargs)
        proteinmpnn_run(args)
    
    # proteinmpnn_run now saves directly to out_folder/seqs/YYYYMMDD_HHMMSS_pdbname.fasta
    # Find the matching file that was just created
    seqs_dir = Path(out_folder) / "seqs"
    files = list(seqs_dir.glob(f"*_{pdb_name}.fasta"))
    if not files:
        raise FileNotFoundError(f"ProteinMPNN design output not found in {seqs_dir}")
    
    # If there are multiple, return the most recently modified one
    latest_file = max(files, key=os.path.getmtime)
    
    return str(latest_file)


def proteinmpnn_score(
    pdb_path: str,
    fasta_path: Optional[str] = None,
    designed_chains: Optional[List[str]] = None,
    num_batches: int = 1,
    model_name: str = "v_48_020",
    backbone_noise: float = 0.0,
) -> str:
    """
    Score sequences against a backbone structure (NLL = -log_prob; lower is better).
    No sampling — evaluation only. Outputs a timestamped FASTA with scores in headers.

    Parameters
    ----------
    pdb_path        : Backbone PDB file
    fasta_path      : FASTA of sequences to score; None = score the native PDB sequence
    designed_chains : Chains to evaluate; None = all chains
    num_batches     : Number of stochastic forward passes to average over

    Returns
    -------
    str  Path to the output FASTA file ({timestamp}_{pdb_name}.fasta)
    """
    out_folder = str(_get_save_path("ProteinMPNN", "Score"))
    kwargs: dict = dict(
        pdb_path=pdb_path,
        out_folder=out_folder,
        score_only=1,
        num_seq_per_target=num_batches,
        model_name=model_name,
        backbone_noise=backbone_noise,
        path_to_fasta=fasta_path or "",
    )
    if designed_chains:
        kwargs["pdb_path_chains"] = " ".join(designed_chains)
    args = _default_args(**kwargs)
    proteinmpnn_run(args)

    # convert .npz score files -> single timestamped FASTA
    import numpy as np
    _ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
    score_dir = Path(out_folder) / "score_only"
    pdb_name = Path(pdb_path).stem
    ts = _timestamp()
    out_fasta = score_dir / f"{ts}_{pdb_name}.fasta"
    with open(out_fasta, "w") as fh:
        for npz_file in sorted(score_dir.glob("*.npz")):
            data = np.load(npz_file, allow_pickle=True)
            seq = "".join(_ALPHABET[i] for i in data["S"])
            score_mean = float(np.mean(data["score"]))
            global_score_mean = float(np.mean(data["global_score"]))
            label = npz_file.stem
            fh.write(f">{label}, score={score_mean:.4f}, global_score={global_score_mean:.4f}\n{seq}\n")
    return str(out_fasta)
