"""Shared model request contracts, independent of API and inference imports."""
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel, Field

_PROTEINMPNN_CKPT_ROOT = Path(__file__).resolve().parents[2] / 'ckpt/ProteinMPNN'
_PROTEINMPNN_VANILLA_DIR = _PROTEINMPNN_CKPT_ROOT / 'vanilla_model_weights'
_PROTEINMPNN_SOLUBLE_DIR = _PROTEINMPNN_CKPT_ROOT / 'soluble_model_weights'
_PROTEINMPNN_CA_DIR = _PROTEINMPNN_CKPT_ROOT / 'ca_model_weights'

class DirectedEvolutionBody(BaseModel):
    input_mode: str = Field(default="sequence", description="sequence or structure")
    function_selection: Optional[str] = Field(default=None)
    file_path: Optional[str] = Field(default=None)
    sequence: Optional[str] = Field(default=None)
    model_name: str = Field(default="ESM2-650M")
    enable_ai: bool = Field(default=False)
    llm_provider: str = Field(default="DeepSeek")
    user_api_key: str = Field(default="")


class ProteinDiscoveryBody(BaseModel):
    pdb_file: str = Field(...)
    protect_start: int = Field(default=1)
    protect_end: int = Field(default=100)
    mmseqs_threads: int = Field(default=96)
    mmseqs_iterations: int = Field(default=3)
    mmseqs_max_seqs: int = Field(default=100)
    cluster_min_seq_id: float = Field(default=0.5)
    cluster_threads: int = Field(default=96)
    top_n_threshold: int = Field(default=10)
    evalue_threshold: float = Field(default=1e-5)


class SequenceDesignBody(BaseModel):
    structure_file: str = Field(..., description="PDB file path.")
    model_family: str = Field(default="soluble", description="soluble | vanilla | ca")
    designed_chains: List[str] = Field(default_factory=list)
    fixed_chains: List[str] = Field(default_factory=list)
    fixed_residues_text: str = Field(default="")
    homomer: bool = Field(default=False)
    num_sequences: int = Field(default=8, ge=1, le=512)
    temperatures: List[float] = Field(default_factory=lambda: [0.1])
    omit_aas: str = Field(default="X")
    model_name: str = Field(default="v_48_020")
    backbone_noise: float = Field(default=0.2)
    ca_only: bool = Field(default=False)
    use_soluble_model: bool = Field(default=True)
    seed: int = Field(default=0)
    batch_size: int = Field(default=1, ge=1)
    max_length: int = Field(default=200000, ge=1)
    tied_positions_text: str = Field(default="")
    omit_aa_rules_text: str = Field(default="")
    aa_bias_text: str = Field(default="")
    bias_by_residue_text: str = Field(default="")
    pssm_rules_text: str = Field(default="")
    pssm_multi: float = Field(default=0.0)
    pssm_threshold: float = Field(default=0.0)
    pssm_log_odds_flag: int = Field(default=0)
    pssm_bias_flag: int = Field(default=0)


def _list_proteinmpnn_model_names(directory: Path) -> list[str]:
    if not directory.exists() or not directory.is_dir():
        return []
    names = sorted({p.stem for p in directory.glob("*.pt")})
    return names


def _proteinmpnn_model_options() -> dict[str, list[str]]:
    from mcp_tools.models.profiles import load_runtime_settings, repository_path
    configured = load_runtime_settings().get('proteinmpnn', {}).get('weights')
    root = repository_path(configured) if configured else None
    vanilla = _list_proteinmpnn_model_names(root / 'vanilla_model_weights' if root else _PROTEINMPNN_VANILLA_DIR)
    soluble = _list_proteinmpnn_model_names(root / 'soluble_model_weights' if root else _PROTEINMPNN_SOLUBLE_DIR)
    ca = _list_proteinmpnn_model_names(root / 'ca_model_weights' if root else _PROTEINMPNN_CA_DIR)
    fallback = ["v_48_020", "v_48_002"]
    return {
        "vanilla": vanilla or fallback,
        "soluble": soluble or fallback,
        "ca": ca or fallback,
    }
