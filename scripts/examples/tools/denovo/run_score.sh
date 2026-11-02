#!/usr/bin/env bash
set -euo pipefail
VF_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd -- "$VF_REPO_ROOT"
export PYTHONPATH="$VF_REPO_ROOT:$VF_REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

# ProteinMPNN sequence scoring: score native PDB sequence against its own backbone.
# Input : single-chain PDB (AlphaFold2 structure)
# Output: .npz scores in temp_outputs/.../ProteinMPNN/Score/score_only/
# Example: ./scripts/examples/tools/denovo/run_score.sh

python -m mcp_tools.models.pretrained.design.tools_cmd \
    score \
    --pdb_path example/database/alphafold/A0A1B0GTW7.pdb \
    --num_batches 1 \
    --model_name v_48_020
