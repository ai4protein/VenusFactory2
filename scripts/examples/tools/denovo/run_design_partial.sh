#!/usr/bin/env bash
set -euo pipefail
VF_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd -- "$VF_REPO_ROOT"
export PYTHONPATH="$VF_REPO_ROOT:$VF_REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

# ProteinMPNN partial fixed design: fix residues 1,5,10 on chain A, redesign the rest.
# Input : single-chain PDB (AlphaFold2 structure)
# Output: FASTA written to temp_outputs/.../ProteinMPNN/Design/seqs/
# Example: ./scripts/examples/tools/denovo/run_design_partial.sh

python -m mcp_tools.models.pretrained.design.tools_cmd \
    design \
    --pdb_path example/database/alphafold/A0A1B0GTW7.pdb \
    --fixed_residues_json '{"A": [1, 5, 10, 20, 50]}' \
    --num_sequences 8 \
    --temperatures 0.1 \
    --model_name v_48_020
