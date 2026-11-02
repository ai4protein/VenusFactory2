#!/usr/bin/env bash
set -euo pipefail
VF_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd -- "$VF_REPO_ROOT"
export PYTHONPATH="$VF_REPO_ROOT:$VF_REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

# ProteinMPNN multi-chain design: redesign chain A only, chains B/C/D fixed as context.
# Input : multi-chain PDB (4HHB: hemoglobin, chains A B C D)
# Output: FASTA written to temp_outputs/.../ProteinMPNN/Design/seqs/
# Example: ./scripts/examples/tools/denovo/run_design_multichain.sh

python -m mcp_tools.models.pretrained.design.tools_cmd \
    design \
    --pdb_path example/database/rcsb/4HHB.pdb \
    --designed_chains A \
    --num_sequences 8 \
    --temperatures 0.1 \
    --model_name v_48_020
