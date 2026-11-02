#!/usr/bin/env bash
set -euo pipefail
VF_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd -- "$VF_REPO_ROOT"
export PYTHONPATH="$VF_REPO_ROOT:$VF_REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

# ProteinMPNN homomeric symmetric design: chains A/B/C/D designed with tied positions.
# Input : multi-chain PDB (4HHB: hemoglobin, 4 identical-topology subunits)
# Output: FASTA written to temp_outputs/.../ProteinMPNN/Design/seqs/
# Example: ./scripts/examples/tools/denovo/run_design_homomer.sh

python -m mcp_tools.models.pretrained.design.tools_cmd \
    design \
    --pdb_path example/database/rcsb/4HHB.pdb \
    --designed_chains A B C D \
    --homomer \
    --num_sequences 8 \
    --temperatures 0.1 \
    --model_name v_48_020
