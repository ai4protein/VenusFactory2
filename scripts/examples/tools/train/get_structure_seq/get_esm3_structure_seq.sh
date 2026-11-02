#!/usr/bin/env bash
set -euo pipefail
VF_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
cd -- "$VF_REPO_ROOT"
export PYTHONPATH="$VF_REPO_ROOT:$VF_REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

### ESM3 structure sequence extraction
# Extract ESM3 structure tokens from PDB for structure-aware models
# Requires GPU and esm package

### Parameters
# dataset: Dataset name
# pdb_type: PDB type (e.g., alphafold esmfold)
# Output: dataset/$dataset/"$pdb_type"_esm3.csv (columns: name, esm3_structure_seq, plddt)

dataset=test
pdb_type=esmfold_pdb
python -m mcp_tools.models.factory.training.data.get_esm3_structure_seq \
    --pdb_dir dataset/$dataset/$pdb_type \
    --out_file dataset/$dataset/${pdb_type}_esm3.csv
