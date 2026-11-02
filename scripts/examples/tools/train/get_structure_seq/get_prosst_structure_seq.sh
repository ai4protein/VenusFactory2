#!/usr/bin/env bash
set -euo pipefail
VF_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
cd -- "$VF_REPO_ROOT"
export PYTHONPATH="$VF_REPO_ROOT:$VF_REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

### ProSST structure token extraction
# Extract ProSST structure tokens from PDB for ProSST series models
# structure_vocab_size: 20/64/128/512/1024/2048/4096, corresponding to ProSST-20/128/512/1024/2048/4096

### Parameters
# dataset: Dataset name
# pdb_type: PDB type (e.g., alphafold_pdb esmfold_pdb)
# structure_vocab_size: Structure vocab size, commonly 2048 (ProSST-2048)
# Output: dataset/$dataset/"$pdb_type"_prosst.csv (columns: name, aa_seq, {vocab}_struct_tokens)

# Example 1: test ESMFold PDB, ProSST-2048
dataset=test
pdb_type=esmfold_pdb
structure_vocab_size=2048
python -m mcp_tools.models.factory.training.data.get_prosst_str_token \
    --pdb_dir dataset/$dataset/$pdb_type \
    --structure_vocab_size $structure_vocab_size \
    --num_processes 12 \
    --num_threads 16 \
    --out_file dataset/$dataset/${pdb_type}_prosst.csv
