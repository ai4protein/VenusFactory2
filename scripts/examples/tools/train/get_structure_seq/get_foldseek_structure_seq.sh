#!/usr/bin/env bash
set -euo pipefail
VF_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
cd -- "$VF_REPO_ROOT"
export PYTHONPATH="$VF_REPO_ROOT:$VF_REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

### FoldSeek structure sequence extraction
# Extract FoldSeek 3Di sequences from PDB structures for VenusREM and other foldseek-based models
# Dependency: conda install -c conda-forge -c bioconda foldseek

### Parameters
# dataset: Dataset name (e.g., DeepSol DeepLocBinary DeepET_Topt)
# pdb_type: PDB type (e.g., alphafold_pdb esmfold_pdb)
# Output: dataset/$dataset/"$pdb_type"_fs.csv (columns: name, foldseek_seq)

# Example 1: DeepET ESMFold PDB
dataset=test
pdb_type=esmfold_pdb
python -m mcp_tools.models.factory.training.data.get_foldseek_structure_seq \
    --pdb_dir dataset/$dataset/$pdb_type \
    --out_file dataset/$dataset/${pdb_type}_fs.csv
