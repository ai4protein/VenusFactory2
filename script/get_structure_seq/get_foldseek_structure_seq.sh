#!/bin/bash
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
python src/data/get_foldseek_structure_seq.py \
    --pdb_dir dataset/$dataset/$pdb_type \
    --out_file dataset/$dataset/${pdb_type}_fs.csv
