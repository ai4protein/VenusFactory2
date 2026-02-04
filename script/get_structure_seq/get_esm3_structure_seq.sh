#!/bin/bash
### ESM3 structure sequence extraction
# Extract ESM3 structure tokens from PDB for structure-aware models
# Requires GPU and esm package

### Parameters
# dataset: Dataset name
# pdb_type: PDB type (e.g., alphafold esmfold)
# Output: dataset/$dataset/"$pdb_type"_esm3.csv (columns: name, esm3_structure_seq, plddt)

dataset=test
pdb_type=esmfold_pdb
python src/data/get_esm3_structure_seq.py \
    --pdb_dir dataset/$dataset/$pdb_type \
    --out_file dataset/$dataset/${pdb_type}_esm3.csv
