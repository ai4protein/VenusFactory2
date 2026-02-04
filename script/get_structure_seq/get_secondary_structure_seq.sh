#!/bin/bash
### Secondary structure sequence extraction (DSSP)
# Extract ss8_seq and ss3_seq from PDB for structure-aware models
# Dependency: biopython, dssp

### Parameters
# data_name: Dataset name
# data_type: PDB directory name (e.g., alphafold_pdb alphafold_pdb_noise_0.5)
# Output: dataset/$data_name/"$data_type"_ss.csv (columns: name, aa_seq, ss8_seq, ss3_seq)

data_name=test
data_type=esmfold_pdb
python src/data/get_secondary_structure_seq.py \
    --pdb_dir dataset/$data_name/$data_type \
    --num_workers 6 \
    --out_file dataset/$data_name/${data_type}_ss.csv
