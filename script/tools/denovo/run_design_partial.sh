#!/usr/bin/env bash
# ProteinMPNN partial fixed design: fix residues 1,5,10 on chain A, redesign the rest.
# Input : single-chain PDB (AlphaFold2 structure)
# Output: FASTA written to temp_outputs/.../ProteinMPNN/Design/seqs/
# Example: ./script/tools/denovo/run_design_partial.sh

python src/tools/denovo/tools_cmd.py \
    design \
    --pdb_path example/database/alphafold/A0A1B0GTW7.pdb \
    --fixed_residues_json '{"A": [1, 5, 10, 20, 50]}' \
    --num_sequences 8 \
    --temperatures 0.1 \
    --model_name v_48_020
