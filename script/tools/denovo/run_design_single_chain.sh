#!/usr/bin/env bash
# ProteinMPNN single-chain sequence design. Run from project root.
# Input : single-chain PDB (AlphaFold2 structure)
# Output: FASTA written to temp_outputs/.../ProteinMPNN/Design/seqs/
# Example: ./script/tools/denovo/run_design_single_chain.sh

python src/tools/denovo/tools_cmd.py \
    design \
    --pdb_path example/database/alphafold/A0A1B0GTW7.pdb \
    --num_sequences 8 \
    --temperatures 0.1 0.2 \
    --model_name v_48_020
