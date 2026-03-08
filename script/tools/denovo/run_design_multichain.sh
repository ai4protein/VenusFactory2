#!/usr/bin/env bash
# ProteinMPNN multi-chain design: redesign chain A only, chains B/C/D fixed as context.
# Input : multi-chain PDB (4HHB: hemoglobin, chains A B C D)
# Output: FASTA written to temp_outputs/.../ProteinMPNN/Design/seqs/
# Example: ./script/tools/denovo/run_design_multichain.sh

python src/tools/denovo/tools_cmd.py \
    design \
    --pdb_path example/database/rcsb/4HHB.pdb \
    --designed_chains A \
    --num_sequences 8 \
    --temperatures 0.1 \
    --model_name v_48_020
