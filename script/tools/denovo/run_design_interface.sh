#!/usr/bin/env bash
# ProteinMPNN interface design: redesign chain A (binder), chain B fixed as target.
# Input : multi-chain PDB (4HHB: hemoglobin, chains A B C D)
# Output: FASTA written to temp_outputs/.../ProteinMPNN/Design/seqs/
# Example: ./script/tools/denovo/run_design_interface.sh

python src/tools/denovo/tools_cmd.py \
    design \
    --pdb_path example/database/rcsb/4HHB.pdb \
    --designed_chains A \
    --fixed_chains B \
    --num_sequences 8 \
    --temperatures 0.1 \
    --model_name v_48_020
