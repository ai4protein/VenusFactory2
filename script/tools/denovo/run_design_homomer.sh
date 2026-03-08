#!/usr/bin/env bash
# ProteinMPNN homomeric symmetric design: chains A/B/C/D designed with tied positions.
# Input : multi-chain PDB (4HHB: hemoglobin, 4 identical-topology subunits)
# Output: FASTA written to temp_outputs/.../ProteinMPNN/Design/seqs/
# Example: ./script/tools/denovo/run_design_homomer.sh

python src/tools/denovo/tools_cmd.py \
    design \
    --pdb_path example/database/rcsb/4HHB.pdb \
    --designed_chains A B C D \
    --homomer \
    --num_sequences 8 \
    --temperatures 0.1 \
    --model_name v_48_020
