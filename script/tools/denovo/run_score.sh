#!/usr/bin/env bash
# ProteinMPNN sequence scoring: score native PDB sequence against its own backbone.
# Input : single-chain PDB (AlphaFold2 structure)
# Output: .npz scores in temp_outputs/.../ProteinMPNN/Score/score_only/
# Example: ./script/tools/denovo/run_score.sh

python src/tools/denovo/tools_cmd.py \
    score \
    --pdb_path example/database/alphafold/A0A1B0GTW7.pdb \
    --num_batches 1 \
    --model_name v_48_020
