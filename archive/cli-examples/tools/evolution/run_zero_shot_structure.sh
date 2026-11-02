#!/usr/bin/env bash
# Zero-shot structure-based mutation prediction (local). Run from project root.
# Supported --model_name: ESM-IF1, MIF-ST, ProtSSN, ProSST-2048, SaProt, VenusREM
# Example: ./script/mutation/run_zero_shot_structure.sh


python src/tools/mutation/tools_cmd.py \
    zero-shot-structure \
    --structure_file example/download/alphafold2_structures/A0A1B0GTW7.pdb \
    --model_name ESM-IF1 \
    --output_csv example/mutation/output/A0A1B0GTW7_esmif1.csv
