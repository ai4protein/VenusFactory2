#!/usr/bin/env bash
# ProSST-2048: zero-shot structure mutation. Run from project root.

python src/tools/mutation/models/prosst.py \
    --structure_file example/download/alphafold2_structures/A0A1B0GTW7.pdb \
    --output_csv example/mutation/output/A0A1B0GTW7_prosst.csv
