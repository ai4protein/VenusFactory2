#!/usr/bin/env bash
# ESM-IF1: zero-shot structure mutation. Run from project root.

python src/tools/mutation/models/esmif1.py \
    --structure_file example/download/alphafold2_structures/A0A1B0GTW7.pdb \
    --output_csv example/mutation/output/A0A1B0GTW7_esmif1.csv
