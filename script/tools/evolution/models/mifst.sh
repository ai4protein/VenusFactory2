#!/usr/bin/env bash
# MIF-ST: zero-shot structure mutation. Run from project root.

python src/tools/mutation/models/mifst.py \
    --structure_file example/download/alphafold2_structures/A0A1B0GTW7.pdb \
    --output_csv example/mutation/output/A0A1B0GTW7_mifst.csv
