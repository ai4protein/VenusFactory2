#!/usr/bin/env bash
# VenusREM: zero-shot structure mutation. Run from project root.

python src/tools/mutation/models/venusrem.py \
    --structure_file example/download/alphafold2_structures/A0A1B0GTW7.pdb \
    --output_csv example/mutation/output/A0A1B0GTW7_venusrem.csv
