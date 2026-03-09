#!/usr/bin/env bash
# ProtSSN: zero-shot structure mutation. Run from project root.

python src/tools/mutation/models/protssn.py \
    --structure_file example/database/alphafold/A0A1B0GTW7.pdb \
    --output_csv example/mutation/output/A0A0C5B5G6_protssn.csv
