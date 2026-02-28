#!/usr/bin/env bash
# ProtSSN: zero-shot structure mutation. Run from project root.

python src/tools/mutation/models/protssn.py \
    --structure_file example/download/alphafold2_structures/A0A0C5B5G6.pdb \
    --output_csv example/mutation/output/A0A0C5B5G6_protssn.csv
