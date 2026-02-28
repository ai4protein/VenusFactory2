#!/usr/bin/env bash
# ESM-1v: zero-shot sequence mutation. Run from project root.

python src/tools/mutation/models/esm1v.py \
    --fasta_file example/download/uniprot_sequences/A0A0C5B5G6.fasta \
    --output_csv example/mutation/output/A0A0C5B5G6_esm1v.csv
