#!/usr/bin/env bash
# ESM2-650M: zero-shot sequence mutation. Run from project root.

python src/tools/mutation/models/esm2.py \
    --fasta_file example/download/uniprot_sequences/A0A0C5B5G6.fasta \
    --output_csv example/mutation/output/A0A0C5B5G6_esm2.csv
