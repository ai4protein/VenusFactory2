#!/usr/bin/env bash
# VenusPLM: zero-shot sequence mutation. Run from project root.

python src/tools/mutation/models/venusplm.py \
    --fasta_file example/download/uniprot_sequences/A0A0C5B5G6.fasta \
    --output_csv example/mutation/output/A0A0C5B5G6_venusplm.csv
