#!/usr/bin/env bash
# Zero-shot sequence-based mutation prediction (local). Run from project root.
# Supported --model_name: ESM-1b, ESM2-650M, ESM-1v, VenusPLM
# Example: ./script/mutation/run_zero_shot_sequence.sh

python src/tools/mutation/tools_cmd.py \
    zero-shot-sequence \
    --fasta_file example/download/uniprot_sequences/A0A0C5B5G6.fasta \
    --model_name VenusPLM \
    --output_csv example/mutation/output/A0A0C5B5G6_venusplm.csv
