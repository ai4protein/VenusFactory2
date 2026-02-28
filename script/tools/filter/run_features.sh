#!/usr/bin/env bash
# Protein property prediction (local via tools_cmd). Run from project root.
# Supported --task: physchem, rsa, sasa, secondary-structure, all
#   physchem          - physical and chemical properties (FASTA or PDB)
#   rsa               - relative solvent accessible surface area (PDB only)
#   sasa              - solvent accessible surface area (PDB only)
#   secondary-structure - secondary structure (PDB only)
#   all               - all of the above from one file (FASTA or PDB)

python src/tools/predict/tools_cmd.py \
    features \
    --task all \
    --fasta_file example/download/uniprot_sequences/A0A0C5B5G6.fasta \
    --output_file example/predict/output/A0A0C5B5G6_features.json
