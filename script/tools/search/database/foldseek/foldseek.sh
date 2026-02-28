#!/usr/bin/env bash
# FoldSeek structure search: submit PDB, download alignments, extract sequences covering a region.
# Run from project root.
# Example: ./script/tools/search/database/foldseek/foldseek.sh

python src/tools/search/database/foldseek/__main__.py \
    --pdb_file example/download/alphafold2_structures/A0A1B0GTW7.pdb \
    --output_dir example/download/foldseek_search \
    --protect_start 1 \
    --protect_end 10
