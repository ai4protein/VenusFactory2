#!/usr/bin/env bash
python src/tools/search/database/ncbi/nvbi_sequence.py \
    --id NP_000517.1 \
    --out_dir download/ncbi_sequences

python src/tools/search/database/ncbi/nvbi_sequence.py \
    --file download/ncbi.txt \
    --out_dir download/ncbi_sequences
