#!/usr/bin/env bash
# Literature search: arxiv, pubmed, biorxiv, semantic_scholar.
# Run from project root.
# Example: ./script/search/source/literature.sh

python src/tools/search/source/literature.py \
    --query "stability" \
    --max_results 5 \
    --source "pubmed"