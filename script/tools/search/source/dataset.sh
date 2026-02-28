#!/usr/bin/env bash
# Dataset search: GitHub, Hugging Face.
# Run from project root.
# Example: ./script/search/source/dataset.sh

python src/tools/search/source/dataset_search.py \
    --query "protein dataset" \
    --max_results 5 \
    --source "github,hugging_face"
