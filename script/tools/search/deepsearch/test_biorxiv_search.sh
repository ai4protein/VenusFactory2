#!/bin/bash
# Test script for biorxiv_search.py
echo "Testing biorxiv_search.py"
python src/tools/search/deepsearch/biorxiv_search.py --query "bioinformatics" --max_results 2
