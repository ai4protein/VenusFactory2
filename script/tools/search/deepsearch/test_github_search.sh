#!/bin/bash
# Test script for github_search.py

echo "Testing github_search.py"
python src/tools/search/deepsearch/github_search.py --query "protein folding" --max_results 2
