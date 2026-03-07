#!/bin/bash
# Test script for arxiv_search.py

echo "Testing arxiv_search.py"
python src/tools/search/deepsearch/arxiv_search.py --query "alphafold 3" --max_results 2
