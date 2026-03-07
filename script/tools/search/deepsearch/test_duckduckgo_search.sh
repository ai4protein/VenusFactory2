#!/bin/bash
# Test script for duckduckgo_search.py
echo "Testing duckduckgo_search.py"
python src/tools/search/deepsearch/duckduckgo_search.py --query "transformer model" --max_results 2
