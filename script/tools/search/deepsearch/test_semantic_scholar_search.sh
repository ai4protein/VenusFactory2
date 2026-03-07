#!/bin/bash
# Test script for semantic_scholar_search.py

echo "Testing semantic_scholar_search.py"
python src/tools/search/deepsearch/semantic_scholar_search.py --query "transformer" --max_results 2
