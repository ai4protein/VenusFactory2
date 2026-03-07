#!/bin/bash
# Test script for google_scholar_search.py

echo "Testing google_scholar_search.py"
python src/tools/search/deepsearch/google_scholar_search.py --query "alphafold" --max_results 2
