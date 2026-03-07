#!/bin/bash
# Test script for pubmed_search.py

echo "Testing pubmed_search.py"
python src/tools/search/deepsearch/pubmed_search.py --query "BRCA1" --max_results 2
