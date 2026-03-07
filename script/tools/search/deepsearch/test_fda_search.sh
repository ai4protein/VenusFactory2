#!/bin/bash
# Test script for fda_search.py

echo "Testing fda_search.py"
python src/tools/search/deepsearch/fda_search.py --query "aspirin" --max_results 2
