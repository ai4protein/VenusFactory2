#!/bin/bash
# Test script for hugging_face_search.py

echo "Testing hugging_face_search.py"
python src/tools/search/deepsearch/hugging_face_search.py --query "protein fold" --max_results 2
