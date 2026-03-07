#!/bin/bash
# Test script for tavily_search.py

echo "Testing tavily_search.py"
python src/tools/search/deepsearch/tavily_search.py --query "current weather in tokyo" --max_results 2
