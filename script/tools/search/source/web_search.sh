#!/usr/bin/env bash
# Web search: DuckDuckGo, Tavily.
# Run from project root.
# Example: ./script/search/source/web_search.sh

python src/tools/search/source/web_search.py \
    --query "protein language model" \
    --max_results 5 \
    --source "duckduckgo,tavily"
