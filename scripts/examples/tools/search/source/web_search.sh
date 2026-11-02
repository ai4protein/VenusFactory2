#!/usr/bin/env bash
set -euo pipefail
VF_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
cd -- "$VF_REPO_ROOT"
export PYTHONPATH="$VF_REPO_ROOT:$VF_REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

# Web search: DuckDuckGo, Tavily.
# Run from project root.
# Example: ./scripts/examples/tools/search/source/web_search.sh

python -m mcp_tools.tools.search.source.web_search \
    --query "protein language model" \
    --max_results 5 \
    --source "duckduckgo,tavily"
