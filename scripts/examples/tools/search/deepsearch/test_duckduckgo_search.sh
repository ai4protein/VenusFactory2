#!/usr/bin/env bash
set -euo pipefail
VF_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
cd -- "$VF_REPO_ROOT"
export PYTHONPATH="$VF_REPO_ROOT:$VF_REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

# Test script for duckduckgo_search.py
echo "Testing duckduckgo_search.py"
python -m mcp_tools.tools.search.deepsearch.duckduckgo_search --query "transformer model" --max_results 2
