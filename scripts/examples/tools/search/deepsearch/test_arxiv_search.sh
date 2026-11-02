#!/usr/bin/env bash
set -euo pipefail
VF_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
cd -- "$VF_REPO_ROOT"
export PYTHONPATH="$VF_REPO_ROOT:$VF_REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

# Test script for arxiv_search.py

echo "Testing arxiv_search.py"
python -m mcp_tools.tools.search.deepsearch.arxiv_search --query "alphafold 3" --max_results 2
