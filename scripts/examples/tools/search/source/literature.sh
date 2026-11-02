#!/usr/bin/env bash
set -euo pipefail
VF_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
cd -- "$VF_REPO_ROOT"
export PYTHONPATH="$VF_REPO_ROOT:$VF_REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

# Literature search: arxiv, pubmed, biorxiv, semantic_scholar.
# Run from project root.
# Example: ./scripts/examples/tools/search/source/literature.sh

python -m mcp_tools.tools.search.source.literature_search \
    --query "stability" \
    --max_results 5 \
    --source "pubmed"
