#!/usr/bin/env bash
set -euo pipefail
VF_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
cd -- "$VF_REPO_ROOT"
export PYTHONPATH="$VF_REPO_ROOT:$VF_REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

# Dataset search: GitHub, Hugging Face.
# Run from project root.
# Example: ./scripts/examples/tools/search/source/dataset.sh

python -m mcp_tools.tools.search.source.dataset_search \
    --query "protein dataset" \
    --max_results 5 \
    --source "github,hugging_face"
