#!/usr/bin/env bash
set -euo pipefail
VF_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd -- "$VF_REPO_ROOT"
export PYTHONPATH="$VF_REPO_ROOT:$VF_REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

# Run AlphaFold operations tests (query_* and download_* from alphafold_operations).
# Output: example/database/alphafold/ (structure, metadata, query samples, logs).
# Run from project root: bash scripts/examples/tools/database/test_alphafold.sh

OUT_DIR="example/database/alphafold"
mkdir -p "$OUT_DIR"

echo "=== alphafold_operations (query_* and download_*) ==="
python -m mcp_tools.tools.database.alphafold.alphafold_operations --test 2>&1 | tee "${OUT_DIR}/test_alphafold_operations.log"

echo ""
echo "AlphaFold operations tests finished. Output under example/database/alphafold"
