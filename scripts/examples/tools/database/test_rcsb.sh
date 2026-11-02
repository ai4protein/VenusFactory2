#!/usr/bin/env bash
set -euo pipefail
VF_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd -- "$VF_REPO_ROOT"
export PYTHONPATH="$VF_REPO_ROOT:$VF_REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

# Run tests for all RCSB PDB tools (non-helper functions in rcsb/*.py).
# Output: example/database/rcsb/ (logs and sample files from each module).
# Run from project root: ./scripts/examples/tools/database/test_rcsb.sh

OUT_DIR="example/database/rcsb"
mkdir -p "$OUT_DIR"

echo "=== rcsb_operations (query_rcsb_*, download_rcsb_*) ==="
python -m mcp_tools.tools.database.rcsb.rcsb_operations --test 2>&1 | tee "${OUT_DIR}/test_operations.log"

echo "All RCSB tool tests finished. Output under example/database/rcsb"
