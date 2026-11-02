#!/usr/bin/env bash
set -euo pipefail
VF_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd -- "$VF_REPO_ROOT"
export PYTHONPATH="$VF_REPO_ROOT:$VF_REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

# Run InterPro operations tests (query_interpro_*, download_interpro_* from interpro_operations).
# Output: example/database/interpro/ (metadata/, proteins/, logs).
# Run from project root: bash scripts/examples/tools/database/test_interpro.sh

OUT_DIR="example/database/interpro"
mkdir -p "$OUT_DIR"

echo "=== interpro_operations (query_interpro_*, download_interpro_*) ==="
python -m mcp_tools.tools.database.interpro.interpro_operations --test 2>&1 | tee "${OUT_DIR}/test_interpro_operations.log"

echo ""
echo "InterPro operations tests finished. Output under example/database/interpro"
