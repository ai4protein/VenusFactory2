#!/usr/bin/env bash
set -euo pipefail
VF_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd -- "$VF_REPO_ROOT"
export PYTHONPATH="$VF_REPO_ROOT:$VF_REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

# Run BRENDA operations tests (query_* and download_* only).
# Output: example/database/brenda/ (logs and sample files).
# Requires: BRENDA_EMAIL and BRENDA_PASSWORD in env or .env for API calls.
# Run from project root: bash scripts/examples/tools/database/test_brenda.sh

# Use relative path so tee/mkdir work even when ROOT was fallback
OUT_DIR="example/database/brenda"
mkdir -p "$OUT_DIR"

echo "=== brenda_operations (query_* and download_*) ==="
python -m mcp_tools.tools.database.brenda.brenda_operations --test 2>&1 | tee "$OUT_DIR/test_brenda_operations.log"

echo ""
echo "BRENDA operations tests finished. Output under example/database/brenda"
