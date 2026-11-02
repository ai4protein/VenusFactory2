#!/usr/bin/env bash
set -euo pipefail
VF_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd -- "$VF_REPO_ROOT"
export PYTHONPATH="$VF_REPO_ROOT:$VF_REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

# Run FoldSeek operations tests (submit_foldseek_*, query_foldseek_*, download_foldseek_* from foldseek_operations).
# Output: example/database/foldseek/ (search/, logs).
# Run from project root: bash scripts/examples/tools/database/test_foldseek.sh

OUT_DIR="example/database/foldseek"
mkdir -p "$OUT_DIR"

echo "=== foldseek_operations (submit_foldseek_*, query_foldseek_*, download_foldseek_*) ==="
python -m mcp_tools.tools.database.foldseek.foldseek_operations --test 2>&1 | tee "${OUT_DIR}/test_foldseek_operations.log"

echo ""
echo "FoldSeek operations tests finished. Output under example/database/foldseek"
