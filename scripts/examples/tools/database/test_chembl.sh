#!/usr/bin/env bash
set -euo pipefail
VF_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd -- "$VF_REPO_ROOT"
export PYTHONPATH="$VF_REPO_ROOT:$VF_REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

# Run ChEMBL operations tests (query_* and download_* from chembl_operations).
# Output: example/database/chembl/ (JSON samples and log).
# Run from project root: bash scripts/examples/tools/database/test_chembl.sh
OUT_DIR="example/database/chembl"
mkdir -p "$OUT_DIR"

echo "=== chembl_operations (query_chembl_* and download_chembl_*) ==="
python -m mcp_tools.tools.database.chembl.chembl_operations --test 2>&1 | tee "${OUT_DIR}/test_chembl_operations.log"

echo ""
echo "ChEMBL operations tests finished. Output under example/database/chembl"
