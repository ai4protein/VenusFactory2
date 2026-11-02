#!/usr/bin/env bash
set -euo pipefail
VF_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd -- "$VF_REPO_ROOT"
export PYTHONPATH="$VF_REPO_ROOT:$VF_REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

# Run tests for all UniProt tools (non-helper functions in uniprot/*.py).
# Output: example/database/uniprot/ (logs and sample files from each module).
# Run from project root: ./scripts/examples/tools/database/test_uniprot.sh

OUT_DIR="example/database/uniprot"
mkdir -p "$OUT_DIR"

echo "=== uniprot_operations ==="
python -m mcp_tools.tools.database.uniprot.uniprot_operations --test 2>&1 | tee "${OUT_DIR}/test_uniprot_operations.log"

echo ""
echo "All UniProt operations tests finished. Output under example/database/uniprot"
