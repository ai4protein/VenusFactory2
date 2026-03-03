#!/usr/bin/env bash
# Run tests for all RCSB PDB tools (non-helper functions in rcsb/*.py).
# Output: example/database/rcsb/ (logs and sample files from each module).
# Run from project root: ./script/tools/search/database/test_rcsb.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../../../../" 2>/dev/null && pwd)"
if [ -z "$ROOT" ] || [ "$ROOT" = "/" ] || [ ! -d "${ROOT}/src" ]; then
  ROOT="$(pwd)"
fi
cd "$ROOT"

OUT_DIR="${ROOT}/example/database/rcsb"
mkdir -p "$OUT_DIR"

echo "=== rcsb_metadata (query_rcsb_entry, download_rcsb_entry) ==="
python src/tools/search/database/rcsb/rcsb_metadata.py --test 2>&1 | tee "${OUT_DIR}/test_metadata.log"

echo ""
echo "=== rcsb_structure (query_rcsb_structure, download_rcsb_structure) ==="
python src/tools/search/database/rcsb/rcsb_structure.py --test 2>&1 | tee "${OUT_DIR}/test_structure.log"

echo ""
echo "All RCSB tool tests finished. Output under example/database/rcsb"
