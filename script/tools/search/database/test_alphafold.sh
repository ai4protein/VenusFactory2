#!/usr/bin/env bash
# Run AlphaFold operations tests (query_* and download_* from alphafold_operations).
# Output: example/database/alphafold/ (structure, metadata, query samples, logs).
# Run from project root: bash script/tools/search/database/test_alphafold.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../../../../" 2>/dev/null && pwd)"
if [ -z "$ROOT" ] || [ "$ROOT" = "/" ] || [ ! -d "${ROOT}/src" ]; then
  ROOT="$(pwd)"
fi
cd "$ROOT"

OUT_DIR="example/database/alphafold"
mkdir -p "$OUT_DIR"

echo "=== alphafold_operations (query_* and download_*) ==="
python src/tools/search/database/alphafold/alphafold_operations.py --test 2>&1 | tee "${OUT_DIR}/test_alphafold_operations.log"

echo ""
echo "AlphaFold operations tests finished. Output under example/database/alphafold"
