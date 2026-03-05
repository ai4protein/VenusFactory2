#!/usr/bin/env bash
# Run InterPro operations tests (query_interpro_*, download_interpro_* from interpro_operations).
# Output: example/database/interpro/ (metadata/, proteins/, logs).
# Run from project root: bash script/tools/search/database/test_interpro.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../../../../" 2>/dev/null && pwd)"
if [ -z "$ROOT" ] || [ "$ROOT" = "/" ] || [ ! -d "${ROOT}/src" ]; then
  ROOT="$(pwd)"
fi
cd "$ROOT"

OUT_DIR="example/database/interpro"
mkdir -p "$OUT_DIR"

echo "=== interpro_operations (query_interpro_*, download_interpro_*) ==="
python src/tools/search/database/interpro/interpro_operations.py --test 2>&1 | tee "${OUT_DIR}/test_interpro_operations.log"

echo ""
echo "InterPro operations tests finished. Output under example/database/interpro"
