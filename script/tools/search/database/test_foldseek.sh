#!/usr/bin/env bash
# Run FoldSeek operations tests (submit_foldseek_*, query_foldseek_*, download_foldseek_* from foldseek_operations).
# Output: example/database/foldseek/ (search/, logs).
# Run from project root: bash script/tools/search/database/test_foldseek.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../../../../" 2>/dev/null && pwd)"
if [ -z "$ROOT" ] || [ "$ROOT" = "/" ] || [ ! -d "${ROOT}/src" ]; then
  ROOT="$(pwd)"
fi
cd "$ROOT"

OUT_DIR="example/database/foldseek"
mkdir -p "$OUT_DIR"

echo "=== foldseek_operations (submit_foldseek_*, query_foldseek_*, download_foldseek_*) ==="
python src/tools/search/database/foldseek/foldseek_operations.py --test 2>&1 | tee "${OUT_DIR}/test_foldseek_operations.log"

echo ""
echo "FoldSeek operations tests finished. Output under example/database/foldseek"
