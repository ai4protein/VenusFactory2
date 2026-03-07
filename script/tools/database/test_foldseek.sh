#!/usr/bin/env bash
# Run FoldSeek operations tests (submit_foldseek_*, query_foldseek_*, download_foldseek_* from foldseek_operations).
# Output: example/database/foldseek/ (search/, logs).
# Run from project root: bash script/tools/database/test_foldseek.sh

OUT_DIR="example/database/foldseek"
mkdir -p "$OUT_DIR"

echo "=== foldseek_operations (submit_foldseek_*, query_foldseek_*, download_foldseek_*) ==="
python src/tools/database/foldseek/foldseek_operations.py --test 2>&1 | tee "${OUT_DIR}/test_foldseek_operations.log"

echo ""
echo "FoldSeek operations tests finished. Output under example/database/foldseek"
