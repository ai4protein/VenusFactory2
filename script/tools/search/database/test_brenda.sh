#!/usr/bin/env bash
# Run BRENDA operations tests (query_* and download_* only).
# Output: example/database/brenda/ (logs and sample files).
# Requires: BRENDA_EMAIL and BRENDA_PASSWORD in env or .env for API calls.
# Run from project root: bash script/tools/search/database/test_brenda.sh

# Use relative path so tee/mkdir work even when ROOT was fallback
OUT_DIR="example/database/brenda"
mkdir -p "$OUT_DIR"

echo "=== brenda_operations (query_* and download_*) ==="
python src/tools/search/database/brenda/brenda_operations.py --test 2>&1 | tee "$OUT_DIR/test_brenda_operations.log"

echo ""
echo "BRENDA operations tests finished. Output under example/database/brenda"
