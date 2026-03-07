#!/usr/bin/env bash
# Run tests for all RCSB PDB tools (non-helper functions in rcsb/*.py).
# Output: example/database/rcsb/ (logs and sample files from each module).
# Run from project root: ./script/tools/database/test_rcsb.sh

OUT_DIR="example/database/rcsb"
mkdir -p "$OUT_DIR"

echo "=== rcsb_operations (query_rcsb_*, download_rcsb_*) ==="
python src/tools/database/rcsb/rcsb_operations.py --test 2>&1 | tee "${OUT_DIR}/test_operations.log"

echo "All RCSB tool tests finished. Output under example/database/rcsb"
