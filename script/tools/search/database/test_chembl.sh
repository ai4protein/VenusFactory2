#!/usr/bin/env bash
# Run ChEMBL operations tests (query_* and download_* from chembl_operations).
# Output: example/database/chembl/ (JSON samples and log).
# Run from project root: bash script/tools/database/test_chembl.sh
OUT_DIR="example/database/chembl"
mkdir -p "$OUT_DIR"

echo "=== chembl_operations (query_chembl_* and download_chembl_*) ==="
python src/tools/database/chembl/chembl_operations.py --test 2>&1 | tee "${OUT_DIR}/test_chembl_operations.log"

echo ""
echo "ChEMBL operations tests finished. Output under example/database/chembl"
