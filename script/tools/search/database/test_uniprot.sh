#!/usr/bin/env bash
# Run tests for all UniProt tools (non-helper functions in uniprot/*.py).
# Output: example/database/uniprot/ (logs and sample files from each module).
# Run from project root: ./script/tools/search/database/test_uniprot.sh

OUT_DIR="example/database/uniprot"
mkdir -p "$OUT_DIR"

echo "=== uniprot_operations ==="
python src/tools/search/database/uniprot/uniprot_operations.py --test 2>&1 | tee "${OUT_DIR}/test_uniprot_operations.log"

echo ""
echo "All UniProt operations tests finished. Output under example/database/uniprot"
