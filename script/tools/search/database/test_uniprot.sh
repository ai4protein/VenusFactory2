#!/usr/bin/env bash
# Run tests for all UniProt tools (non-helper functions in uniprot/*.py).
# Output: example/database/uniprot/ (logs and sample files from each module).
# Run from project root: ./script/tools/search/database/test_uniprot.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../../../../" && pwd)"
cd "$ROOT"

OUT_DIR="${ROOT}/example/database/uniprot"
mkdir -p "$OUT_DIR"

echo "=== uniprot_search (uniprot_search, uniprot_retrieve, uniprot_mapping, uniprot_search_and_retrieve) ==="
python src/tools/search/database/uniprot/uniprot_search.py --test 2>&1 | tee "${OUT_DIR}/test_uniprot_search.log"

echo ""
echo "=== uniprot_metadata (query_uniprot_meta, download_uniprot_meta) ==="
python src/tools/search/database/uniprot/uniprot_metadata.py --test 2>&1 | tee "${OUT_DIR}/test_metadata.log"

echo ""
echo "=== uniprot_sequence (query_uniprot_seq, download_uniprot_seq, download_uniprot_sequence) ==="
python src/tools/search/database/uniprot/uniprot_sequence.py --test 2>&1 | tee "${OUT_DIR}/test_sequence.log"

echo ""
echo "All UniProt tool tests finished. Output under example/database/uniprot"
