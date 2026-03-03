#!/usr/bin/env bash
# Run tests for all AlphaFold tools (all non-helper functions in alphafold_structure + alphafold_metadata).
# Output: example/database/alphafold/ (structure, metadata, query samples, logs).
# Run from project root: ./script/tools/search/database/test_alphafold.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../../../../" && pwd)"
cd "$ROOT"

# All logs and intermediate output under example/
OUT_DIR="${ROOT}/example/database/alphafold"
mkdir -p "$OUT_DIR"

echo "=== alphafold_structure (query_alphafold_structure, download_alphafold_structure) ==="
python src/tools/search/database/alphafold/alphafold_structure.py --test 2>&1 | tee "${OUT_DIR}/test_structure.log"

echo ""
echo "=== alphafold_metadata (query_alphafold_metadata, download_alphafold_metadata) ==="
python src/tools/search/database/alphafold/alphafold_metadata.py --test 2>&1 | tee "${OUT_DIR}/test_metadata.log"

echo ""
echo "All AlphaFold tool tests finished. Output under example/database/alphafold"
