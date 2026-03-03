#!/usr/bin/env bash
# Run tests for all InterPro tools (non-helper functions in interpro/*.py).
# Output: example/database/interpro/ (logs and sample files from each module).
# Run from project root: ./script/tools/search/database/test_interpro.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../../../../" 2>/dev/null && pwd)"
if [ -z "$ROOT" ] || [ "$ROOT" = "/" ] || [ ! -d "${ROOT}/src" ]; then
  ROOT="$(pwd)"
fi
cd "$ROOT"

OUT_DIR="${ROOT}/example/database/interpro"
mkdir -p "$OUT_DIR"

echo "=== interpro_metadata (query_interpro_metadata, download_interpro_metadata) ==="
python src/tools/search/database/interpro/interpro_metadata.py --test 2>&1 | tee "${OUT_DIR}/test_metadata.log"

echo ""
echo "=== interpro_proteins (query_interpro_by_uniprot, download_interpro_by_uniprot, query_interpro_proteins, download_interpro_proteins, query_interpro_uniprot_list, download_interpro_uniprot_list) ==="
python src/tools/search/database/interpro/interpro_proteins.py --test 2>&1 | tee "${OUT_DIR}/test_proteins.log"

echo ""
echo "All InterPro tool tests finished. Output under example/database/interpro"
