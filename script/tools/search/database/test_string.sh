#!/usr/bin/env bash
# Run tests for all STRING tools (query_* and download_* in string_operations).
# Output: example/database/string/ (logs and sample files).
# Run from project root: ./script/tools/search/database/test_string.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../../../../" 2>/dev/null && pwd)"
if [ -z "$ROOT" ] || [ "$ROOT" = "/" ] || [ ! -d "${ROOT}/src" ]; then
  ROOT="$(pwd)"
fi
cd "$ROOT"

OUT_DIR="${ROOT}/example/database/string"
mkdir -p "$OUT_DIR"

echo "=== string_operations (query_string_version, download_string_version, query_string_map_ids, download_string_map_ids, query_string_network, download_string_network, query_string_network_image, download_string_network_image, query_string_interaction_partners, download_string_interaction_partners, query_string_enrichment, download_string_enrichment, query_string_ppi_enrichment, download_string_ppi_enrichment, query_string_homology, download_string_homology) ==="
python src/tools/search/database/string/string_operations.py --test 2>&1 | tee "${OUT_DIR}/test_string_operations.log"

echo ""
echo "All STRING tool tests finished. Output under example/database/string"
