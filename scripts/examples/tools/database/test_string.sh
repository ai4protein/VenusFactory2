#!/usr/bin/env bash
set -euo pipefail
VF_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd -- "$VF_REPO_ROOT"
export PYTHONPATH="$VF_REPO_ROOT:$VF_REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

# Run tests for all STRING tools (query_* and download_* in string_operations).
# Output: example/database/string/ (logs and sample files).
# Run from project root: ./scripts/examples/tools/database/test_string.sh

OUT_DIR="example/database/string"
mkdir -p "$OUT_DIR"

echo "=== string_operations (query_string_version, download_string_version, query_string_map_ids, download_string_map_ids, query_string_network, download_string_network, query_string_network_image, download_string_network_image, query_string_interaction_partners, download_string_interaction_partners, query_string_enrichment, download_string_enrichment, query_string_ppi_enrichment, download_string_ppi_enrichment, query_string_homology, download_string_homology) ==="
python -m mcp_tools.tools.database.string.string_operations --test 2>&1 | tee "${OUT_DIR}/test_string_operations.log"

echo ""
echo "All STRING tool tests finished. Output under example/database/string"
