#!/usr/bin/env bash
# Run tests for all KEGG tools (query_* and download_* in kegg_operations).
# Output: example/database/kegg/ (logs and sample files).
# Academic use only. Run from project root: ./script/tools/search/database/test_kegg.sh

OUT_DIR="example/database/kegg"
mkdir -p "$OUT_DIR"

echo "=== kegg_operations (query_kegg_info, download_kegg_info, query_kegg_list, download_kegg_list, query_kegg_find, download_kegg_find, query_kegg_get, download_kegg_get, query_kegg_conv, download_kegg_conv, query_kegg_link, download_kegg_link, query_kegg_ddi, download_kegg_ddi) ==="
python src/tools/search/database/kegg/kegg_operations.py --test 2>&1 | tee "${OUT_DIR}/test_kegg_operations.log"

echo ""
echo "All KEGG tool tests finished. Output under example/database/kegg"
