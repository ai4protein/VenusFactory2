#!/usr/bin/env bash
# Run tests for all NCBI tools (non-helper functions in ncbi/*.py).
# Output: example/database/ncbi/ (logs and sample files from each module).
# Run from project root: ./script/tools/database/test_ncbi.sh

OUT_DIR="example/database/ncbi"
mkdir -p "$OUT_DIR"

echo "=== ncbi_operations (query_ncbi_meta, download_ncbi_meta, query_ncbi_seq, download_ncbi_seq, download_ncbi_fasta, query_ncbi_blast, download_ncbi_blast, parse_blast_xml, build_clinvar_term, clinvar_esearch, get_clinvar_summary, get_clinvar_ftp_url, ...) ==="
python src/tools/database/ncbi/ncbi_operations.py --test 2>&1 | tee "${OUT_DIR}/test_operations.log"

echo ""
echo "All NCBI tool tests finished. Output under example/database/ncbi"
