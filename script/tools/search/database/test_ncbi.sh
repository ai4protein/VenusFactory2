#!/usr/bin/env bash
# Run tests for all NCBI tools (non-helper functions in ncbi/*.py).
# Output: example/database/ncbi/ (logs and sample files from each module).
# Run from project root: ./script/tools/search/database/test_ncbi.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../../../../" && pwd)"
cd "$ROOT"

OUT_DIR="${ROOT}/example/database/ncbi"
mkdir -p "$OUT_DIR"

echo "=== ncbi_metadata (query_ncbi_meta, download_ncbi_meta) ==="
python src/tools/search/database/ncbi/ncbi_metadata.py --test 2>&1 | tee "${OUT_DIR}/test_metadata.log"

echo ""
echo "=== ncbi_sequence (query_ncbi_seq, download_ncbi_seq, download_ncbi_fasta) ==="
python src/tools/search/database/ncbi/ncbi_sequence.py --test 2>&1 | tee "${OUT_DIR}/test_sequence.log"

echo ""
echo "=== ncbi_blast (query_ncbi_blast, download_ncbi_blast, parse_blast_xml) ==="
python src/tools/search/database/ncbi/ncbi_blast.py --test 2>&1 | tee "${OUT_DIR}/test_blast.log"

echo ""
echo "=== ncbi_clinvar (build_clinvar_term, clinvar_esearch, get_clinvar_summary, get_clinvar_ftp_url, ...) ==="
python src/tools/search/database/ncbi/ncbi_clinvar.py --test 2>&1 | tee "${OUT_DIR}/test_clinvar.log"

echo ""
echo "=== batch_gene_lookup (batch_esearch, batch_esummary, batch_lookup_by_ids, batch_lookup_by_symbols) ==="
python src/tools/search/database/ncbi/batch_gene_lookup.py --test 2>&1 | tee "${OUT_DIR}/test_batch_gene_lookup.log"

echo ""
echo "=== fetch_gene_data (get_taxon_id, fetch_gene_by_id, fetch_gene_by_symbol, fetch_multiple_genes) ==="
python src/tools/search/database/ncbi/fetch_gene_data.py --test 2>&1 | tee "${OUT_DIR}/test_fetch_gene_data.log"

echo ""
echo "=== query_gene (esearch, esummary, efetch, search_and_summarize, fetch_by_id) ==="
python src/tools/search/database/ncbi/query_gene.py --test 2>&1 | tee "${OUT_DIR}/test_query_gene.log"

echo ""
echo "All NCBI tool tests finished. Output under example/database/ncbi"
