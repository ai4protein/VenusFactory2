#!/usr/bin/env bash
set -euo pipefail
VF_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd -- "$VF_REPO_ROOT"
export PYTHONPATH="$VF_REPO_ROOT:$VF_REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

# Run tests for all NCBI tools (non-helper functions in ncbi/*.py).
# Output: example/database/ncbi/ (logs and sample files from each module).
# Run from project root: ./scripts/examples/tools/database/test_ncbi.sh

OUT_DIR="example/database/ncbi"
mkdir -p "$OUT_DIR"

echo "=== ncbi_operations (query_ncbi_meta, download_ncbi_meta, query_ncbi_seq, download_ncbi_seq, download_ncbi_fasta, query_ncbi_blast, download_ncbi_blast, parse_blast_xml, build_clinvar_term, clinvar_esearch, get_clinvar_summary, get_clinvar_ftp_url, ...) ==="
python -m mcp_tools.tools.database.ncbi.ncbi_operations --test 2>&1 | tee "${OUT_DIR}/test_operations.log"

echo ""
echo "All NCBI tool tests finished. Output under example/database/ncbi"
