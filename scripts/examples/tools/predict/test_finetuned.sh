#!/usr/bin/env bash
set -euo pipefail
VF_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd -- "$VF_REPO_ROOT"
export PYTHONPATH="$VF_REPO_ROOT:$VF_REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

# Run finetuned operations tests (predict_protein_function only).
# Output: example/tools/predict/finetuned/ (logs and sample JSON).
# For a real run, pass --fasta_file and --task (optional --adapter_path; else ckpt/{dataset}/{model}).
# Run from project root: bash scripts/examples/tools/predict/test_finetuned.sh
# Example: bash scripts/examples/tools/predict/test_finetuned.sh --fasta_file /path/to/seq.fasta --task "Optimal Temperature" --model_name Ankh-large

OUT_DIR="example/predict/finetuned"
mkdir -p "$OUT_DIR"

echo "=== fintuned_operations (predict_protein_function) ==="
python -m mcp_tools.models.factory.prediction.fintuned_operations --test "$@" 2>&1 | tee "$OUT_DIR/test_finetuned_operations.log"

echo ""
echo "Finetuned operations tests finished. Output under example/predict/finetuned"
