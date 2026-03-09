#!/usr/bin/env bash
# Run finetuned operations tests (predict_protein_function only).
# Output: example/tools/predict/finetuned/ (logs and sample JSON).
# For a real run, pass --fasta_file and --task (optional --adapter_path; else ckpt/{dataset}/{model}).
# Run from project root: bash script/tools/predict/test_finetuned.sh
# Example: bash script/tools/predict/test_finetuned.sh --fasta_file /path/to/seq.fasta --task "Optimal Temperature" --model_name Ankh-large

OUT_DIR="example/predict/finetuned"
mkdir -p "$OUT_DIR"

echo "=== fintuned_operations (predict_protein_function) ==="
python src/tools/predict/finetuned/fintuned_operations.py --test "$@" 2>&1 | tee "$OUT_DIR/test_finetuned_operations.log"

echo ""
echo "Finetuned operations tests finished. Output under example/predict/finetuned"
