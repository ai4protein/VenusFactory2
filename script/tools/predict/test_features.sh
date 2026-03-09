#!/usr/bin/env bash
# Run features_operations tests (calculate_physchem, rsa, sasa, ss, all_properties).
# Output: example/tools/predict/features/ (logs and sample JSONs).
# Run from project root: bash script/tools/predict/test_features.sh

OUT_DIR="example/tools/predict/features"
mkdir -p "$OUT_DIR"

echo "=== features_operations (calculate_xxx) ==="
python src/tools/predict/features/features_operations.py --test "$@" 2>&1 | tee "$OUT_DIR/test_features_operations.log"

echo ""
echo "Features operations tests finished. Output under $OUT_DIR"
