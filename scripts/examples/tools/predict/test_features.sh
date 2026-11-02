#!/usr/bin/env bash
set -euo pipefail
VF_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd -- "$VF_REPO_ROOT"
export PYTHONPATH="$VF_REPO_ROOT:$VF_REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

# Run features_operations tests (calculate_physchem, rsa, sasa, ss, all_properties).
# Output: example/tools/predict/features/ (logs and sample JSONs).
# Run from project root: bash scripts/examples/tools/predict/test_features.sh

OUT_DIR="example/tools/predict/features"
mkdir -p "$OUT_DIR"

echo "=== features_operations (calculate_xxx) ==="
python -m mcp_tools.tools.predict.features.features_operations --test "$@" 2>&1 | tee "$OUT_DIR/test_features_operations.log"

echo ""
echo "Features operations tests finished. Output under $OUT_DIR"
