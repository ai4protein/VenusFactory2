#!/usr/bin/env bash
set -euo pipefail
VF_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
cd -- "$VF_REPO_ROOT"
export PYTHONPATH="$VF_REPO_ROOT:$VF_REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

python -m mcp_tools.tools.file.converter.maxit \
    --file example/file/convert/cas12i3.cif \
    --out_dir example/file/convert \
    --strategy cif2pdb
