#!/usr/bin/env bash
set -euo pipefail
VF_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
cd -- "$VF_REPO_ROOT"
export PYTHONPATH="$VF_REPO_ROOT:$VF_REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

python -m mcp_tools.models.factory.training.data.embed.ems2 \
    --model_name_or_path facebook/esm2_t33_650M_UR50D \
    --batch_size 32 \
    --fasta_file_path download/uniprot_sequences/merged.fasta \
    --pooling mean \
    --chunk_num 1 \
    --chunk_id 0 \
    --output_dir dataset/database/embed
