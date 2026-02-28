#!/usr/bin/env bash
python src/tools/search/database/rcsb/rcsb_metadata.py \
    --pdb_id_file example/database/rcsb/rcsb.txt \
    --out_dir example/database/rcsb/metadata \
    --error_file example/database/rcsb/error_download_rcsb.csv
