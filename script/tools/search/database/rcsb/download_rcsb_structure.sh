#!/usr/bin/env bash
python src/tools/search/database/rcsb/rcsb_structure.py \
    -f example/database/rcsb/rcsb.txt \
    -o example/database/rcsb/structure \
    -t pdb \
    -e example/database/rcsb/error_download_rcsb.csv \
    -u
