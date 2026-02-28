#!/usr/bin/env bash
python src/tools/search/database/uniprot/uniprot_metadata.py \
    -f example/database/uniprot/uniprot.txt \
    -o example/database/uniprot/metadata \
    -e example/database/uniprot/error_download_uniprot.csv