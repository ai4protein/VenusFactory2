#!/usr/bin/env bash
python src/tools/search/database/uniprot/uniprot_sequence.py \
    -f example/database/uniprot/uniprot.txt \
    -o example/database/uniprot/sequence \
    -e example/database/uniprot/error_download_uniprot.csv