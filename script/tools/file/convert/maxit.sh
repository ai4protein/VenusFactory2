#!/usr/bin/env bash
python src/tools/file/converter/maxit.py \
    --file example/file/convert/cas12i3.cif \
    --out_dir example/file/convert \
    --strategy cif2pdb
