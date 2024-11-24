base_dir=data/uricase_search
python src/crawler/structure/download_rcsb.py \
    --pdb_file $base_dir/human_source_enzyme_homodimer_monomer_uniprot.txt \
    --out_dir $base_dir/pdb \
    --type pdb \
    --error_file $base_dir/pdb_error.csv \
    --unzip
