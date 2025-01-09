base_dir=/home/tanyang/data/colabfold/provaccine
python src/crawler/structure/download_alphafold.py \
    -f $base_dir/g2_uids.txt \
    -o $base_dir/af_pdb_g2 \
    -e $base_dir/error_download_g2.csv \
    -i 0