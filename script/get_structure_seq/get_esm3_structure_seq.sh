dataset=deeploc
pdb_type=alphafold
CUDA_VISIBLE_DEVICES=0 python src/data/get_esm3_structure_seq.py \
    --pdb_dir data/raw/$dataset/"$pdb_type"_pdb \
    --out_file data/raw/$dataset/"$pdb_type"_esm3.json