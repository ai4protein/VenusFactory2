dataset=yeast_ppi
CUDA_VISIBLE_DEVICES=0 python src/esmfold.py \
    --fasta_file data/$dataset/yeast_ppi.fasta \
    --out_dir data/$dataset/esmfold_pdb \
    --fold_chunk_size 64