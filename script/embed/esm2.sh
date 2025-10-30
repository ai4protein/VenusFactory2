python src/data/embed/ems2.py \
    --model_name_or_path facebook/esm2_t33_650M_UR50D \
    --batch_size 32 \
    --fasta_file_path download/uniprot_sequences/merged.fasta \
    --pooling mean \
    --chunk_num 1 \
    --chunk_id 0 \
    --output_dir dataset/database/embed