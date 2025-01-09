base_dir=data/interpro_motif/IPR000048
python src/crawler/sequence/download_uniprot_seq.py \
    -f $base_dir/failed.txt \
    -o $base_dir/unfold_fasta

base_dir=data/interpro_domain
python src/crawler/sequence/download_uniprot_seq.py \
    -f $base_dir/unfolded_proteins.txt \
    -o $base_dir/unfold_fasta

base_dir=data/interpro_motif
for dir in "$base_dir"/IPR*; do
    if [ -d "$dir" ]; then
        inter_id=$(basename "$dir")
        python src/crawler/sequence/download_uniprot_seq.py -f $base_dir/$inter_id/failed.txt -o data/motif_unfold
    fi
done
