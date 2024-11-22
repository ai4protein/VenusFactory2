data_name=MetalIonBinding
data_type=alphafold_pdb_noise_0.5
python src/data/get_ss_seq.py \
    --pdb_dir data/raw/$data_name/$data_type \
    --num_worker 6 \
    --out_file data/raw/$data_name/"$data_type"_ss.json