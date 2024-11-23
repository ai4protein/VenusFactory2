### Dataset
# ESMFold & AlphaFold2: DeepLocBinary DeepLocMulti MetalIonBinding EC Thermostability
# ESMFold: DeepSol DeepSoluE
# No structure: FLIP_AAV FLIP_GB1

### Protein Language Model (PLM)
# facebook: esm2_t30_150M_UR50D esm2_t33_650M_UR50D esm2_t36_3B_UR50D
# rostLab: prot_bert prot_bert_bfd prot_t5_xl_uniref50 prot_t5_xl_bfd ankh-base ankh-large

# train with batch token
dataset=DeepLocBinary
pdb_type=AlphaFold2
plm_model=esm2_t30_150M_UR50D
lr=5e-4
python src/train.py \
    --plm_model facebook/$plm_model \
    --dataset_config data/$dataset/"$dataset"_"$pdb_type"_HF.json \
    --learning_rate $lr \
    --gradient_accumulation_step 1 \
    --batch_token 10000 \
    --ckpt_dir debug/$dataset/$plm_model \
    --model_name af2_lr"$lr"_bt10k.pt

# train with batch_size
dataset=DeepLocBinary
pdb_type=AlphaFold2
plm_model=esm2_t30_150M_UR50D
lr=5e-4
python src/train.py \
    --plm_model facebook/$plm_model \
    --dataset_config data/$dataset/"$dataset"_"$pdb_type"_HF.json \
    --learning_rate $lr \
    --gradient_accumulation_step 8 \
    --batch_size 12 \
    --max_seq_len 1024 \
    --ckpt_dir debug/$dataset/$plm_model \
    --model_name af2_lr"$lr"_bs12_ga8.pt