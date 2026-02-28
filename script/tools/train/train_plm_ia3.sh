### Dataset
### Protein Language Model (PLM)
# facebook: esm2_t30_150M_UR50D esm2_t33_650M_UR50D esm2_t36_3B_UR50D
# rostLab: prot_bert prot_bert_bfd prot_t5_xl_uniref50 prot_t5_xl_bfd ankh-base ankh-large

# ESM model target_modules name: query key value
# Bert_base(prot_bert) model target_modules name: query key value
# T5_base(ankh, t5) model target_modules name: q k v

# if need to use HF mirror
export HF_ENDPOINT=https://hf-mirror.com
dataset=GO_BP
pdb_type=ESMFold
pooling_head=mean
plm_source=Rostlab
plm_model=prot_t5_xl_half_uniref50-enc
lr=5e-4
training_method=plm-ia3
python src/train.py \
    --plm_model $plm_source/$plm_model \
    --dataset_config data/$dataset/"$dataset"_"$pdb_type"_HF.json \
    --learning_rate $lr \
    --gradient_accumulation_steps 8 \
    --num_epochs 100 \
    --batch_token 12000 \
    --patience 10 \
    --output_dir debug/$dataset/$plm_model \
    --output_model_name "$training_method"_"$pdb_type"_lr"$lr"_bt12k_ga8.pt \
    --training_method $training_method \
    --lora_target_modules q k v wo \
    --feedforward_modules wo
