### Dataset
# ESMFold & AlphaFold2: DeepLocBinary DeepLocMulti MetalIonBinding EC Thermostability
# ESMFold: DeepSol DeepSoluE
# No structure: FLIP_AAV FLIP_GB1

### Protein Language Model (PLM)
# facebook: esm2_t30_150M_UR50D esm2_t33_650M_UR50D esm2_t36_3B_UR50D
# rostLab: prot_bert prot_bert_bfd prot_t5_xl_uniref50 prot_t5_xl_bfd ankh-base ankh-large

# ESM model target_modules name: query key value
# T5_base model target_modules name: q k v
# Bert_base model target_modules name: query key value


dataset=DeepLocBinary
pdb_type=ESMFold
pooling_head=mean
plm_model=esm2_650m
lr=5e-4
CUDA_VISIBLE_DEVICES=0 python lora_train.py \
    --plm_model ckpt/$plm_model \
    --num_attention_heads 8 \
    --pooling_method $pooling_head \
    --pooling_dropout 0.1 \
    --dataset_config dataset/$dataset/"$dataset"_"$pdb_type".json \
    --lr $lr \
    --num_workers 4 \
    --gradient_accumulation_steps 8 \
    --max_train_epochs 100 \
    --max_batch_token 2000 \
    --patience 10 \
    --structure_seqs foldseek_seq,ss8_seq \
    --ckpt_root result \
    --ckpt_dir adapter_debug_lora/$plm_model/$dataset \
    --model_name "$pdb_type"_"$pooling_head"_"$plm_model"_"$lr".pt \
    --use_lora \
    --lora_target_modules query key value
    
