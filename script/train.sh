# dataset (ESMFold & AlphaFold2): DeepLocBinary DeepLocMulti MetalIonBinding EC Thermostability
# dataset (ESMFold only): DeepSol DeepSoluE
# plm_model (Facebook): esm2_t30_150M_UR50D esm2_t33_650M_UR50D esm2_t36_3B_UR50D
# plm_model (RostLab): prot_bert prot_bert_bfd prot_t5_xl_uniref50 prot_t5_xl_bfd ankh-base ankh-large
dataset=Thermostability
pdb_type=AlphaFold2
pooling_head=mean
plm_model=esm2_t30_150M_UR50D
lr=5e-4
CUDA_VISIBLE_DEVICES=1 python train.py \
    --plm_model ckpt/$plm_model \
    --num_attention_heads 8 \
    --pooling_method $pooling_head \
    --pooling_dropout 0.1 \
    --dataset_config dataset/$dataset/"$dataset"_"$pdb_type"_HF.json \
    --lr $lr \
    --num_workers 4 \
    --gradient_accumulation_steps 1 \
    --max_train_epochs 50 \
    --max_batch_token 60000 \
    --patience 3 \
    --structure_seqs foldseek_seq,ss8_seq \
    --ckpt_root result \
    --ckpt_dir adapter_debug/$plm_model/$dataset \
    --model_name "$pdb_type"_"$pooling_head"_"$plm_model"_"$lr".pt \
    --wandb \
    --wandb_entity ty_ang \
    --wandb_project adapter_debug \
    --wandb_run_name "$dataset"_"$pdb_type"_"$pooling_head"_"$plm_model"_"$lr"

dataset=DeepSoluE
pdb_type=ESMFold
pooling_head=mean
plm_model=esm2_t30_150M_UR50D
lr=5e-4
CUDA_VISIBLE_DEVICES=0 python train.py \
    --plm_model ckpt/$plm_model \
    --num_attention_heads 8 \
    --pooling_method $pooling_head \
    --pooling_dropout 0.1 \
    --dataset_config dataset/$dataset/"$dataset"_"$pdb_type".json \
    --lr $lr \
    --num_workers 4 \
    --gradient_accumulation_steps 1 \
    --max_train_epochs 10 \
    --max_batch_token 60000 \
    --patience 3 \
    --structure_seqs foldseek_seq \
    --ckpt_root result \
    --ckpt_dir adapter_debug/$plm_model/$dataset \
    --model_name woss_"$pdb_type"_"$pooling_head"_"$plm_model"_"$lr".pt \
    --wandb \
    --wandb_entity ty_ang \
    --wandb_project adapter_debug \
    --wandb_run_name woss_"$dataset"_"$pdb_type"_"$pooling_head"_"$plm_model"_"$lr"

dataset=DeepSoluE
pdb_type=ESMFold
pooling_head=mean
plm_model=esm2_t30_150M_UR50D
lr=5e-4
CUDA_VISIBLE_DEVICES=0 python train.py \
    --plm_model ckpt/$plm_model \
    --num_attention_heads 8 \
    --pooling_method $pooling_head \
    --pooling_dropout 0.1 \
    --dataset_config dataset/$dataset/"$dataset"_"$pdb_type".json \
    --lr $lr \
    --num_workers 4 \
    --gradient_accumulation_steps 1 \
    --max_train_epochs 10 \
    --max_batch_token 60000 \
    --patience 3 \
    --structure_seqs ss8_seq \
    --ckpt_root result \
    --ckpt_dir adapter_debug/$plm_model/$dataset \
    --model_name wofs_"$pdb_type"_"$pooling_head"_"$plm_model"_"$lr".pt \
    --wandb \
    --wandb_entity ty_ang \
    --wandb_project adapter_debug \
    --wandb_run_name wofs_"$dataset"_"$pdb_type"_"$pooling_head"_"$plm_model"_"$lr"

dataset=DeepSoluE
pdb_type=ESMFold
pooling_head=mean
plm_model=esm2_t30_150M_UR50D
lr=5e-4
CUDA_VISIBLE_DEVICES=0 python train.py \
    --plm_model ckpt/$plm_model \
    --num_attention_heads 8 \
    --pooling_method $pooling_head \
    --pooling_dropout 0.1 \
    --dataset_config dataset/$dataset/"$dataset"_"$pdb_type".json \
    --lr $lr \
    --num_workers 4 \
    --gradient_accumulation_steps 1 \
    --max_train_epochs 10 \
    --max_batch_token 60000 \
    --patience 3 \
    --ckpt_root result \
    --ckpt_dir adapter_debug/$plm_model/$dataset \
    --model_name wofsss_"$pdb_type"_"$pooling_head"_"$plm_model"_"$lr".pt \
    --wandb \
    --wandb_entity ty_ang \
    --wandb_project adapter_debug \
    --wandb_run_name wofsss_"$dataset"_"$pdb_type"_"$pooling_head"_"$plm_model"_"$lr"