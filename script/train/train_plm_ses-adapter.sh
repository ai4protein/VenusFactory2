### Dataset
# ESMFold & AlphaFold2: DeepLocBinary DeepLocMulti MetalIonBinding EC Thermostability
# ESMFold: DeepSol DeepSoluE
# No structure: FLIP_AAV FLIP_GB1

### Protein Language Model (PLM)
# facebook: esm2_t30_150M_UR50D esm2_t33_650M_UR50D esm2_t36_3B_UR50D
# rostLab: prot_bert prot_bert_bfd prot_t5_xl_uniref50 prot_t5_xl_bfd ankh-base ankh-large

dataset=DeepLocBinary
pdb_type=AlphaFold2
plm_model=esm2_t30_150M_UR50D
lr=5e-4
python train.py \
    --plm_model facebook/$plm_model \
    --dataset_config data/$dataset/"$dataset"_"$pdb_type"_HF.json \
    --lr $lr \
    --train_epoch 50 \
    --batch_token 60000 \
    --patience 3 \
    --structure_seq foldseek_seq,ss8_seq \
    --ckpt_root result \
    --ckpt_dir debug/$dataset/$plm_model \
    --model_name "$pdb_type"_"$lr".pt
