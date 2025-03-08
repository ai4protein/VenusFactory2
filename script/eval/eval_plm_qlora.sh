### Dataset
# ESMFold & AlphaFold2: DeepLocBinary DeepLocMulti MetalIonBinding EC Thermostability
# ESMFold: DeepSol DeepSoluE
# No structure: FLIP_AAV FLIP_GB1

### Protein Language Model (PLM)
# facebook: esm2_t30_150M_UR50D esm2_t33_650M_UR50D esm2_t36_3B_UR50D
# rostLab: prot_bert prot_bert_bfd prot_t5_xl_uniref50 prot_t5_xl_bfd ankh-base ankh-large

# ESM model target_modules name: query key value
# Bert_base(prot_bert) model target_modules name: query key value
# T5_base(ankh, t5) model target_modules name: q k v


export HF_ENDPOINT=https://hf-mirror.com
dataset=eSOL
pdb_type=ESMFold
pooling_method=mean
plm_source=model_ckpt
plm_model=prot_t5_xl
output_root=ckpt
lr=5e-4
eval_method=plm-qlora
test_result_dir=ckpt/debug_test/eSOL/test_res
problem_type=regression
num_labels=1
#
python src/eval.py \
    --plm_model $plm_source/$plm_model \
    --dataset_config data/$dataset/"$dataset"_"$pdb_type"_HF.json \
    --batch_token 12000 \
    --output_root $output_root \
    --output_dir debug_test/$dataset/$plm_model \
    --output_model_name "$eval_method"_ESMFold_lr5e-4_bt12k_ga8.pt \
    --eval_method $eval_method \
    --test_result_dir $test_result_dir \
    --problem_type $problem_type \
    --metrics spearman_corr \
    --pooling_method $pooling_method \
    --num_labels $num_labels