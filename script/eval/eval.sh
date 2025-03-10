export HF_ENDPOINT=https://hf-mirror.com
dataset=DeepLocBinary
pdb_type=AlphaFold2
plm_model=esm2_t6_8M_UR50D
model_path=ckpt/dev_models/DLB_AF2_ESM2_8M_SES.pt
python src/eval.py \
    --eval_method ses-adapter \
    --plm_model facebook/$plm_model \
    --problem_type single_label_classification \
    --test_file tyang816/DeepLocBinary_AlphaFold2 \
    --dataset $dataset \
    --model_path $model_path \
    --batch_token 12000 \
    --structure_seq foldseek_seq,ss8_seq \
    --metrics accuracy,auroc,f1,precision,recall \
    --test_result_dir result/$dataset/$plm_model/$pdb_type