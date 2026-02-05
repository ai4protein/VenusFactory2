### Structure PLM (ProSST) Training
# ProSST: structure-aware protein language model, requires stru_token_{vocab} column
# If HF dataset lacks stru_token, provide --pdb_dir to auto-generate and merge

### ProSST models
# AI4Protein/ProSST-20 ProSST-128 ProSST-512 ProSST-1024 ProSST-2048 ProSST-4096

### Dataset
# HF dataset must have aa_seq + stru_token_{vocab}, or provide pdb_dir for auto-generation
# e.g. DeepET_Topt_AlphaFold2, DeepLocBinary (with pdb_dir)

# if need HF mirror
# export HF_ENDPOINT=https://hf-mirror.com

# EXAMPLE 1: DeepET_Topt for peotein level prediction
dataset=DeepET_Topt
plm_model=AI4Protein/ProSST-2048
lr=5e-4
# PDB dir for generating stru_token when HF dataset lacks it (required for ProSST)
pdb_dir=dataset/DeepET/alphafold_pdb

CUDA_VISIBLE_DEVICES=3 python src/train.py \
    --plm_model $plm_model \
    --dataset_config data/DeepET_Topt/${dataset}_HF.json \
    --pdb_dir $pdb_dir \
    --learning_rate $lr \
    --gradient_accumulation_steps 8 \
    --batch_token 12000 \
    --output_dir ckpt/debug/$dataset/ProSST-2048 \
    --output_model_name prosst_vanilla_bt2k_ga8.pt \
    --quick_test


# EXAMPLE 2: VenusX_Res_Act_MP30 for residue level prediction
dataset=VenusX_Res_Act_MP30
plm_model=AI4Protein/ProSST-2048
lr=5e-4
# PDB dir for generating stru_token when HF dataset lacks it (required for ProSST)
pdb_dir=dataset/VenusX/VenusX_Act_AlphaFold2_PDB

CUDA_VISIBLE_DEVICES=3 python src/train.py \
    --plm_model $plm_model \
    --dataset_config data/VenusX/${dataset}_HF.json \
    --pdb_dir $pdb_dir \
    --learning_rate $lr \
    --gradient_accumulation_steps 8 \
    --batch_token 4000 \
    --output_dir ckpt/debug/$dataset/ProSST-2048 \
    --output_model_name prosst_vanilla_bt4k_ga8.pt \
    --quick_test
