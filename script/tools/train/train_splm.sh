### Structure PLM (ProSST) Training
# ProSST: structure-aware protein language model, requires stru_token_{vocab} column
# If HF dataset lacks stru_token, provide --pdb_dir to auto-generate and merge

### ProSST models
# AI4Protein/ProSST-20 ProSST-128 ProSST-512 ProSST-1024 ProSST-2048 ProSST-4096
### SaProt models
# westlake-repl/SaProt_650M_AF2 westlake-repl/SaProt_650M_PDB westlake-repl/SaProt_35M_AF2

### Dataset
# HF dataset must have aa_seq + stru_token_{vocab}, or provide pdb_dir for auto-generation
# e.g. DeepET_Topt_AlphaFold2, DeepLocBinary (with pdb_dir)

# if need HF mirror
# export HF_ENDPOINT=https://hf-mirror.com

# EXAMPLE 1: DeepET_Topt for peotein level prediction
dataset=DeepET_Topt
plm_model=westlake-repl/SaProt_650M_AF2
lr=5e-4
# PDB dir for generating stru_token when HF dataset lacks it
pdb_dir=dataset/DeepET/alphafold_pdb

CUDA_VISIBLE_DEVICES=0 python src/train.py \
    --plm_model $plm_model \
    --dataset_config data/DeepET_Topt/${dataset}_HF.json \
    --pdb_dir $pdb_dir \
    --learning_rate $lr \
    --gradient_accumulation_steps 8 \
    --batch_token 12000 \
    --output_dir ckpt/debug/$dataset/SaProt_650M_AF2 \
    --output_model_name saprot_vanilla_bt2k_ga8.pt \
    --quick_test


# EXAMPLE 2: VenusX_Res_Act_MP30 for residue level prediction
dataset=VenusX_Res_Act_MP30
plm_model=westlake-repl/SaProt_650M_AF2
lr=5e-4
# PDB dir for generating stru_token when HF dataset lacks it
pdb_dir=dataset/VenusX/VenusX_Act_AlphaFold2_PDB

CUDA_VISIBLE_DEVICES=0 python src/train.py \
    --plm_model $plm_model \
    --dataset_config data/VenusX/${dataset}_HF.json \
    --pdb_dir $pdb_dir \
    --learning_rate $lr \
    --gradient_accumulation_steps 8 \
    --batch_token 12000 \
    --output_dir ckpt/debug/$dataset/SaProt_650M_AF2 \
    --output_model_name saprot_vanilla_bt4k_ga8.pt \
    --quick_test


### ProtSSN Training (freeze only, no LoRA)
# ProtSSN: structure-based embedding from PDB; requires --pdb_dir to get pdb_path per sample
# Optional: --gnn_config, --gnn_model_path, --c_alpha_max_neighbors (default 10)

# EXAMPLE 3: Protein-level with ProtSSN
dataset=DeepET_Topt
plm_model=ProtSSN-k20_h512
lr=5e-4
pdb_dir=dataset/DeepET/alphafold_pdb

CUDA_VISIBLE_DEVICES=0 python src/train.py \
    --plm_model $plm_model \
    --dataset_config data/DeepET_Topt/${dataset}_HF.json \
    --pdb_dir $pdb_dir \
    --training_method freeze \
    --learning_rate $lr \
    --gradient_accumulation_steps 8 \
    --batch_token 12000 \
    --output_dir ckpt/debug/$dataset/ProtSSN-k20_h512 \
    --output_model_name protssn_k20_h512_freeze.pt \
    --quick_test

# EXAMPLE 4: Residue-level with ProtSSN (e.g. VenusX)
dataset=VenusX_Res_Act_MP30
plm_model=ProtSSN-k20_h512
lr=5e-4
pdb_dir=dataset/VenusX/VenusX_Act_AlphaFold2_PDB

CUDA_VISIBLE_DEVICES=0 python src/train.py \
    --plm_model $plm_model \
    --dataset_config data/VenusX/${dataset}_HF.json \
    --pdb_dir $pdb_dir \
    --training_method freeze \
    --learning_rate $lr \
    --gradient_accumulation_steps 8 \
    --batch_token 12000 \
    --output_dir ckpt/debug/$dataset/ProtSSN-k20_h512 \
    --output_model_name protssn_k20_h512_freeze.pt \
    --quick_test