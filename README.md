<div align="right">
  <a href="README.md">English</a> | <a href="README_CN.md">简体中文</a>
</div>

<p align="center">
  <img src="img/banner_2503.png" width="70%" alt="VenusFactory Banner">
</p>

<div align="center">

[![GitHub stars](https://img.shields.io/github/stars/AI4Protein/VenusFactory?style=flat-square)](https://github.com/AI4Protein/VenusFactory/stargazers) [![GitHub forks](https://img.shields.io/github/forks/AI4Protein/VenusFactory?style=flat-square)](https://github.com/AI4Protein/VenusFactory/network/members) [![GitHub issues](https://img.shields.io/github/issues/AI4Protein/VenusFactory?style=flat-square)](https://github.com/AI4Protein/VenusFactory/issues) [![GitHub license](https://img.shields.io/github/license/AI4Protein/VenusFactory?style=flat-square)](https://github.com/AI4Protein/VenusFactory/blob/main/LICENSE)

[![Python Version](https://img.shields.io/badge/Python-3.12-blue?style=flat-square&logo=python)](https://www.python.org/) [![Documentation](https://img.shields.io/badge/docs-latest-brightgreen?style=flat-square)](https://venusfactory.readthedocs.io/) [![Downloads](https://img.shields.io/github/downloads/AI4Protein/VenusFactory/total?style=flat-square)](https://github.com/AI4Protein/VenusFactory/releases) [![Youtube](https://img.shields.io/badge/Youtube-AI4Protein-red?style=flat-square&logo=youtube)](https://www.youtube.com/watch?v=MT6lPH5kgCc&ab_channel=BxinZhou) [![Demo](https://img.shields.io/badge/Demo-OpenBayes-blue)](https://openbayes.com/console/public/tutorials/O3RCA0XUKa0)

**🤖 Agent-Driven Protein Engineering Platform**
*One platform, three interfaces, infinite possibilities*

</div>

---

## 🌟 Recent News

- [2026-01-23] 🚀 Added [VenusX](https://github.com/AI4Protein/VenusX) in VenusFactory
- [2025-08-10] 🎉 Free website released at [venusfactory.cn/playground](https://venusfactory.cn/playground/)
- [2025-04-19] 🎉 [VenusREM](https://github.com/ai4protein/VenusREM) #1 in [ProteinGym](https://proteingym.org/benchmarks) & [VenusMutHub](https://lianglab.sjtu.edu.cn/muthub/)!

<details>
<summary>📨 Join our WeChat Group / 📝 Share Your Feedback</summary>
<p align="center">
  <img src="img/wechat.png" width="60%" alt="WeChat Group">
</p>
</details>

---

## 🎯 What is VenusFactory?

**VenusFactory** is an **Agent-driven protein engineering platform** combining 40+ AI models with 11 biological databases. Designed for everyone — from biologists to AI researchers.

<p align="center">
  <img src="https://img.shields.io/badge/🤖_Agent_Driven-Core-FF6B6B?style=for-the-badge">
  <img src="https://img.shields.io/badge/Models-40+-4ECDC4?style=for-the-badge">
  <img src="https://img.shields.io/badge/Databases-11+-95E1D3?style=for-the-badge">
  <img src="https://img.shields.io/badge/Tools-9_Categories-F38181?style=for-the-badge">
</p>

### 🚀 Why VenusFactory?

| **🤖 Agent-First** | **🎯 Three Interfaces** | **⚡ Zero to Results** |
|:------------------:|:----------------------:|:---------------------:|
| Natural language → Multi-step automation | Web UI / REST API / CLI | Upload → Predict in seconds |
| 40+ models + 11 databases | Same power, different styles | Or train custom models in minutes |

> **📖 Easy to Learn**: Designed for life science professionals with no programming background required. Intuitive Web UI, comprehensive bilingual documentation, rich examples and video tutorials help you quickly grow from beginner to protein AI expert.

### 💡 Capabilities at a Glance

| Task | Solution | Time |
|:-----|:---------|:-----|
| 🧬 Mutation effects | ESM-2, ProSST, ProtSSN (zero-shot) | <1 min |
| 🎯 Protein function | 30+ fine-tuned models | <30 sec |
| 🔬 Custom training | 7 PEFT methods (LoRA, QLoRA, etc.) | 10-60 min |
| 💾 Data download | AlphaFold, UniProt, RCSB, KEGG, etc. | Real-time |
| 📚 Literature | AI-powered search & analysis | <2 min |

---

## ⚡ Quick Start

### 1. Install
```bash
git clone https://github.com/AI4Protein/VenusFactory.git && cd VenusFactory
conda create -n venus python=3.12 && conda activate venus
pip install -r requirements.txt  # Detailed guide below ↓
```

### 2. Launch
```bash
# Web UI (Recommended)
python src/webui.py  # → http://localhost:7860

# REST API
python src/api_server.py  # → http://localhost:5000/docs

# CLI
bash script/train/train_plm_lora.sh
```

### 3. Get Results

<details>
<summary><b>🤖 Try Agent-0.1 | ⚡ Quick Tools | 🔬 Train Models</b> (Click to expand examples)</summary>

**Agent-0.1 (Natural Language)**
```
Q: "Predict stability for sequence MKTAYIAKQRQISFV..."
→ Agent auto-selects model → Runs prediction → Returns results + explanations
```

**Quick Mutation Scoring**
```
Upload: PDB/FASTA → Mutations: A23V, K45R → Get: Stability scores
```

**Train Your Model**
```
Model: ESM2-650M → Dataset: DeepSol → Method: LoRA → 15 min → Trained model ✓
```

</details>

<p align="center">
  <video width="70%" controls>
    <source src="./img/venusfactory.mp4" type="video/mp4">
  </video>
</p>

---

## 🤖 Agent-0.1: The Brain

**Agent-0.1** orchestrates all tools via natural language. Powered by LangGraph + LangChain.

```
You: "Design thermostable mutations for PDB:1ABC"
         ↓
    🤖 Agent Planning
         ↓
  📥 Download → 🧬 Predict → 🎯 Score → 📊 Report
  RCSB PDB     ESM-2 scan    Stability   Ranked list
```

<details>
<summary><b>✨ Agent Capabilities</b></summary>

| Category | Features |
|:---------|:---------|
| **🔬 Analysis** | Mutation prediction • Function/stability scoring • Structure analysis |
| **💾 Data** | Multi-database search • Format conversion • Batch processing |
| **🧠 Planning** | Multi-step automation • Tool orchestration • Error handling |
| **📚 Research** | Literature mining • Family analysis • Report generation |

</details>

<details>
<summary><b>💬 Example Conversations</b></summary>

**Mutation Design:**
```
You: "Improve thermostability of MKTAYIAKQR..."
Agent: ✓ ESM-2 scanning... ✓ Stability scoring...
→ Top 3: A5V (+2.8 kcal/mol), K9R (+1.9), T2S (+1.5)
```

**Database Search:**
```
You: "Find lysozyme structures <2.0Å resolution"
Agent: ✓ Searching RCSB... → Found 47 structures
→ Downloaded to: temp_outputs/lysozyme_structures/
```

</details>

> 💡 **Note:** Requires API key (OpenAI/Anthropic). Currently in Beta.

---

## 🏗️ Architecture

```
🌐 Interfaces: Web UI | REST API | CLI
        ↓
   🤖 Agent Layer (LangGraph + LangChain)
        ↓
   🔧 Application: Train | Eval | Predict | Tools
        ↓
   🛠️ Core Tools: 9 categories (mutation, database, search, etc.)
        ↓
   📊 Resources: 40+ Models | 30+ Datasets | 11+ Databases
```

<details>
<summary><b>📚 Integrated Resources</b></summary>

**Models (40+):** ESM, ProtBert, ProtT5, Venus/PETA/ProSST series

**Databases (11+):** AlphaFold • RCSB PDB • UniProt • NCBI • KEGG • STRING • BRENDA • ChEMBL • HPA • FDA • Foldseek

**Datasets (30+):** Function • Localization • Stability • Solubility • Mutation fitness

</details>

<details>
<summary><b>🔧 Tool Categories</b></summary>

| Tool | Description | Agent | CLI |
|:-----|:------------|:-----:|:---:|
| 🧬 Mutation | ESM-1v, ESM-2, ProSST, ProtSSN, MIF-ST | ✅ | ✅ |
| 🎯 Prediction | 30+ fine-tuned models | ✅ | ✅ |
| 💾 Database | 11 integrations | ✅ | ✅ |
| 🔍 Search | PubMed, FDA, patents | ✅ | ✅ |
| 🏋️ Training | LoRA, QLoRA, DoRA, etc. | ✅ | ✅ |
| 📁 File | Format conversion | ✅ | ✅ |
| 🔬 Denovo | Protein design | ✅ | ✅ |
| 🧪 Discovery | Novel discovery | ✅ | ✅ |
| 📊 Visualize | 3D viewer | ✅ | ✅ |

</details>

---

## 🧬 Supported Models

<details>
<summary><b>40+ Protein Language Models</b> (Click to expand)</summary>

**Venus Series (Liang's Lab):**
ProSST-20/128/512/1024/2048/4096 (110M) • ProPrime-690M • VenusPLM-300M • PETA-base/bpe/unigram (80M)

**ESM Series (Meta AI):**
ESM2: 8M, 35M, 150M, 650M, 3B, 15B • ESM-1v: 5 models (650M each)

**ProtBert & ProtT5:**
ProtBert-Uniref100/BFD (420M) • IgBert (420M) • ProtT5-XL/XXL (3B-11B) • Ankh-base/large (450M-1.2B)

**Selection Guide:**
- GPU <8GB: ESM2-8M/35M, ProSST
- GPU 8-16GB: ESM2-150M/650M, ProtBert
- GPU 24GB+: ESM2-3B, ProtT5-XL
- Multi-GPU: ESM2-15B, ProtT5-XXL

**By Task:**
- Classification: ESM2, ProtBert
- Structure: Ankh
- Generation: ProtT5
- Antibody: IgBert/IgT5
- Lightweight: ProSST, PETA

</details>

---

## 📚 Supported Datasets

<details>
<summary><b>30+ Supervised + Zero-Shot Datasets</b></summary>

**Zero-Shot:** VenusMutHub • ProteinGym (217 DMS)

**Function:** EC • GO_BP • GO_CC • GO_MF
**Localization:** DeepLocBinary • DeepLocMulti • DeepLoc2Multi
**Stability:** Thermostability • TAPE_Stability
**Solubility:** DeepSol • DeepSoluE • eSOL • ProtSolM • PETA_CHS/LGK/TEM_Sol
**Mutation:** FLIP_AAV (7 splits) • FLIP_GB1 (5 splits) • TAPE_Fluorescence
**Others:** DeepET_Topt • MetalIonBinding • SortingSignal • PaCRISPR

All datasets available on [HuggingFace](https://huggingface.co/AI4Protein)

</details>

---

## 📦 Installation

<details>
<summary><b>🍎 macOS (M1/M2/M3)</b></summary>

```bash
git clone https://github.com/AI4Protein/VenusFactory.git && cd VenusFactory
conda create -n venus python=3.12 && conda activate venus
pip install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu
pip install torch_scatter torch-sparse torch-geometric -f https://data.pyg.org/whl/torch-2.8.0+cpu.html
pip install -r requirements_for_macOS.txt
```

</details>

<details>
<summary><b>🪟 Windows / 🐧 Linux (CUDA 12.8)</b></summary>

```bash
git clone https://github.com/AI4Protein/VenusFactory.git && cd VenusFactory
conda create -n venus python=3.12 && conda activate venus
pip install torch==2.8.0 torchvision --index-url https://download.pytorch.org/whl/cu128
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.8.0+cu128.html
pip install -r requirements.txt
```

</details>

<details>
<summary><b>🪟 Windows / 🐧 Linux (CUDA 11.8)</b></summary>

```bash
git clone https://github.com/AI4Protein/VenusFactory.git && cd VenusFactory
conda create -n venus python=3.12 && conda activate venus
pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu118
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.7.0+cu118.html
pip install -r requirements.txt
```

</details>

<details>
<summary><b>💻 CPU Only</b></summary>

```bash
git clone https://github.com/AI4Protein/VenusFactory.git && cd VenusFactory
conda create -n venus python=3.12 && conda activate venus
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.8.0+cpu.html
pip install -r requirements.txt
```

</details>

**Verify:** `python -c "import torch; print(torch.__version__)"`

---

## 🚀 Usage

### Web UI

```bash
python src/webui.py  # → http://localhost:7860
```

| Tab | Purpose | Features |
|:----|:--------|:---------|
| **Training** | Train custom models | Model selection • PEFT methods • Real-time monitoring • Wandb |
| **Evaluation** | Benchmark testing | Load model • Select metrics • CSV export |
| **Prediction** | Inference | Single/batch prediction • Result visualization |
| **Agent** | Natural language | Multi-step automation • Tool orchestration |
| **Quick Tools** | Rapid prediction | Mutation scoring • Function prediction |
| **Advanced** | Deep analysis | Sequence/structure-based models |
| **Download** | Data retrieval | AlphaFold • UniProt • RCSB • InterPro |
| **Manual** | Documentation | Guides & tutorials |

<details>
<summary><b>Screenshots</b></summary>

![Training](img/Train/Model_Dataset_Config.png)
![Evaluation](img/Eval/Model_Dataset_Config.png)
![Prediction](img/Predict/Predict_Tab.png)

</details>

### CLI & API

<details>
<summary><b>Command Line Examples</b></summary>

```bash
# Train model
bash script/train/train_plm_lora.sh \
  --model facebook/esm2_t33_650M_UR50D \
  --dataset DeepSol --batch_size 32

# Evaluate
bash script/eval/eval.sh \
  --model_path ckpt/DeepSol/best_model \
  --test_dataset DeepSol

# Download data
bash script/tools/database/alphafold/download_alphafold_structure.sh
bash script/tools/database/uniprot/download_uniprot_seq.sh

# Generate structure sequences
bash script/get_structure_seq/get_esm3_structure_seq.sh
```

</details>

<details>
<summary><b>REST API Examples</b></summary>

```bash
# Start server
python src/api_server.py  # → http://localhost:5000/docs

# Mutation prediction
curl -X POST http://localhost:5000/api/mutation/predict \
  -H "Content-Type: application/json" \
  -d '{"sequence": "MKTAYIA...", "mutations": ["A23V", "K45R"]}'

# Function prediction
curl -X POST http://localhost:5000/api/predict/function \
  -H "Content-Type: application/json" \
  -d '{"sequence": "MKTAYIA...", "tasks": ["solubility", "stability"]}'

# Database search
curl http://localhost:5000/api/database/uniprot/search?query=lysozyme&limit=10
```

</details>

<details>
<summary><b>Python API</b></summary>

```python
from src.tools.mutation import predict_mutation_effects
from src.tools.predict import predict_protein_function
from src.tools.database import download_alphafold_structure

# Mutations
results = predict_mutation_effects(
    sequence="MKTAYIAKQR...",
    mutations=["A5V", "K9R"],
    model="esm2"
)

# Function
predictions = predict_protein_function(
    sequence="MKTAYIA...",
    tasks=["solubility", "stability"]
)

# Data
pdb_file = download_alphafold_structure("P12345")
```

</details>

---

## 📊 Training Methods

| Method | Memory | Speed | Performance | Best For |
|:-------|:------:|:-----:|:-----------:|:---------|
| **LoRA** | Low | Fast | Good | General tasks |
| **QLoRA** | Very Low | Slow | Good | Limited GPU |
| **DoRA** | Low | Medium | Better | Improved LoRA |
| **AdaLoRA** | Low | Medium | Better | Adaptive rank |
| **SES-Adapter** | Medium | Medium | Better | Selective tuning |
| **IA3** | Very Low | Fast | Good | Lightweight |
| **Freeze** | Low | Fast | Good | Simple tuning |

---

## 🙌 Citation

```bibtex
@inproceedings{tan2025venusfactory,
  title={VenusFactory: An Integrated System for Protein Engineering with Data Retrieval and Language Model Fine-Tuning},
  author={Tan, Yang and Liu, Chen and Gao, Jingyuan and Banghao, Wu and Li, Mingchen and Wang, Ruilin and Zhang, Lingrong and Yu, Huiqun and Fan, Guisheng and Hong, Liang and others},
  booktitle={Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 3: System Demonstrations)},
  pages={230--241},
  year={2025}
}
```

---

## 🎊 Acknowledgement

Developed by [Liang's Lab](https://ins.sjtu.edu.cn/people/lhong/index.html) at Shanghai Jiao Tong University.

**Resources:** [Docs](https://venusfactory.readthedocs.io/) • [YouTube](https://www.youtube.com/watch?v=MT6lPH5kgCc&ab_channel=BxinZhou) • [Playground](https://venusfactory.cn/playground/) • [Issues](https://github.com/AI4Protein/VenusFactory/issues)

---

<div align="center">

**Made with ❤️ for the protein engineering community**

[⭐ Star](https://github.com/AI4Protein/VenusFactory) • [🐛 Report Bug](https://github.com/AI4Protein/VenusFactory/issues) • [💡 Request Feature](https://github.com/AI4Protein/VenusFactory/issues)

</div>
