![# VenusFactory](img/banner.png)

Recent News:

- Welcome to VenusFactory!

## ✏️ Table of Contents

- [Features](#-features)
- [Supported Models](#-supported-models)
- [Supported Training Approaches](#-supported-training-approaches)
- [Supported Datasets](#-supported-datasets)
- [Supported Metrics](#-supported-metrics)
- [Requirements](#-requirements)
- [Quick Start with Venus Board UI](#-quick-start-with-venus-board-ui)
- [Code-line Usage](#-code-line-usage)
- [Citation](#-citation)
- [Acknowledgement](#-acknowledgement)

## 📑 Features

- **Vaious protein langugae models**: ESM2, ESM-b, ESM-1v, ProtBert, ProtT5, Ankh, etc
- **Comprehensive supervised datasets**: Localization, Fitness, Solubility, Stability, etc
- **Easy and quick data collector**: AlphaFold2 Database, RCSB, InterPro, Uniprot, etc
- **Experiment moitors**: Wandb, Local
- **Friendly interface**: Gradio UI

## 🤖 Supported Models

| Model                                                        | Model size              | Template                        |
| ------------------------------------------------------------ | ----------------------- | ------------------------------- |
| [ESM2](https://huggingface.co/facebook/esm2_t33_650M_UR50D)  | 8M/35M/150M/650M/3B/15B | facebook/esm2_t33_650M_UR50D    |
| [ESM-1b](https://huggingface.co/facebook/esm1b_t33_650M_UR50S) | 650M                    | facebook/esm1b_t33_650M_UR50S   |
| [ESM-1v](https://huggingface.co/facebook/esm1v_t33_650M_UR90S_1) | 650M                    | facebook/esm1v_t33_650M_UR90S_1 |
| [ProtBert-Uniref100](https://huggingface.co/Rostlab/prot_bert) | 420M                    | Rostlab/prot_bert_uniref100          |
| [ProtBert-BFD100](https://huggingface.co/Rostlab/prot_bert_bfd) | 420M                    | Rostlab/prot_bert_bfd           |
| [IgBert](https://huggingface.co/Exscientia/IgBert) | 420M                    | Exscientia/IgBert           |
| [IgBert_unpaired](https://huggingface.co/Exscientia/IgBert_unpaired) | 420M                    | Exscientia/IgBert_unpaired           |
| [ProtT5-Uniref50](https://huggingface.co/Rostlab/prot_t5_xl_uniref50) | 3B/11B                  | Rostlab/prot_t5_xl_uniref50     |
| [ProtT5-BFD100](https://huggingface.co/Rostlab/prot_t5_xl_bfd) | 3B/11B                  | Rostlab/prot_t5_xl_bfd          |
| [IgT5](https://huggingface.co/Exscientia/IgT5) | 3B                  | Exscientia/IgT5          |
| [IgT5_unpaired](https://huggingface.co/Exscientia/IgT5_unpaired) | 3B                  | Exscientia/IgT5_unpaired          |
| [Ankh](https://huggingface.co/ElnaggarLab/ankh-base)         | 450M/1.2B               | ElnaggarLab/ankh-base           |
| [ProSST](https://huggingface.co/AI4Protein/ProSST-2048)  |20/128/512/1024/2048/4096  |AI4Protein/ProSST-2048     |
| [ProPrime](https://huggingface.co/AI4Protein/Prime_690M)  |690M                     |AI4Protein/Prime_690M     |
| [PETA](https://huggingface.co/AI4Protein/deep_base)     |base/bpe_50/bep_100/bep_200/bep_400/bep_800/bpe_1600/bpe_3200/<br>unigram_50/unigram_100/unigram_200/unigram_400/unigram_800/unigram_1600/unigram_3200 |AI4Protein/deep_base     |

## 🔬 Supported Training Approaches

| Approach               | Full-tuning | Freeze-tuning      | SES-Adapter        | AdaLoRA            | QLoRA      | LoRA               | DoRA            | IA3              | 
| ---------------------- | ----------- | ------------------ | ------------------ | ------------------ |----------- | ------------------ | -----------------| -----------------|
| Pre-Training           | ❎          | ❎                | ❎                 | ❎                |❎          | ❎                | ❎               | ❎              | 
| Supervised Fine-Tuning | ✅          | ✅                | ✅                 | ✅                |✅          | ✅                | ✅               | ✅              |

## 📚 Supported Datasets

<details><summary>Pre-training datasets</summary>


- [CATH_V43_S40](https://huggingface.co/datasets/tyang816/cath) | structures

</details>

<details><summary>Supervised fine-tuning datasets (amino acid sequences/ foldseek sequences/ ss8 sequences)</summary>

- DeepLocBinary | protein-wise | single_label_classification
    - [DeepLocBinary_AlphaFold2](https://huggingface.co/datasets/tyang816/DeepLocBinary_AlphaFold2)
    - [DeepLocBinary_ESMFold](https://huggingface.co/datasets/tyang816/DeepLocBinary_ESMFold)
- DeepLocMulti | protein-wise | single_label_classification
    - [DeepLocMulti_AlphaFold2](https://huggingface.co/datasets/tyang816/DeepLocMulti_AlphaFold2)
    - [DeepLocMulti_ESMFold](https://huggingface.co/datasets/tyang816/DeepLocMulti_ESMFold)
- DeepLoc2Multi | protein-wise | single_label_classification
    - [DeepLoc2Multi_AlphaFold2](https://huggingface.co/datasets/tyang816/DeepLoc2Multi_AlphaFold2)
    - [DeepLoc2Multi_ESMFold](https://huggingface.co/datasets/tyang816/DeepLoc2Multi_ESMFold)
- DeepSol | protein-wise | single_label_classification
    - [DeepSol_ESMFold](https://huggingface.co/datasets/tyang816/DeepSol_ESMFold)
- DeepSoluE | protein-wise | single_label_classification
    - [DeepSoluE_ESMFold](https://huggingface.co/datasets/tyang816/DeepSoluE_ESMFold)
- ProtSolM | protein-wise | single_label_classification
    - [ProtSolM_ESMFold](https://huggingface.co/datasets/tyang816/ProtSolM_ESMFold)
- eSOL | protein-wise | regression
    - [eSOL_AlphaFold2](https://huggingface.co/datasets/tyang816/eSOL_AlphaFold2)
    - [eSOL_ESMFold](https://huggingface.co/datasets/tyang816/eSOL_ESMFold)
- DeepET_Topt | protein-wise | regression
    - [DeepET_Topt_AlphaFold2](https://huggingface.co/datasets/tyang816/DeepET_Topt_AlphaFold2)
    - [DeepET_Topt_ESMFold](https://huggingface.co/datasets/tyang816/DeepET_Topt_ESMFold)
- EC | protein-wise | multi_label_classification
    - [EC_AlphaFold2](https://huggingface.co/datasets/tyang816/EC_AlphaFold2)
    - [EC_ESMFold](https://huggingface.co/datasets/tyang816/EC_ESMFold)
- GO_BP | protein-wise | multi_label_classification
    - [GO_BP_AlphaFold2](https://huggingface.co/datasets/tyang816/GO_BP_AlphaFold2)
    - [GO_BP_ESMFold](https://huggingface.co/datasets/tyang816/GO_BP_ESMFold)
- GO_CC | protein-wise | multi_label_classification
    - [GO_CC_AlphaFold2](https://huggingface.co/datasets/tyang816/GO_CC_AlphaFold2)
    - [GO_CC_ESMFold](https://huggingface.co/datasets/tyang816/GO_CC_ESMFold)
- GO_MF | protein-wise | multi_label_classification
    - [GO_MF_AlphaFold2](https://huggingface.co/datasets/tyang816/GO_MF_AlphaFold2)
    - [GO_MF_ESMFold](https://huggingface.co/datasets/tyang816/GO_MF_ESMFold)
- MetalIonBinding | protein-wise | single_label_classification
    - [MetalIonBinding_AlphaFold2](https://huggingface.co/datasets/tyang816/MetalIonBinding_AlphaFold2)
    - [MetalIonBinding_ESMFold](https://huggingface.co/datasets/tyang816/MetalIonBinding_ESMFold)
- Thermostability | protein-wise | regression
    - [Thermostability_AlphaFold2](https://huggingface.co/datasets/tyang816/Thermostability_AlphaFold2)
    - [Thermostability_ESMFold](https://huggingface.co/datasets/tyang816/Thermostability_ESMFold)

> ✨ Only structural sequences are different for the same dataset, for example, ``DeepLocBinary_ESMFold`` and ``DeepLocBinary_AlphaFold2`` share the same amino acid sequences, this means if you only want to use the ``aa_seqs``, both are ok! 

</details>

<details><summary>Supervised fine-tuning datasets (amino acid sequences)</summary>

- [Demo_Solubility](https://huggingface.co/datasets/tyang816/Demo_Solubility) | protein-wise | single_label_classification
- [DeepLocBinary](https://huggingface.co/datasets/tyang816/DeepLocBinary) | protein-wise | single_label_classification
- [DeepLocMulti](https://huggingface.co/datasets/tyang816/DeepLocMulti) | protein-wise | single_label_classification
- [DeepLoc2Multi](https://huggingface.co/datasets/tyang816/DeepLoc2Multi) | protein-wise | single_label_classification
- [DeepSol](https://huggingface.co/datasets/tyang816/DeepSol) | protein-wise | single_label_classification
- [DeepSoluE](https://huggingface.co/datasets/tyang816/DeepSoluE) | protein-wise | single_label_classification
- [ProtSolM](https://huggingface.co/datasets/tyang816/ProtSolM) | protein-wise | single_label_classification
- [eSOL](https://huggingface.co/datasets/tyang816/eSOL) | protein-wise | regression
- [DeepET_Topt](https://huggingface.co/datasets/tyang816/DeepET_Topt) | protein-wise | regression
- [EC](https://huggingface.co/datasets/tyang816/EC) | protein-wise | multi_label_classification
- [GO_BP](https://huggingface.co/datasets/tyang816/GO_BP) | protein-wise | multi_label_classification
- [GO_CC](https://huggingface.co/datasets/tyang816/GO_CC) | protein-wise | multi_label_classification
- [GO_MF](https://huggingface.co/datasets/tyang816/GO_MF) | protein-wise | multi_label_classification
- [MetalIonBinding](https://huggingface.co/datasets/tyang816/MetalIonBinding) | protein-wise | single_label_classification
- [Thermostability](https://huggingface.co/datasets/tyang816/Thermostability) | protein-wise | regression
- [PaCRISPR](https://huggingface.co/datasets/tyang816/PaCRISPR) | protein-wise
- [PETA_CHS_Sol](https://huggingface.co/datasets/tyang816/PETA_CHS_Sol) | protein-wise
- [PETA_LGK_Sol](https://huggingface.co/datasets/tyang816/PETA_LGK_Sol) | protein-wise
- [PETA_TEM_Sol](https://huggingface.co/datasets/tyang816/PETA_TEM_Sol) | protein-wise
- [SortingSignal](https://huggingface.co/datasets/tyang816/SortingSignal) | protein-wise
- FLIP_AAV | protein-site | regression
    - [FLIP_AAV_one-vs-rest](https://huggingface.co/datasets/tyang816/FLIP_AAV_one-vs-rest), [FLIP_AAV_two-vs-rest](https://huggingface.co/datasets/tyang816/FLIP_AAV_two-vs-rest), [FLIP_AAV_mut-des](https://huggingface.co/datasets/tyang816/FLIP_AAV_mut-des), [FLIP_AAV_des-mut](https://huggingface.co/datasets/tyang816/FLIP_AAV_des-mut), [FLIP_AAV_seven-vs-rest](https://huggingface.co/datasets/tyang816/FLIP_AAV_seven-vs-rest), [FLIP_AAV_low-vs-high](https://huggingface.co/datasets/tyang816/FLIP_AAV_low-vs-high), [FLIP_AAV_sampled](https://huggingface.co/datasets/tyang816/FLIP_AAV_sampled)
- FLIP_GB1 | protein-site | regression
    - [FLIP_GB1_one-vs-rest](https://huggingface.co/datasets/tyang816/FLIP_GB1_one-vs-rest), [FLIP_GB1_two-vs-rest](https://huggingface.co/datasets/tyang816/FLIP_GB1_two-vs-rest), [FLIP_GB1_three-vs-rest](https://huggingface.co/datasets/tyang816/FLIP_GB1_three-vs-rest), [FLIP_GB1_low-vs-high](https://huggingface.co/datasets/tyang816/FLIP_GB1_low-vs-high), [FLIP_GB1_sampled](https://huggingface.co/datasets/tyang816/FLIP_GB1_sampled)
- [TAPE_Fluorescence](https://huggingface.co/datasets/tyang816/TAPE_Fluorescence) | protein-site | regression
- [TAPE_Stability](https://huggingface.co/datasets/tyang816/TAPE_Stability) | protein-site | regression

</details>

## 📈 Supported Metrics

| Name          | Torchmetrics     | Problem Type                                            |
| ------------- | ---------------- | ------------------------------------------------------- |
| accuracy      | Accuracy         | single_label_classification/ multi_label_classification |
| recall        | Recall           | single_label_classification/ multi_label_classification |
| precision     | Precision        | single_label_classification/ multi_label_classification |
| f1            | F1Score          | single_label_classification/ multi_label_classification |
| mcc           | MatthewsCorrCoef | single_label_classification/ multi_label_classification |
| auc           | AUROC            | single_label_classification/ multi_label_classification |
| f1_max        | F1ScoreMax       | multi_label_classification                              |
| spearman_corr | SpearmanCorrCoef | regression                                              |
| mse           | MeanSquaredError | regression                                              |

## ✈️ Requirements

### Hardware Requirements
- Recommended: NVIDIA RTX 3090 (24GB) or better
- Actual requirements depend on your chosen protein language model

### Software Requirements
- [Anaconda3](https://www.anaconda.com/download) or [Miniconda3](https://docs.conda.io/projects/miniconda/en/latest/)
- Python 3.10

### Basic Installation
```bash
git clone https://github.com/tyang816/VenusFactory.git
cd VenusFactory
conda create -n venus python=3.10
conda activate venus  # For Windows
# source activate venus  # For Linux
pip install -r requirements.txt
```

## 🚀 Quick Start with Venus Web UI

Get started quickly with our intuitive graphical interface powered by [Gradio](https://github.com/gradio-app/gradio):

```bash
python ./src/webui.py
```

This will launch the Venus Web UI where you can:
- Configure and run fine-tuning experiments
- Monitor training progress
- Evaluate models
- Visualize results

### Using Each Tab

1. **Training Tab**:

    ![Model_Dataset_Config](img/Train/Model_Dataset_Config.png)

    Select a protein language model from the dropdown menu. Upload your dataset or select from available datasets and choose metrics appropriate for your problem type.

    ![Training_Parameters](img/Train/Training_Parameters.png)
   Choose a training method (Freeze, SES-Adapter, LoRA, QLoRA etc.) and configure training parameters (batch size, learning rate, etc.).
   
    ![Preview_Command](img/Train/Preview_Command.png)
    ![Training_Progress](img/Train/Training_Progress.png)
    ![Best_Model](img/Train/Best_Model.png)
    ![Monitor_Figs](img/Train/Monitor_Figs.png)
   Click "Start Training" and monitor progress in real-time.

    <p align="center">
      <img src="img/Train/Metric_Results.png" width="60%" alt="Metric_Results">
    </p>
   
   Click "Download CSV" to download the test metrics results.


2. **Evaluation Tab**:

    ![Model_Dataset_Config](img/Eval/Model_Dataset_Config.png)

   Load your trained model by specifying the model path. Select the same protein language model and model configs used during training. Select a test dataset and configure batch size. Choose evaluation metrics appropriate for your problem type. Finally, click "Start Evaluation" to view performance metrics.

3. **Prediction Tab**:

   ![Predict_Tab](img/Predict/Predict_Tab.png)

   Load your trained model by specifying the model path. Select the same protein language model and model configs used during training.

   For single sequence: Enter a protein sequence in the text box.

   For batch prediction: Upload a CSV file with sequences.

   ![Batch](img/Predict/Batch.png)

   Click "Predict" to generate and view results.

4. **Download Tab**:

   - **AlphaFold2 Structures**: Enter UniProt IDs to download protein structures
   - **UniProt**: Search for protein information using keywords or IDs
   - **InterPro**: Retrieve protein family and domain information
   - **RCSB PDB**: Download experimental protein structures

5. **Manual Tab**:

   Select a language (English/Chinese).

   Navigate through the documentation using the table of contents and find step-by-step guides.

## 🧬 Code-line Usage

### Training Methods
```bash
# Freeze-tuning
bash ./script/train/train_plm_vanilla.sh

# SES-Adapter
bash ./script/train/train_plm_ses-adapter.sh

# AdaLoRA
bash ./script/train/train_plm_adalora.sh

# QLoRA
bash ./script/train/train_plm_qlora.sh

# LoRA
bash ./script/train/train_plm_lora.sh

# DoRA
bash ./script/train/train_plm_dora.sh

# IA3
bash ./script/train/train_plm_ia3.sh
```

**eval**: Run the following scripts to evaluate the trained model.
```
bash ./script/eval/eval.sh
```

**Get structure sequence use esm3**
```
bash ./script/get_get_structure_seq/get_esm3_structure_seq.sh
```

**Get secondary structure sequence**
```
bash ./script/get_get_structure_seq/get_secondary_structure_seq.sh
```

### Crawler Collector
**Convert the cif to pdb format**
```
bash ./crawler/convert/maxit.sh
```

**Download the meta data from RCSB database**
```
bash ./crawler/metadata/download_rcsb.sh
```

**Download the protein sequence from Uniprot database**
```
bash ./crawler/sequence/download_uniprot_seq.sh
``` 

**Download the protein structure from AlphaFold2 or RCSB database**

AlphaFold2:
```
bash ./crawler/structure/download_alphafold.sh
```
RCSB: 
```
bash ./crawler/structure/download_rcsb.sh
```

## 🙌 Citation

Please cite our work if you have used our code or data.

## 🎊 Acknowledgement

Thanks the support of [Liang's Lab](https://ins.sjtu.edu.cn/people/lhong/index.html).
