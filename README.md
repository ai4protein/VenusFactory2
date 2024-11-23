![# ProFactory](img/banner.png)

Recent News:

- Welcome to ProFactory!

## ✏️ Table of Contents

- [Features](#-features)
- [Supported Models](#-supported-models)
- [Supported Training Approaches](#-supported-training-approaches)
- [Supported Datasets](#-supported-datasets)
- [Supported Metrics](#supported-metrics)
- [Citation](#citation)

## 📑 Features

## 🧬 Supported Models

| Model                                                        | Model size              | Template                        |
| ------------------------------------------------------------ | ----------------------- | ------------------------------- |
| [ESM2](https://huggingface.co/facebook/esm2_t33_650M_UR50D)  | 8M/35M/150M/650M/3B/15B | facebook/esm2_t33_650M_UR50D    |
| [ESM-1b](https://huggingface.co/facebook/esm1b_t33_650M_UR50S) | 650M                    | facebook/esm1b_t33_650M_UR50S   |
| [ESM-1v](https://huggingface.co/facebook/esm1v_t33_650M_UR90S_1) | 650M                    | facebook/esm1v_t33_650M_UR90S_1 |
| [ProtBert-Uniref100](https://huggingface.co/Rostlab/prot_bert) | 420M                    | Rostlab/prot_bert_bfd           |
| [ProtBert-BFD100](https://huggingface.co/Rostlab/prot_bert_bfd) | 420M                    | Rostlab/prot_bert_bfd           |
| [ProtT5-Uniref50](https://huggingface.co/Rostlab/prot_t5_xl_uniref50) | 3B/11B                  | Rostlab/prot_t5_xl_uniref50     |
| [ProtT5-BFD100](https://huggingface.co/Rostlab/prot_t5_xl_bfd) | 3B/11B                  | Rostlab/prot_t5_xl_bfd          |
| [Ankh](https://huggingface.co/ElnaggarLab/ankh-base)         | 450M/1.2B               | ElnaggarLab/ankh-base           |

## 🔬 Supported Training Approaches

| Approach               | Full-tuning | Freeze-tuning      | LoRA               | SES-Adapter        |
| ---------------------- | ----------- | ------------------ | ------------------ | ------------------ |
| Pre-Training           | ❎           | ❎                  | ❎                  | ❎                  |
| Supervised Fine-Tuning | ❎           | :white_check_mark: | :white_check_mark: | :white_check_mark: |

## 📚 Supported Datasets

<details><summary>Pre-training datasets</summary>


- [CATH_V43_S40](https://huggingface.co/datasets/tyang816/cath) | structures

</details>

<details><summary>Supervised fine-tuning datasets (amino acid sequences/ foldseek sequences/ ss8 sequences)</summary>

- [DeepLocBinary_ESMFold](https://huggingface.co/datasets/tyang816/DeepLocBinary_ESMFold) | protein-wise | single_label_classification
- [DeepLocBinary_AlphaFold2](https://huggingface.co/datasets/tyang816/DeepLocBinary_AlphaFold2) | protein-wise | single_label_classification
- [DeepLocMulti_ESMFold](https://huggingface.co/datasets/tyang816/DeepLocMulti_ESMFold) | protein-wise | single_label_classification
- [DeepLocMulti_AlphaFold2](https://huggingface.co/datasets/tyang816/DeepLocMulti_AlphaFold2) | protein-wise | single_label_classification
- [DeepSol_ESMFold](https://huggingface.co/datasets/tyang816/DeepSol_ESMFold) | protein-wise | single_label_classification
- [DeepSoluE_ESMFold](https://huggingface.co/datasets/tyang816/DeepSoluE_ESMFold) | protein-wise | single_label_classification
- [ProtSolM_ESMFold](https://huggingface.co/datasets/tyang816/ProtSolM_ESMFold) | protein-wise | single_label_classification
- [EC_ESMFold](https://huggingface.co/datasets/tyang816/EC_ESMFold) | protein-wise | multi_label_classification
- [EC_AlphaFold2](https://huggingface.co/datasets/tyang816/EC_AlphaFold2) | protein-wise | multi_label_classification
- [GO_BP_ESMFold](https://huggingface.co/datasets/tyang816/GO_BP_ESMFold) | protein-wise | multi_label_classification
- [GO_BP_AlphaFold2](https://huggingface.co/datasets/tyang816/GO_BP_AlphaFold2) | protein-wise | multi_label_classification
- [GO_CC_ESMFold](https://huggingface.co/datasets/tyang816/GO_CC_ESMFold) | protein-wise | multi_label_classification
- [GO_CC_AlphaFold2](https://huggingface.co/datasets/tyang816/GO_CC_AlphaFold2) | protein-wise | multi_label_classification
- [GO_MF_ESMFold](https://huggingface.co/datasets/tyang816/GO_MF_ESMFold) | protein-wise | multi_label_classification
- [GO_MF_AlphaFold2](https://huggingface.co/datasets/tyang816/GO_MF_AlphaFold2) | protein-wise | multi_label_classification
- [MetalIonBinding_ESMFold](https://huggingface.co/datasets/tyang816/MetalIonBinding_ESMFold) | protein-wise | single_label_classification
- [MetalIonBinding_AlphaFold2](https://huggingface.co/datasets/tyang816/MetalIonBinding_AlphaFold2) | protein-wise | single_label_classification
- [Thermostability_ESMFold](https://huggingface.co/datasets/tyang816/Thermostability_ESMFold) | protein-wise | regression
- [MetalIonBinding_AlphaFold2](https://huggingface.co/datasets/tyang816/MetalIonBinding_AlphaFold2) | protein-wise | regression

> [!TIP]
> Only structural sequences are different for the same dataset, for example, ``DeepLocBinary_ESMFold`` and ``DeepLocBinary_AlphaFold2`` share the same amino acid sequences, this means if you only want to use the ``aa_seqs``, both are ok! 

</details>

<details><summary>Pre-training datasets (amino acid sequences)</summary>

- FLIP_AAV |  protein-site | regression
    - [FLIP_AAV_one-vs-rest](https://huggingface.co/datasets/tyang816/FLIP_AAV_one-vs-rest), [FLIP_AAV_two-vs-rest](https://huggingface.co/datasets/tyang816/FLIP_AAV_two-vs-rest), [FLIP_AAV_mut-des](https://huggingface.co/datasets/tyang816/FLIP_AAV_mut-des), [FLIP_AAV_des-mut](https://huggingface.co/datasets/tyang816/FLIP_AAV_des-mut), [FLIP_AAV_seven-vs-rest](https://huggingface.co/datasets/tyang816/FLIP_AAV_seven-vs-rest), [FLIP_AAV_low-vs-high](https://huggingface.co/datasets/tyang816/FLIP_AAV_low-vs-high), [ FLIP_AAV_sampled](https://huggingface.co/datasets/tyang816/FLIP_AAV_sampled)
- FLIP_GB1 |  protein-site | regression
    - [FLIP_GB1_one-vs-rest](https://huggingface.co/datasets/tyang816/FLIP_GB1_one-vs-rest), [FLIP_GB1_two-vs-rest](https://huggingface.co/datasets/tyang816/FLIP_GB1_two-vs-rest), [FLIP_GB1_three-vs-rest](https://huggingface.co/datasets/tyang816/FLIP_GB1_three-vs-rest), [FLIP_GB1_low-vs-high](https://huggingface.co/datasets/tyang816/FLIP_GB1_low-vs-high), [FLIP_GB1_sampled](https://huggingface.co/datasets/tyang816/FLIP_GB1_sampled)

</details>

## 📈 Supported Metrics

| Metric Name   | Full Name        | Problem Type                                            |
| ------------- | ---------------- | ------------------------------------------------------- |
| accuracy      | Accuracy         | single_label_classification/ multi_label_classification |
| recall        | Recall           | single_label_classification/ multi_label_classification |
| precision     | Precision        | single_label_classification/ multi_label_classification |
| f1            | F1Score          | single_label_classification/ multi_label_classification |
| mcc           | MatthewsCorrCoef | single_label_classification/ multi_label_classification |
| auc           | AUROC            | single_label_classification/ multi_label_classification |
| f1_max        | F1ScoreMax       | multi_label_classification                              |
| spearman_corr | SpearmanCorrCoef | regression                                              |

## 🙌 Citation

Please cite our work if you have used our code or data.

```

```