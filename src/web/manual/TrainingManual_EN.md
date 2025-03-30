# VenusFactory Training Module User Guide

## 1. Introduction

The VenusFactory Training Module is a powerful tool that allows you to train custom models using protein sequence data. These models can predict various protein properties such as subcellular localization, function, stability, and more. The training module provides an intuitive interface that enables biological researchers to train high-performance protein prediction models without programming knowledge.

## 2. Supported Protein Language Models

VenusFactory supports various advanced protein language models. You can choose the appropriate model based on your task requirements and computational resources.

| Model Name                                                    | Model Parameter Size     | Number of Models | Model Example                   |
| ------------------------------------------------------------ | ----------------------- | ---------------- | ------------------------------- |
| [ESM2](https://huggingface.co/facebook/esm2_t33_650M_UR50D)  | 8M/35M/150M/650M/3B/15B | 6                | facebook/esm2_t33_650M_UR50D    |
| [ESM-1b](https://huggingface.co/facebook/esm1b_t33_650M_UR50S) | 650M                    | 1                | facebook/esm1b_t33_650M_UR50S   |
| [ESM-1v](https://huggingface.co/facebook/esm1v_t33_650M_UR90S_1) | 650M                    | 5                | facebook/esm1v_t33_650M_UR90S_1 |
| [ProtBert-Uniref100](https://huggingface.co/Rostlab/prot_bert) | 420M                    | 1                | Rostlab/prot_bert_bfd           |
| [ProtBert-BFD100](https://huggingface.co/Rostlab/prot_bert_bfd) | 420M                    | 1                | Rostlab/prot_bert_bfd           |
| [IgBert](https://huggingface.co/Exscientia/IgBert) | 420M                    | 1                | Exscientia/IgBert               |
| [IgBert_unpaired](https://huggingface.co/Exscientia/IgBert_unpaired) | 420M                    | 1                | Exscientia/IgBert_unpaired      |
| [ProtT5-Uniref50](https://huggingface.co/Rostlab/prot_t5_xl_uniref50) | 3B/11B                  | 2                | Rostlab/prot_t5_xl_uniref50     |
| [ProtT5-BFD100](https://huggingface.co/Rostlab/prot_t5_xl_bfd) | 3B/11B                  | 2                | Rostlab/prot_t5_xl_bfd          |
| [IgT5](https://huggingface.co/Exscientia/IgT5) | 3B                      | 1                | Exscientia/IgT5                 |
| [IgT5_unpaired](https://huggingface.co/Exscientia/IgT5_unpaired) | 3B                      | 1                | Exscientia/IgT5_unpaired        |
| [Ankh](https://huggingface.co/ElnaggarLab/ankh-base)         | 450M/1.2B               | 2                | ElnaggarLab/ankh-base           |
| [ProSST](https://huggingface.co/AI4Protein/ProSST-2048)      | 110M                    | 7                | AI4Protein/ProSST-2048          |
| [ProPrime](https://huggingface.co/AI4Protein/Prime_690M)     | 690M                    | 1                | AI4Protein/Prime_690M           |



## 3. Supported Fine-tuning Methods

VenusFactory provides multiple training methods, each with specific advantages and applicable scenarios.

| Fine-tuning Method | Description | Data Type |
|---------|------|------------|
| **Freeze** | Freezes the pre-trained model, training only the classifier | Sequence information |
| **Full** | Full parameter fine-tuning, training all parameters | Sequence information |
| **LoRA** | Uses Low-Rank Adaptation method to reduce parameter count | Sequence information |
| **DoRA** | Uses Weight-Decomposed Low-Rank Adaptation method | Sequence information |
| **AdaLoRA** | Uses Adaptive Low-Rank Adaptation method | Sequence information |
| **IA3** | Uses Infused Adapter by Inhibiting and Amplifying Inner Activations method | Sequence information |
| **QLoRA** | Uses Quantized Low-Rank Adaptation method to reduce memory requirements | Sequence information |
| **SES-Adapter** | Uses Structure-Enhanced Sequence Adapter, integrating sequence and structure information | Sequence & Structure information |

## 4. Supported Evaluation Metrics

VenusFactory provides multiple evaluation metrics to assess model performance.

| Abbreviation | Metric Name | Applicable Problem Types | Description | Optimization Direction |
|---------|------|------------|------|---------|
| **Accuracy** | Accuracy | Single-label/Multi-label classification | Proportion of correctly predicted samples, suitable for balanced datasets | Higher is better |
| **Recall** | Recall | Single-label/Multi-label classification | Proportion of correctly identified positive classes, focuses on reducing false negatives | Higher is better |
| **Precision** | Precision | Single-label/Multi-label classification | Proportion of correctly predicted positive classes, focuses on reducing false positives | Higher is better |
| **F1** | F1 Score | Single-label/Multi-label classification | Harmonic mean of precision and recall, suitable for imbalanced datasets | Higher is better |
| **MCC** | Matthews Correlation Coefficient | Single-label/Multi-label classification | Metric that considers all confusion matrix elements, fairer for imbalanced datasets | Higher is better |
| **AUROC** | Area Under ROC Curve | Single-label/Multi-label classification | Evaluates classification performance at different thresholds | Higher is better |
| **F1_max** | Maximum F1 Score | Multi-label classification | Maximum F1 value at different thresholds, suitable for multi-label classification | Higher is better |
| **Spearman_corr** | Spearman Correlation Coefficient | Regression | Evaluates the monotonic relationship between predicted and true values, range [-1,1] | Higher is better |
| **MSE** | Mean Squared Error | Regression | Evaluates prediction error of regression models | Lower is better |

## 5. Training Interface Details

The training interface is divided into several main sections, each containing specific configuration options.

### 5.1 Model and Dataset Configuration

#### Protein Language Model Selection
- **Protein Language Model**: Select a pre-trained model from the dropdown menu
  - Consider your computational resources and task complexity when selecting
  - Larger models require more computational resources

#### Dataset Selection
- **Dataset Selection**: Choose the dataset source
  - **Use Pre-defined Dataset**: Use system-defined datasets
    - **Dataset Configuration**: Select a dataset from the dropdown menu
    - The system will automatically load the problem type, number of labels, and evaluation metrics
  - **Use Custom Dataset**: Use a custom dataset
    - **Custom Dataset Path**: Enter the Hugging Face dataset path (format: `username/dataset_name`)
    - **Problem Type**: Select the problem type
      - `single_label_classification`: Single-label classification
      - `multi_label_classification`: Multi-label classification
      - `regression`: Regression
    - **Number of Labels**: Set the number of labels (for classification problems)
    - **Metrics**: Select evaluation metrics (multiple selections allowed)
      - `accuracy`: Accuracy
      - `f1`: F1 Score
      - `precision`: Precision
      - `recall`: Recall
      - `mcc`: Matthews Correlation Coefficient
      - `auroc`: Area Under the ROC Curve
      - `f1max`: Maximum F1 Score
      - `spearman_corr`: Spearman Correlation Coefficient
      - `mse`: Mean Squared Error

      For more information, refer to [4. Supported Evaluation Metrics](#header-4)


#### Dataset Preview
- **Preview Dataset**: Click this button to preview the selected dataset
  - Displays dataset statistics: number of samples in training, validation, and test sets
  - Displays dataset examples: including sequences and labels

### 5.2 Training Method Configuration

- **Training Method**: Select the training method
  - `freeze`: Freeze the pre-trained model, train only the classifier
  - `full`: Full parameter fine-tuning, train all parameters
  - `plm-lora`: Use LoRA (Low-Rank Adaptation) method to reduce parameter count
  - `dora`: Use DoRA (Weight-Decomposed Low-Rank Adaptation) method
  - `adalora`: Use AdaLoRA (Adaptive Low-Rank Adaptation) method
  - `ia3`: Use IA³ (Infused Adapter by Inhibiting and Amplifying Inner Activations) method
  - `plm-qlora`: Use QLoRA (Quantized Low-Rank Adaptation) method to reduce memory requirements
  - `ses-adapter`: Use Structure-Enhanced Sequence Adapter, integrating sequence and structure information

  For more information, refer to [3. Supported Fine-tuning Methods](#header-3)

- **Pooling Method**: Select the pooling method
  - `mean`: Mean pooling
  - `attention1d`: Attention pooling
  - `light_attention`: Lightweight attention pooling

- **Structure Sequence** (visible when `ses-adapter` is selected):
  - Select structure sequence types (multiple selections allowed), default is `foldseek_seq` and `ss8_seq`

- **LoRA Parameters** (visible when `plm-lora` or `plm-qlora` is selected):
  - **LoRA Rank**: The rank of LoRA, default is 8, affects parameter count and performance
  - **LoRA Alpha**: The alpha value of LoRA, default is 32, affects scaling factor
  - **LoRA Dropout**: The dropout rate of LoRA, default is 0.1, affects regularization
  - **LoRA Target Modules**: Target modules for LoRA application, default is `query,key,value`

### 5.3 Batch Processing Configuration

- **Batch Processing Mode**: Select the batch processing mode
  - **Batch Size Mode**: Fixed batch size
    - **Batch Size**: Set the number of samples per batch, default is 16
  - **Batch Token Mode**: Fixed token count
    - **Tokens per Batch**: Set the number of tokens per batch, default is 10000
    - Suitable for datasets with large variations in sequence length

### 5.4 Training Parameters

- **Learning Rate**: Learning rate, default is 5e-4
  - Affects the step size of model training; larger values may cause non-convergence, smaller values may cause slow training

- **Number of Epochs**: Number of training epochs, default is 100
  - Number of complete passes through the dataset
  - Actual training may end earlier due to early stopping

- **Early Stopping Patience**: Early stopping patience N, default is 10
  - Training will stop early if validation performance does not improve for N consecutive epochs

- **Max Sequence Length**: Maximum sequence length, default is None (-1 indicates no limit)
  - Maximum protein sequence length to process

- **Scheduler Type**: Learning rate scheduler type
  - `linear`: Linear decay
  - `cosine`: Cosine decay
  - `step`: Step decay
  - `None`: No scheduler

- **Warmup Steps**: Number of warmup steps, default is 0
  - Number of steps where the learning rate gradually increases from a small value to the set value
  - Helps stabilize early training

- **Gradient Accumulation Steps**: Number of gradient accumulation steps, default is 1
  - Accumulates gradients from multiple batches before updating the model
  - Can simulate larger batch sizes

- **Max Gradient Norm**: Gradient clipping threshold, default is -1 (no clipping)
  - Limits the maximum norm of gradients to prevent gradient explosion
  - Recommended range: 1.0 to 5.0

- **Number of Workers**: Number of data loading worker threads, default is 4
  - Affects data loading speed
  - Adjust based on CPU core count

### 5.5 Output and Logging Settings

- **Save Directory**: Save directory, default is `ckpt`
  - Path to save model and training results

- **Output Model Name**: Output model name, default is `model.pt`
  - Filename of the saved model

- **Enable W&B Logging**: Whether to enable Weights & Biases logging
  - When checked, you can set W&B project name and entity
  - Used for experiment tracking and visualization

### 5.6 Training Control and Output

- **Preview Command**: Preview the training command to be executed
  - Click to display the complete command line arguments

- **Abort**: Abort the current training process

- **Start**: Start the training process

- **Model Statistics**: Display model parameter statistics
  - Parameter counts for the training model, pre-trained model, and combined model
  - Percentage of trainable parameters

- **Training Progress**: Display training progress
  - Current phase (training, validation, testing)
  - Progress percentage
  - Time elapsed and estimated time remaining
  - Current loss value and gradient steps

- **Best Performance**: Display best model information
  - Best epoch and corresponding evaluation metrics

- **Training and Validation Loss**: Loss curve graph
  - Training loss and validation loss over time

- **Validation Metrics**: Validation set evaluation metrics graph
  - Various evaluation metrics over time

- **Test Results**: Test results
  - Final performance metrics on the test set
  - Evaluation metrics can be downloaded in CSV format

## 6. Training Process Guide

Below is a complete guide to using the VenusFactory training module, from data preparation to model evaluation.

### 6.1 Preparing the Dataset

#### Using Pre-defined Datasets
1. Select "Use Pre-defined Dataset" in **Dataset Selection**
2. Choose a dataset from the **Dataset Configuration** dropdown menu
3. Click the **Preview Dataset** button to view dataset statistics and examples

#### Using Custom Datasets
1. Prepare a dataset that meets the requirements and upload it to Hugging Face (see [Custom Dataset Format Requirements](#header-7))
2. Select "Use Custom Dataset" in **Dataset Selection**
3. Enter the Hugging Face dataset path in **Custom Dataset Path** (format: `username/dataset_name`)
4. Set **Problem Type**, **Number of Labels**, and **Metrics**
5. Click the **Preview Dataset** button to verify that the dataset is loaded correctly

### 6.2 Selecting a Model and Training Method

1. Choose a pre-trained model from the **Protein Language Model** dropdown menu

2. Select an appropriate **Training Method**

3. Choose a **Pooling Method**

4. If selecting `ses-adapter`, ensure you specify structure sequence types in **Structure Sequence**
5. If selecting `plm-lora` or `plm-qlora`, adjust LoRA parameters as needed

### 6.3 Configuring Batch Processing and Training Parameters

1. Select **Batch Processing Mode**
   - Use **Batch Size Mode** when sequence lengths are similar
   - Use **Batch Token Mode** when sequence lengths vary significantly

2. Set batch size or token count
   - Adjust based on GPU memory; reduce if memory errors occur

3. Set **Learning Rate**

4. Set **Number of Epochs**
   - Use early stopping mechanism; set **Early Stopping Patience** to 10-20 to prevent overfitting

5. Set **Max Sequence Length**

6. Adjust advanced parameters as needed
   - **Scheduler Type**: Recommend using `linear` or `cosine`
   - **Warmup Steps**: Recommend setting to 5-10% of total steps
   - **Gradient Accumulation Steps**: Increase if memory is insufficient
   - **Max Gradient Norm**: Set to 1.0-5.0 if training is unstable

### 6.4 Setting Output and Logging

1. Set **Save Directory** as the path to save the model
2. Set **Output Model Name** as the model filename
3. If you need to track training, check **Enable W&B Logging** and set project information

### 6.5 Starting Training

1. Click **Preview Command** to preview the training command
2. Click the **Start** button to begin training
3. Observe training progress and metric changes
4. After training is complete, view the test results
   - Check various evaluation metrics
   - Download results in CSV format if needed
5. To stop training, click the **Abort** button

## 7. Custom Dataset Format Requirements

To use a custom dataset, you need to upload the dataset to the Hugging Face platform and ensure it meets the following format requirements.

### 7.1 Basic Requirements

- The dataset must include `train`, `validation`, and `test` subsets
- Each sample must contain the following fields:
  - `aa_seq`: Amino acid sequence using standard single-letter codes
  - `label`: Label, format depends on the problem type

### 7.2 Label Formats for Different Problem Types

#### Single-label Classification (single_label_classification)
- `label`: Integer value representing the class index (starting from 0)
- Example: 0, 1, 2, ...

CSV format example:
```csv
aa_seq,label
MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG,1
MLKFQQFGKGVLTEQKHALSELVCGLLEGRPFSQHEKETITIGIINIANNNDLFSAYK,0
MSDKIIHLTDDSFDTDVLKADGAILVDFWAEWCGPCKMIAPILDEIADEYQGKLTVAK,2
```

#### Multi-label Classification (multi_label_classification)
- `label`: String of comma-separated class indices representing present classes
- Example: "373,449,584,674,780,883,897,911,1048,1073,1130,1234"

CSV format example:
```csv
aa_seq,label
MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG,"373,449,584,674,780,883,897,911,1048,1073,1130,1234"
MLKFQQFGKGVLTEQKHALSELVCGLLEGRPFSQHEKETITIGIINIANNNDLFSAYK,"15,42,87,103,256"
MSDKIIHLTDDSFDTDVLKADGAILVDFWAEWCGPCKMIAPILDEIADEYQGKLTVAK,"7,98,120,256,512,789"
```

#### Regression (regression)
- `label`: Floating-point number representing a continuous value
- Examples: 0.75, -1.2, ...

CSV format example:
```csv
aa_seq,label
MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG,0.75
MLKFQQFGKGVLTEQKHALSELVCGLLEGRPFSQHEKETITIGIINIANNNDLFSAYK,-1.2
MSDKIIHLTDDSFDTDVLKADGAILVDFWAEWCGPCKMIAPILDEIADEYQGKLTVAK,3.45
```

### 7.3 Structural Information (Optional)

If using the `ses-adapter` training method, you can add the following structural information fields:

- `foldseek_seq`: FoldSeek structure sequence, using single-letter codes to represent structural elements
- `ss8_seq`: 8-class secondary structure sequence, using single-letter codes to represent secondary structures

CSV format example:
```csv
name,aa_seq,labelname,aa_seq,foldseek_seq,ss8_seq,label
Q9LSD8,MPEEDLVELKFRLYDGSDVGPFQYSPTATVSMLKERIVSEWPKDKKIVPKSASDIKLINAGKILENGKTVAQCKAPFDDLPKSVITMHVVVQLSPTKARPEKKIEKEEAPQRSFCSCTIM,DPPQLWAFAWEAEPVRDIDDRDTDHQQQFLLVVLQVCLVRPDPPDPDHAPHSVQKWKDDPNDTGDRNDGNNRRDDPPDDDSPDHHYIYIDGRDPPVVPPVPPPPPPPPPPPPPPPPPPPD,LLLLLLEEEEEELTTSLEEEEEEELTTLBHHHHHHHHHHTLLTTLSSLLSSGGGEEEEETTEELLTTLBHHHHLLLLLLLTTLLEEEEEEELLLLLLLLLLLLLLLLLLLLLLLLLLLLL,0
```

### 7.4 Uploading to Hugging Face

1. Create separate CSV files for training, validation, and test sets:
   - `train.csv`: Training data
   - `validation.csv`: Validation data
   - `test.csv`: Test data

2. Upload the dataset to Hugging Face

- The relevant steps are shown in the following images:

![HF1](/img/HuggingFace/HF1.png)
![HF2](/img/HuggingFace/HF2.png)
![HF3](/img/HuggingFace/HF3.png)
![HF4](/img/HuggingFace/HF4.png)

After uploading, use `Owner/Dataset name` as the Custom Dataset Path in VenusFactory