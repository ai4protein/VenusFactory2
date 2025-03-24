# Quick Demo Guide

This document provides a comprehensive guide to help you quickly understand the main features of VenusFactory and perform fine-tuning, evaluation, and prediction on a demo dataset for protein solubility prediction.

## 1. Environment Preparation

Before starting, please ensure that you have successfully installed **VenusFactory** and correctly configured the corresponding environment and Python dependencies. If not yet installed, please refer to the **✈️ Requirements** section in [README.md](README.md) for installation instructions.

## 2. Launch Web Interface

Enter the following command in the command line to launch the Web UI:

```bash
python src/webui.py
```

## 3. Training (Training Tab)

### 3.1 Select Pre-trained Model

Choose a suitable pre-trained model from the Protein Language Model dropdown. It is recommended to start with ESM2-8M, which has lower computational cost and is suitable for beginners.

### 3.2 Select Dataset

In the Dataset Configuration section, select the Demo_Solubility dataset (default option). Click the Preview Dataset button to preview the dataset content.

### 3.3 Set Task Parameters

- Problem Type, Number of Labels, and Metrics options will be automatically filled when selecting a Pre-defined Dataset.

- For Batch Processing Mode, it is recommended to select Batch Token Mode to avoid uneven batch processing due to high variance in protein sequence lengths.

- Batch Token is recommended to be set to 4000. If you encounter CUDA memory errors, you can reduce this value accordingly.

### 3.4 Choose Training Method

In the Training Parameters section:

- Training Method is a key selection. This Demo dataset does not currently support the SES-Adapter method (due to lack of structural sequence information).

- You can choose the Freeze method to only fine-tune the classification head, or use the LoRA method for efficient parameter fine-tuning.

### 3.5 Start Training

- Click Preview Command to preview the command line script.

- Click Start to begin training. The Web interface will display model statistics and real-time training monitoring.

- After training is complete, the interface will show the model's Metrics on the test set to evaluate model performance.

## 4. Evaluation (Evaluation Tab)

### 4.1 Select Model Path

In the **Model Path** option, enter the path of the trained model (under the `ckpt` root directory). Ensure that the selected **PLM** and **method** are consistent with those used during training.

### 4.2 Evaluation Dataset Loading Rules

- The evaluation system will automatically load the test set of the corresponding dataset.
- If the test set cannot be found, data will be loaded in the order of **validation set → training set**.
- For custom datasets uploaded to Hugging Face:
  - **If only a single CSV file is uploaded**, the evaluation system will automatically load that file, regardless of naming.
  - **If training, validation, and test sets are uploaded**, please ensure accurate file naming.

### 4.3 Start Evaluation

Click **Start Evaluation** to begin the evaluation.

> **Example Model**  
> This project provides a model **demo_provided.pt** that has already been trained on the **Demo_Solubility** dataset using the **Freeze** method, which can be used directly for evaluation.

## 5. Prediction (Prediction Tab)

### 5.1 Single Sequence Prediction

Enter a single amino acid sequence to directly predict its solubility.

### 5.2 Batch Prediction

- By uploading a CSV file, you can predict the solubility of proteins in batch and download the results (in CSV format).

## 6. Download (Download Tab)

For detailed instructions and examples regarding the **Download Tab**, please refer to the **Download** section in the **Manual Tab**.