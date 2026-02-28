# VenusFactory Prediction Module User Guide

## Table of Contents
1. Introduction
2. Overview of the Prediction Interface
3. Single Sequence Prediction
4. Batch Prediction
5. Frequently Asked Questions

## 1. Introduction

The VenusFactory prediction module allows researchers to use trained models to predict the functions of new protein sequences, supporting quick predictions for single sequences and efficient processing for batch sequences, providing important computational assistance for protein function research and drug development. The prediction module is closely integrated with the training module, ensuring consistency between model training and prediction processes, while providing an intuitive user interface that allows biologists to obtain high-quality prediction results without needing to delve into the details of machine learning techniques.

## 2. Overview of the Prediction Interface

The prediction interface is divided into a model configuration area and two main functional tabs: single sequence prediction and batch prediction.

### 2.1 Model Configuration Area

The model configuration area contains all the basic parameter settings required for prediction, which must be consistent with the parameters used during model training:

- **Model Path**: The path to the trained model file, usually the model file saved during the training process (e.g., `ckpt/model.pt`)
- **Protein Language Model**: Selection of the pre-trained protein language model, which must be the same as the model used during training
- **Evaluation Method**: Selection of evaluation methods, including `freeze` (freeze the pre-trained model), `full` (full fine-tuning), `ses-adapter` (structure-enhanced adapter), `plm-lora`, and `plm-qlora` (parameter-efficient fine-tuning methods)
- **Pooling Method**: Selection of pooling methods, including `mean` (mean pooling), `attention1d` (attention pooling), and `light_attention` (lightweight attention pooling)
- **Problem Type**: Selection of problem types, including single-label classification, multi-label classification, and regression
- **Number of Labels**: Setting the number of labels (for classification problems)

When the `ses-adapter` evaluation method is selected, additional structural sequence options will be displayed:
- **Structure Sequences**: Options to select `foldseek_seq` (FoldSeek generated structural sequences) and `ss8_seq` (8-class secondary structure sequences)

### 2.2 Prediction Functional Tabs

The prediction module provides two prediction modes, accessed through different tabs:

- **Sequence Prediction**: Single sequence prediction, suitable for quickly predicting the function of a single protein sequence
- **Batch Prediction**: Batch prediction, suitable for simultaneously predicting the functions of multiple protein sequences

## 3. Single Sequence Prediction

The single sequence prediction function allows users to input a single protein sequence and obtain instant prediction results, suitable for quick validation and exploratory analysis.

### 3.1 Input Sequence

Enter the standard amino acid sequence (using single-letter codes) in the "Amino Acid Sequence" text box. If using the `ses-adapter` method, structural sequence information (FoldSeek sequence and/or secondary structure sequence) must also be entered in the corresponding text boxes.

### 3.2 Execute Prediction

1. Ensure all model configuration parameters are set correctly
2. Click the "Predict" button to start the prediction process
3. The system will display prediction progress and status information
4. To abort the prediction, click the "Abort" button

### 3.3 Display Prediction Results

After the prediction is complete, the results will be displayed in tabular form:

- **Single-label Classification**: Displays the predicted categories and the probability distribution for each category
- **Multi-label Classification**: Displays the prediction results (0/1) and probability values for each label
- **Regression**: Displays the predicted numerical results

## 4. Batch Prediction

The batch prediction function allows users to process multiple protein sequences simultaneously, suitable for large-scale screening and systematic analysis.

### 4.1 Prepare Input File

Batch prediction requires preparing an input file in CSV format, which should contain the following columns:
- `aa_seq` (required): Amino acid sequence
- `id` (optional): Sequence identifier
- `foldseek_seq` (optional, required only for `ses-adapter` method): FoldSeek structural sequence
- `ss8_seq` (optional, required only for `ses-adapter` method): Secondary structure sequence

### 4.2 Upload File and Configure Batch Processing Parameters

1. Click the "Upload CSV File" button to upload the prepared CSV file
2. After uploading, you can preview the file content in the "File Preview" area
3. Set the "Batch Size" parameter to control the number of sequences processed in each batch (default is 8)
   - Larger batches can speed up the prediction process but require more memory/graphics memory
   - For long sequences, it is recommended to use a smaller batch size

### 4.3 Execute Batch Prediction

1. Ensure all model configuration parameters are set correctly
2. Click the "Start Batch Prediction" button to begin batch prediction
3. The system will display a progress bar and status information, including:
   - Total number of sequences
   - Current processing progress
   - Estimated remaining time
4. To abort the prediction, click the "Abort" button

### 4.4 Batch Prediction Results

After the prediction is complete, the results will be displayed in tabular form and provide the following functionalities:
- Summary statistics of results (e.g., predicted category distribution or numerical statistics)
- Complete prediction results table
- "Download Predictions" button for downloading the complete prediction results in CSV format

The downloaded CSV file contains the original sequence information and corresponding prediction results, which can be further processed and visualized using other analysis tools.