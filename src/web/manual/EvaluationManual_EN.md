# VenusFactory Evaluation Module User Guide

## 1. Introduction

The VenusFactory evaluation module is a powerful tool that allows you to comprehensively assess the performance of trained protein analysis models. Through this module, you can test a model's predictive capabilities on various datasets, obtain detailed evaluation metrics, and analyze the model's strengths and weaknesses. Evaluation results can help you compare the performance of different models, select the most suitable model for specific tasks, and guide further model improvements.

The evaluation module supports the assessment of various model fine-tuning approaches. You can use predefined datasets or custom datasets for evaluation and select multiple evaluation metrics to gain a comprehensive understanding of model performance.

## 2. Supported Evaluation Metrics

VenusFactory provides multiple evaluation metrics to assess model performance. Different metrics are applicable depending on the problem type.

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

## 3. Evaluation Interface Details

The evaluation interface is divided into several main sections, each containing specific configuration options. Below is a detailed description of the functionality and settings for each section.

![Model_Dataset_Config](/img/Eval/Model_Dataset_Config.png)

### 3.1 Model and Dataset Configuration

#### Model Path and Protein Language Model Selection
- **Model Path**: Enter the path to the trained model file
  - Usually a model file saved during the training process (e.g., `ckpt/model.pt`)
  - Can be a relative or absolute path
  - Ensure the path is correct, otherwise evaluation will not start properly

- **Protein Language Model**: Select a pre-trained model from the dropdown menu
  - Must be the same pre-trained model used during model training
  - This ensures consistency in model architecture

#### Evaluation Method and Pooling Method
- **Evaluation Method**: Choose a method based on model architecture
  - `freeze`: Freeze the pre-trained model, only train the classifier
  - `full`: Full parameter fine-tuning, train all parameters
  - `plm-lora`: Use LoRA (Low-Rank Adaptation) method for training, reducing parameter count
  - `dora`: Use DoRA (Weight-Decomposed Low-Rank Adaptation) method for training
  - `adalora`: Use AdaLoRA (Adaptive Low-Rank Adaptation) method for training
  - `ia3`: Use IA³ (Infused Adapter by Inhibiting and Amplifying Inner Activations) method for training
  - `plm-qlora`: Use QLoRA (Quantized Low-Rank Adaptation) method for training, reducing memory requirements
  - `ses-adapter`: Use structure-enhanced adapter for training, integrating sequence and structure information
  - **Must be the same method used during model training, otherwise evaluation will fail**

- **Pooling Method**: Select a pooling method
  - `mean`: Mean pooling
    - Calculates the average of all token representations
    - Computationally efficient, suitable for most tasks
  - `attention1d`: Attention pooling
    - Uses attention mechanism to weight-average token representations
    - May provide better performance but at higher computational cost
  - `light_attention`: Lightweight attention pooling
    - Simplified version of attention pooling
    - Balances performance and computational efficiency
  - **Must be the same method used during model training**

#### Dataset Selection
- **Dataset Selection**: Choose the dataset source
  - **Use Pre-defined Dataset**: Use system-predefined datasets
    - **Evaluation Dataset**: Select a dataset from the dropdown menu
    - The system automatically loads the problem type, number of labels, and evaluation metrics for the dataset
    - Suitable for quick evaluation and standard benchmarking
  - **Use Custom Dataset**: Use custom datasets (see the "7.4 Upload Dataset to Hugging Face" section in the Training Module User Guide for details)
    - **Custom Dataset Path**: Enter the Hugging Face dataset path (format: `username/dataset_name`)
    - Requires manual setting of problem type, number of labels, and evaluation metrics
    - Suitable for evaluating model performance on custom data

#### Dataset Preview
- **Preview Dataset**: Click this button to preview the selected dataset
  - Displays dataset statistics: number of samples in training, validation, and test sets
  - Shows dataset examples: including sequences and labels
  - Helps verify that the dataset is loaded correctly
  - Allows you to view data format and content to ensure compatibility with the model

#### Problem Type and Labels
- **Problem Type**: Select the problem type
  - `single_label_classification`: Single-label classification problem
    - Each sample belongs to only one category
    - Labels are typically integer values representing category indices
  - `multi_label_classification`: Multi-label classification problem
    - Each sample may belong to multiple categories
    - Labels are typically comma-separated category index strings
  - `regression`: Regression problem
    - Predicts continuous values
    - Labels are typically floating-point numbers
  - **Must be the same problem type used during model training**

- **Number of Labels**: Set the number of labels (for classification problems)
  - For single-label classification, represents the total number of categories
  - For multi-label classification, represents the total number of possible labels
  - For regression problems, set to 1
  - **Must be the same number of labels used during model training**

#### Evaluation Metrics
- **Metrics**: Select evaluation metrics
  - Multiple metrics can be selected
  - Common metrics include: `Accuracy,MCC,F1,Precision,Recall,AUROC,F1max,Spearman_corr,MSE`
  - Choose appropriate metrics based on problem type:
    - Classification problems: `Accuracy,MCC,F1,Precision,Recall,AUROC`
    - Regression problems: `MSE,Spearman_corr`
  - Selecting multiple metrics provides a comprehensive evaluation of model performance

### 3.2 Structure Sequence Configuration (only applicable to ses-adapter method)

- **Structure Sequence**: Select structure sequence types
  - `foldseek_seq`: Use structure sequences generated by FoldSeek
    - Sequence representation based on protein 3D structure
    - Encoding containing structural information
  - `ss8_seq`: Use 8-class secondary structure sequences
    - Represents protein secondary structure elements (e.g., α-helices, β-sheets, etc.)
    - Provides information about local protein structure
  - Multiple types can be selected simultaneously to enhance structural information representation
  - **Must be the same structure sequence types used during model training**

### 3.3 Batch Processing Configuration

- **Batch Processing Mode**: Select batch processing mode
  - **Batch Size Mode**: Fixed batch size
    - **Batch Size**: Set the number of samples per batch, default is 16
    - Suitable for datasets with similar sequence lengths
    - Larger batches can speed up evaluation but require more memory
  - **Batch Token Mode**: Fixed token count
    - **Tokens per Batch**: Set the number of tokens per batch, default is 10000
    - Suitable for datasets with varying sequence lengths
    - Automatically adjusts the number of samples per batch to ensure the total token count is close to the set value
    - Helps optimize memory usage and processing efficiency

### 3.4 Evaluation Control and Output

- **Preview Command**: Preview the evaluation command to be executed
  - Displays the complete command line arguments when clicked
  - Helps you understand the specific parameters used in the evaluation process
  - Used to verify that all settings are correct

- **Start Evaluation**: Begin the evaluation process
  - Starts model evaluation when clicked
  - Progress bar and status information are displayed during evaluation

- **Abort**: Terminate the current evaluation process
  - Can stop an ongoing evaluation at any time
  - Useful when evaluation takes too long or when incorrect settings are discovered

- **Evaluation Status & Results**: Display evaluation progress and results
  - Evaluation progress: current stage, progress percentage, time elapsed, number of samples processed
  - Evaluation results: various evaluation metrics and their values
  - Presented in table format for clarity and ease of understanding

- **Download CSV**: Download detailed evaluation metrics in CSV format
  - Visible after evaluation is complete
  - Contains all calculated evaluation metrics
  - Can be used for further analysis or comparison with other model results

## 4. Evaluation Process Guide

Below is a complete guide to using the VenusFactory evaluation module, from model preparation to result analysis.

### 4.1 Preparing Models and Datasets

1. **Prepare the trained model file**
   - Ensure you have a model file generated through the training module (e.g., `ckpt/model.pt`)
   - Record the pre-trained model, training method, and pooling method used during training
   - Ensure the model file path is accessible

2. **Select an evaluation dataset**
   - You can use the same dataset as training to evaluate model performance on the test set
   - You can also use a new dataset to evaluate the model's generalization ability
   - Ensure the dataset format is compatible with the training dataset

### 4.2 Configuring Evaluation Parameters

1. **Set up the model and pre-trained model**
   - Enter the model file path in **Model Path**
   - Select the same pre-trained model used during training from the **Protein Language Model** dropdown menu
   - Ensure both match, otherwise architectural incompatibility may occur

2. **Select evaluation method and pooling method**
   - Choose the same method used during training in **Evaluation Method**
   - Choose the same pooling method used during training in **Pooling Method**
   - These settings must be consistent with training to ensure correct loading of model weights

3. **Select dataset**
   - If using a predefined dataset:
     - Select **Use Pre-defined Dataset**
     - Choose a dataset from the dropdown menu
     - The system will automatically load relevant configurations
   - If using a custom dataset:
     - Select **Use Custom Dataset**
     - Enter the Hugging Face dataset path (format: `username/dataset_name`)
     - Manually set the problem type, number of labels, and evaluation metrics

4. **Preview dataset**
   - Click the **Preview Dataset** button to view dataset statistics and examples
   - Confirm that the dataset format is correct
   - Check sample count and distribution
   - Verify that the label format matches the problem type

5. **Set problem type and labels**
   - Set the same **Problem Type** used during training
   - Set the same **Number of Labels** used during training
   - These settings must be consistent with training to ensure model output layer compatibility

6. **Select evaluation metrics**
   - Enter evaluation metrics in **Metrics**, separated by commas
   - Classification problems: recommended to use `accuracy,mcc,f1,precision,recall,auroc`
   - Regression problems: recommended to use `mse,spearman_corr`
   - Selecting multiple metrics provides a comprehensive evaluation of model performance

7. **Configure structure sequences (if applicable)**
   - If using the `ses-adapter` method, select the same **Structure Sequence** types used during training
   - You can select `foldseek_seq`, `ss8_seq`, or both
   - Ensure the dataset contains the corresponding structure sequence fields

8. **Set batch processing parameters**
   - Select **Batch Processing Mode**:
     - **Batch Size Mode**: Suitable for datasets with similar sequence lengths
     - **Batch Token Mode**: Suitable for datasets with varying sequence lengths
   - Set batch size or token count
     - Larger batches can speed up evaluation but require more memory
     - If you encounter out-of-memory errors, try reducing the batch size

### 4.3 Previewing and Executing Evaluation

1. **Preview evaluation command**
   - Click the **Preview Command** button to view the evaluation command
   - Check that all parameters are set correctly
   - Confirm that the command includes all necessary parameters
   - This step helps identify potential configuration errors

2. **Start evaluation**
   - Click the **Start Evaluation** button to begin evaluation
   - Observe the evaluation progress bar to understand current progress
   - During evaluation, you can view the number of samples processed and time elapsed
   - Depending on dataset size and model complexity, evaluation may take from a few minutes to several hours

3. **Monitor the evaluation process**
   - Observe the progress bar and status information
   - If evaluation is too slow, consider increasing the batch size
   - If you encounter memory errors, try reducing the batch size

4. **Abort evaluation (if needed)**
   - If you encounter errors or need to stop evaluation due to incorrect parameters, click the **Abort** button

### 4.4 Analyzing Evaluation Results

1. **View evaluation metrics**
   - After evaluation is complete, review the evaluation metrics table
   - For classification problems, focus on `Accuracy`, `F1`, `Precision`, `Recall`, `MCC`, etc.
   - For regression problems, focus on `MSE`, `Spearman_corr`, etc.
   - Analyze various metrics to understand the model's strengths and weaknesses

2. **Download detailed results**
   - Click the **Download CSV** button to download detailed evaluation results
   - The CSV file contains all calculated evaluation metrics
   - Can be imported into Excel or other tools for further analysis
   - Useful for comparing with results from other models

3. **Result interpretation and decision-making**
   - Based on evaluation results, determine if model performance meets requirements
   - If performance is not satisfactory, consider adjusting training parameters or using different model architectures