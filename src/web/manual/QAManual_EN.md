# ProFactory Frequently Asked Questions (FAQ)

## Installation and Environment Configuration Issues

### Q1: How to properly install ProFactory?

**Answer**: The steps to install ProFactory are as follows:

1. Ensure your system meets the basic requirements:
   - Python 3.8 or higher
   - CUDA-supported GPU (for training large models)

2. Clone the repository and install dependencies:
   ```bash
   git clone https://github.com/yourusername/ProFactory.git
   cd ProFactory
   pip install -r requirements.txt
   ```

3. If you encounter dependency conflicts, it is recommended to use a virtual environment:
   ```bash
   python -m venv profactory_env
   source profactory_env/bin/activate  # Linux/Mac
   # or
   profactory_env\Scripts\activate  # Windows
   pip install -r requirements.txt
   ```

### Q2: What should I do if I encounter the error "Could not find a specific dependency" during installation?

**Answer**: There are several solutions for this situation:

1. Try installing the problematic dependency individually:
   ```bash
   pip install name_of_the_problematic_library
   ```

2. If it is a CUDA-related library, ensure you have installed a PyTorch version compatible with your CUDA version:
   ```bash
   # For example, for CUDA 11.7
   pip install torch==2.0.0+cu117 -f https://download.pytorch.org/whl/torch_stable.html
   ```

3. For some special libraries, you may need to install system dependencies first. For example, on Ubuntu:
   ```bash
   sudo apt-get update
   sudo apt-get install build-essential
   ```

### Q3: How can I check if my CUDA is installed correctly?

**Answer**: You can verify if CUDA is installed correctly by the following methods:

1. Check the CUDA version:
   ```bash
   nvidia-smi
   ```

2. Verify if PyTorch can recognize CUDA in Python:
   ```python
   import torch
   print(torch.cuda.is_available())  # Should return True
   print(torch.cuda.device_count())  # Displays the number of GPUs
   print(torch.cuda.get_device_name(0))  # Displays the GPU name
   ```

3. If PyTorch cannot recognize CUDA, ensure you have installed the matching versions of PyTorch and CUDA.

## Hardware and Resource Issues

### Q4: What should I do if I encounter a "CUDA out of memory" error during runtime?

**Answer**: This error indicates that your GPU memory is insufficient. Solutions include:

1. **Reduce the batch size**: This is the most direct and effective method. Reduce the batch size in the training configuration by half or more.

2. **Use a smaller model**: Choose a pre-trained model with fewer parameters, such as switching from ProtBERT to ESM-1b.

3. **Enable gradient accumulation**: Increase the `gradient_accumulation_steps` parameter value, for example, set it to 2 or 4, which can reduce memory usage without decreasing the effective batch size.

4. **Use mixed precision training**: Enable the `fp16` option in the training options, which can significantly reduce memory usage.

5. **Reduce the maximum sequence length**: If your data allows, you can decrease the `max_seq_length` parameter.

### Q5: How can I determine what batch size I should use?

**Answer**: Determining the appropriate batch size requires balancing memory usage and training effectiveness:

1. **Start small and gradually increase**: Begin with smaller values (like 4 or 8) and gradually increase until memory is close to its limit.

2. **Refer to benchmarks**: For common protein models, most studies use a batch size of 16-64, but this depends on your GPU memory and sequence length.

3. **Monitor the training process**: A larger batch size may make each training iteration more stable but may require a higher learning rate.

4. **Rule of thumb for memory issues**: If you encounter memory errors, first try halving the batch size.

## Dataset Issues

### Q6: How do I prepare a custom dataset?

**Answer**: Preparing a custom dataset requires the following steps:

1. **Format the data**: The data should be organized into a CSV file, containing at least the following columns:
   - `sequence`: The protein sequence, represented using standard amino acid letters
   - Label column: Depending on your task type, this can be numerical (regression) or categorical (classification)

2. **Split the data**: Prepare training, validation, and test sets, such as `train.csv`, `validation.csv`, and `test.csv`.

3. **Upload to Hugging Face**:
   - Create a dataset repository on Hugging Face
   - Upload your CSV file
   - Reference it in ProFactory using the `username/dataset_name` format

4. **Create dataset configuration**: The configuration should include the problem type (regression or classification), number of labels, and evaluation metrics.

### Q7: What should I do if I encounter a format error when importing my dataset?

**Answer**: Common format issues and their solutions:

1. **Incorrect column names**: Ensure the CSV file contains the necessary columns, especially the `sequence` column and label column.

2. **Sequence format issues**:
   - Ensure the sequence contains only valid amino acid letters (ACDEFGHIKLMNPQRSTVWY)
   - Remove spaces, line breaks, or other illegal characters from the sequence
   - Check if the sequence length is within a reasonable range

3. **Encoding issues**: Ensure the CSV file is saved with UTF-8 encoding.

4. **CSV delimiter issues**: Ensure the file uses the correct delimiter (usually a comma). You can use a text editor to view and correct it.

5. **Handling missing values**: Ensure there are no missing values in the data, or handle them appropriately.

### Q8: My dataset is large, and the system loads slowly or crashes. What should I do?

**Answer**: For large datasets, you can:

1. **Reduce the dataset size**: If possible, test your method with a subset of the data first.

2. **Increase data loading efficiency**:
   - Use the `batch_size` parameter to control the amount of data loaded at a time
   - Enable data caching to avoid repeated loading
   - Preprocess the data to reduce file size (e.g., remove unnecessary columns)

3. **Dataset sharding**: Split large datasets into multiple smaller files and process them one by one.

4. **Increase system resources**: If possible, increase RAM or use a server with more memory.

## Training Issues

### Q9: How can I recover if the training suddenly interrupts?

**Answer**: Methods to handle training interruptions:

1. **Check checkpoints**: The system periodically saves checkpoints (usually in the `ckpt` directory). You can recover from the most recent checkpoint:
   - Look for the last saved model file (usually named `checkpoint-X`, where X is the step number)
   - Specify the checkpoint path as the starting point in the training options

2. **Use the checkpoint recovery feature**: Enable the checkpoint recovery option in the training configuration.

3. **Save checkpoints more frequently**: Adjust the frequency of saving checkpoints, for example, save every 500 steps instead of the default every 1000 steps.

### Q10: How can I speed up training if it is very slow?

**Answer**: Methods to speed up training:

1. **Hardware aspects**:
   - Use a more powerful GPU
   - Use multi-GPU training (if supported)
   - Ensure data is stored on an SSD rather than an HDD

2. **Parameter settings**:
   - Use mixed precision training (enable the fp16 option)
   - Increase the batch size (if memory allows)
   - Reduce the maximum sequence length (if the task allows)
   - Decrease validation frequency (the `eval_steps` parameter)

3. **Model selection**:
   - Choose a smaller pre-trained model
   - Use parameter-efficient fine-tuning methods (like LoRA)

### Q11: What does it mean if the loss value does not decrease or if NaN values appear during training?

**Answer**: This usually indicates that there is a problem with the training:

1. **Reasons for loss not decreasing and solutions**:
   - **Learning rate too high**: Try reducing the learning rate, for example, from 5e-5 to 1e-5
   - **Optimizer issues**: Try different optimizers, such as switching from Adam to AdamW
   - **Initialization issues**: Check the model initialization settings
   - **Data issues**: Validate if the training data has outliers or label errors

2. **Reasons for NaN values and solutions**:
   - **Gradient explosion**: Add gradient clipping, set the `max_grad_norm` parameter
   - **Learning rate too high**: Significantly reduce the learning rate
   - **Numerical instability**: This may occur when using mixed precision training; try disabling the fp16 option
   - **Data anomalies**: Check if there are extreme values in the input data

### Q12: What is overfitting, and how can it be avoided?

**Answer**: Overfitting refers to a model performing well on training data but poorly on new data. Methods to avoid overfitting include:

1. **Increase the amount of data**: Use more training data or data augmentation techniques.

2. **Regularization methods**:
   - Add dropout (usually set to 0.1-0.3)
   - Use weight decay
   - Early stopping: Stop training when the validation performance no longer improves

3. **Simplify the model**:
   - Use fewer layers or smaller hidden dimensions
   - Freeze some layers of the pre-trained model (using the freeze method)

4. **Cross-validation**: Use k-fold cross-validation to obtain a more robust model.

## Evaluation Issues

### Q13: How do I interpret evaluation metrics? Which metric is the most important?

**Answer**: Different tasks focus on different metrics:

1. **Classification tasks**:
   - **Accuracy**: The proportion of correct predictions, suitable for balanced datasets
   - **F1 Score**: The harmonic mean of precision and recall, suitable for imbalanced datasets
   - **MCC (Matthews Correlation Coefficient)**: A comprehensive measure of classification performance, more robust to class imbalance
   - **AUROC (Area Under the ROC Curve)**: Measures the model's ability to distinguish between different classes

2. **Regression tasks**:
   - **MSE (Mean Squared Error)**: The sum of the squared differences between predicted and actual values, the smaller the better
   - **RMSE (Root Mean Squared Error)**: The square root of MSE, in the same units as the original data
   - **MAE (Mean Absolute Error)**: The average of the absolute differences between predicted and actual values
   - **R² (Coefficient of Determination)**: Measures the proportion of variance explained by the model, the closer to 1 the better

3. **Most important metric**: Depends on your specific application needs. For example, in drug screening, you may focus more on true positive rates; for structural prediction, you may focus more on RMSE.

### Q14: What should I do if the evaluation results are poor?

**Answer**: Common strategies to improve model performance:

1. **Data quality**:
   - Check for errors or noise in the data
   - Increase the number of training samples
   - Ensure the training and test set distributions are similar

2. **Model adjustments**:
   - Try different pre-trained models
   - Adjust hyperparameters like learning rate and batch size
   - Use different fine-tuning methods (full parameter fine-tuning, LoRA, etc.)

3. **Feature engineering**:
   - Add structural information (e.g., using foldseek features)
   - Consider sequence characteristics (e.g., hydrophobicity, charge, etc.)

4. **Ensemble methods**:
   - Train multiple models and combine results
   - Use cross-validation to obtain a more robust model

### Q15: Why does my model perform much worse on the test set than on the validation set?

**Answer**: Common reasons for decreased performance on the test set:

1. **Data distribution shift**:
   - The training, validation, and test set distributions are inconsistent
   - The test set contains protein families or features not seen during training

2. **Overfitting**:
   - The model overfits the validation set because it was used for model selection
   - Increasing regularization or reducing the number of training epochs may help

3. **Data leakage**:
   - Unintentionally leaking test data information into the training process
   - Ensure data splitting is done before preprocessing to avoid cross-contamination

4. **Randomness**:
   - If the test set is small, results may be influenced by randomness
   - Try training multiple models with different random seeds and averaging the results

## Prediction Issues

### Q16: How can I speed up the prediction process?

**Answer**: Methods to speed up predictions:

1. **Batch prediction**: Use batch prediction mode instead of single-sequence prediction, which can utilize the GPU more efficiently.

2. **Reduce computation**:
   - Use a smaller model or a more efficient fine-tuning method
   - Reduce the maximum sequence length (if possible)

3. **Hardware optimization**:
   - Use a faster GPU or CPU
   - Ensure predictions are done on the GPU rather than the CPU

4. **Model optimization**:
   - Try model quantization (e.g., int8 quantization)
   - Exporting to ONNX format may provide faster inference speeds

### Q17: What could be the reason for the prediction results being significantly different from expectations?

**Answer**: Possible reasons for prediction discrepancies:

1. **Data mismatch**:
   - The sequences being predicted differ from the training data distribution
   - There are significant differences in sequence length, composition, or structural features

2. **Model issues**:
   - The model is under-trained or overfitted
   - An unsuitable pre-trained model was chosen for the task

3. **Parameter configuration**:
   - Ensure the parameters used during prediction (like maximum sequence length) are consistent with those used during training
   - Check if the correct problem type (classification/regression) is being used

4. **Data preprocessing**:
   - Ensure the prediction data undergoes the same preprocessing steps as the training data
   - Check if the sequence format is correct (standard amino acid letters, no special characters)

### Q18: How can I batch predict a large number of sequences?

**Answer**: Steps for efficient batch prediction:

1. **Prepare the input file**:
   - Create a CSV file containing all sequences
   - The file must include a `sequence` column
   - Optionally include an ID or other identifier columns

2. **Use the batch prediction feature**:
   - Go to the prediction tab
   - Select "Batch Prediction" mode
   - Upload the sequence file
   - Set an appropriate batch size (usually 16-32 is a good balance)

3. **Optimize settings**:
   - Increasing the batch size can improve throughput (if memory allows)
   - Reducing unnecessary feature calculations can speed up processing

4. **Result handling**:
   - After prediction is complete, the system will generate a CSV file containing the original sequences and prediction results
   - You can download this file for further analysis

## Model and Result Issues

### Q19: Which pre-trained model should I choose?

**Answer**: Model selection recommendations:

1. **For general tasks**:
   - ESM-2 is suitable for various protein-related tasks, balancing performance and efficiency
   - ProtBERT performs well on certain sequence classification tasks

2. **Considerations**:
   - **Data volume**: When data is limited, a smaller model may be better (to avoid overfitting)
   - **Sequence length**: For long sequences, consider models that support longer contexts
   - **Computational resources**: When resources are limited, choose smaller models or parameter-efficient methods
   - **Task type**: Different models have their advantages in different tasks

3. **Recommended strategy**: If conditions allow, try several different models and choose the one that performs best on the validation set.

### Q20: How do I interpret the loss curve during training?

**Answer**: Guidelines for interpreting the loss curve:

1. **Ideal curve**:
   - Both training loss and validation loss decrease steadily
   - The two curves eventually stabilize and converge
   - The validation loss stabilizes near its lowest point

2. **Common patterns and their meanings**:
   - **Training loss continues to decrease while validation loss increases**: Signal of overfitting; consider increasing regularization
   - **Both losses stagnate at high values**: Indicates underfitting; may need a more complex model or longer training
   - **Curve fluctuates dramatically**: The learning rate may be too high; consider lowering it
   - **Validation loss is lower than training loss**: This may indicate a data splitting issue or batch normalization effect

3. **Adjusting based on the curve**:
   - If validation loss stops improving early, consider early stopping
   - If training loss decreases very slowly, try increasing the learning rate
   - If there are sudden jumps in the curve, check for data issues or learning rate scheduling

### Q21: How do I save and share my model?

**Answer**: Guidelines for saving and sharing models:

1. **Local saving**:
   - After training is complete, the model will be automatically saved in the specified output directory
   - The complete model includes model weights, configuration files, and tokenizer information

2. **Important files**:
   - `pytorch_model.bin`: Model weights
   - `config.json`: Model configuration
   - `special_tokens_map.json` and `tokenizer_config.json`: Tokenizer configuration

3. **Sharing the model**:
   - **Hugging Face Hub**: The easiest way is to upload to Hugging Face
     - Create a model repository
     - Upload your model files
     - Add model descriptions and usage instructions in the readme
   
   - **Local export**: You can also compress the model folder and share it
     - Ensure all necessary files are included
     - Provide environment requirements and usage instructions

4. **Documentation**: Regardless of the sharing method, you should provide:
   - Description of the training data
   - Model architecture and parameters
   - Performance metrics
   - Usage examples

## Interface and Operation Issues

### Q22: What should I do if the interface loads slowly or crashes?

**Answer**: Solutions for interface issues:

1. **Browser-related**:
   - Try using different browsers (Chrome usually has the best compatibility)
   - Clear browser cache and cookies
   - Disable unnecessary browser extensions

2. **Resource issues**:
   - Ensure the system has enough memory
   - Close other resource-intensive programs
   - If running on a remote server, check the server load

3. **Network issues**:
   - Ensure the network connection is stable
   - If using through an SSH tunnel, check if the connection is stable

4. **Restart services**:
   - Try restarting the Gradio service
   - In extreme cases, restart the server

### Q23: Why does my training stop responding midway?

**Answer**: Possible reasons and solutions for training stopping responding:

1. **Resource exhaustion**:
   - Insufficient system memory
   - GPU memory overflow
   - Solution: Reduce batch size, use more efficient training methods, or increase system resources

2. **Process termination**:
   - The system's OOM (Out of Memory) killer terminated the process
   - Server timeout policies may terminate long-running processes
   - Solution: Check system logs, use tools like screen or tmux to run in the background, reduce resource usage

3. **Network or interface issues**:
   - Browser crashes or network disconnections
   - Solution: Run training via command line, or ensure a stable network connection

4. **Data or code issues**:
   - Anomalies or incorrect formats in the dataset causing processing to hang
   - Solution: Check the dataset, and test the process with a small subset of data