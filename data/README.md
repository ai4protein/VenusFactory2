# Dataset Configuration Format

This document describes the JSON configuration format used for protein localization datasets.

## Configuration Structure

Each dataset is configured using a JSON file with the following structure:

## Fields Description

| Field | Description | Example Values |
|-------|-------------|----------------|
| `dataset` | HuggingFace dataset path | `"tyang816/DeepLocMulti_ESMFold"` |
| `pdb_type` | Type of protein structure prediction | `"ESMFold"`, `"AlphaFold2"` |
| `num_labels` | Number of classification labels | `10` |
| `problem_type` | Type of machine learning problem | `"single_label_classification"` |
| `metrics` | Evaluation metric | `"accuracy"` |
| `monitor` | Metric to monitor during training | `"accuracy"` |
| `normalize` | Normalization method | `"None"` |

## Usage

Place your configuration files in the `data/DeepLocMulti/` directory with the naming convention `DeepLocMulti_[ModelType]_HF.json`, where `[ModelType]` represents the structure prediction model used (e.g., ESMFold, AlphaFold2).

## Notes

- All datasets are hosted on HuggingFace
- Currently supports single-label classification tasks
- Accuracy is used as both the evaluation and monitoring metric
- No normalization is applied by default