import numpy as np
from typing import List, Tuple, Any
from sklearn.preprocessing import StandardScaler, RobustScaler

def min_max_normalize_dataset(train_dataset, val_dataset, test_dataset):
    """Min-max normalization (0-1 scaling)."""
    labels = [e["label"] for e in train_dataset]
    min_label, max_label = min(labels), max(labels)
    normalized_train_dataset = []
    normalized_val_dataset = []
    normalized_test_dataset = []
    for e in train_dataset:
        e["label"] = (e["label"] - min_label) / (max_label - min_label)
        normalized_train_dataset.append(e)
    for e in val_dataset:
        e["label"] = (e["label"] - min_label) / (max_label - min_label)
        normalized_val_dataset.append(e)
    for e in test_dataset:
        e["label"] = (e["label"] - min_label) / (max_label - min_label)
        normalized_test_dataset.append(e)
    print(normalized_train_dataset[0])
    return normalized_train_dataset, normalized_val_dataset, normalized_test_dataset

def standard_normalize_dataset(train_dataset, val_dataset, test_dataset):
    """Z-score normalization (standardization)."""
    train_labels = np.array([e["label"] for e in train_dataset])
    mean_label = np.mean(train_labels)
    std_label = np.std(train_labels)
    normalized_train_dataset = []
    normalized_val_dataset = []
    normalized_test_dataset = []
    for e in train_dataset:
        e["label"] = (e["label"] - mean_label) / std_label
        normalized_train_dataset.append(e)
    for e in val_dataset:
        e["label"] = (e["label"] - mean_label) / std_label
        normalized_val_dataset.append(e)
    for e in test_dataset:
        e["label"] = (e["label"] - mean_label) / std_label
        normalized_test_dataset.append(e)
    return normalized_train_dataset, normalized_val_dataset, normalized_test_dataset

def robust_normalize_dataset(train_dataset, val_dataset, test_dataset):
    """Robust scaling using statistics that are robust to outliers."""
    scaler = RobustScaler()
    train_labels = np.array([e["label"] for e in train_dataset]).reshape(-1, 1)
    scaler.fit(train_labels)
    normalized_train_dataset = []
    normalized_val_dataset = []
    normalized_test_dataset = []
    for e in train_dataset:
        e["label"] = scaler.transform([[e["label"]]])[0][0]
        normalized_train_dataset.append(e)
    for e in val_dataset:
        e["label"] = scaler.transform([[e["label"]]])[0][0]
        normalized_val_dataset.append(e)
    for e in test_dataset:
        e["label"] = scaler.transform([[e["label"]]])[0][0]
        normalized_test_dataset.append(e)
    return normalized_train_dataset, normalized_val_dataset, normalized_test_dataset

def log_normalize_dataset(train_dataset, val_dataset, test_dataset, offset=1.0):
    """Log normalization, useful for skewed data."""
    normalized_train_dataset = []
    normalized_val_dataset = []
    normalized_test_dataset = []
    for e in train_dataset:
        e["label"] = np.log(e["label"] + offset)
        normalized_train_dataset.append(e)
    for e in val_dataset:
        e["label"] = np.log(e["label"] + offset)
        normalized_val_dataset.append(e)
    for e in test_dataset:
        e["label"] = np.log(e["label"] + offset)
        normalized_test_dataset.append(e)
    return normalized_train_dataset, normalized_val_dataset, normalized_test_dataset

def quantile_normalize_dataset(train_dataset, val_dataset, test_dataset, n_quantiles=1000):
    """Quantile normalization to achieve a uniform distribution."""
    from sklearn.preprocessing import QuantileTransformer
    
    transformer = QuantileTransformer(n_quantiles=n_quantiles, output_distribution='uniform')
    train_labels = np.array([e["label"] for e in train_dataset]).reshape(-1, 1)
    transformer.fit(train_labels)
    
    normalized_train_dataset = []
    normalized_val_dataset = []
    normalized_test_dataset = []
    for e in train_dataset:
        e["label"] = transformer.transform([[e["label"]]])[0][0]
        normalized_train_dataset.append(e)
    for e in val_dataset:
        e["label"] = transformer.transform([[e["label"]]])[0][0]
        normalized_val_dataset.append(e)
    for e in test_dataset:
        e["label"] = transformer.transform([[e["label"]]])[0][0]
        normalized_test_dataset.append(e)
    return normalized_train_dataset, normalized_val_dataset, normalized_test_dataset

def normalize_dataset(train_dataset, val_dataset, test_dataset, method='min_max', **kwargs):
    """
    Unified interface for different normalization methods.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset
        method: Normalization method ('min_max', 'standard', 'robust', 'log', 'quantile')
        **kwargs: Additional arguments for specific normalization methods
    
    Returns:
        Normalized datasets (train, val, test)
    """
    normalization_methods = {
        'min_max': min_max_normalize_dataset,
        'standard': standard_normalize_dataset,
        'robust': robust_normalize_dataset,
        'log': log_normalize_dataset,
        'quantile': quantile_normalize_dataset
    }
    
    if method not in normalization_methods:
        raise ValueError(f"Unsupported normalization method: {method}. "
                        f"Available methods: {list(normalization_methods.keys())}")
    
    return normalization_methods[method](train_dataset, val_dataset, test_dataset, **kwargs)

