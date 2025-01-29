import random
import numpy as np
import torch
from typing import List, Dict, Any

def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def min_max_normalize_dataset(train_dataset: List[Dict[str, Any]], 
                            val_dataset: List[Dict[str, Any]], 
                            test_dataset: List[Dict[str, Any]]) -> tuple:
    """Normalize datasets using min-max normalization."""
    # Get all labels from training set
    train_labels = [data['label'] for data in train_dataset]
    
    # Calculate min and max from training set
    min_val = min(train_labels)
    max_val = max(train_labels)
    
    # Normalize all datasets
    for dataset in [train_dataset, val_dataset, test_dataset]:
        for data in dataset:
            data['label'] = (data['label'] - min_val) / (max_val - min_val)
    
    return train_dataset, val_dataset, test_dataset

def check_early_stopping(val_list: List[float], 
                        optimize_func: callable, 
                        patience: int = 10) -> bool:
    """Check if training should stop early."""
    if len(val_list) < patience:
        return False
        
    best_val = optimize_func(val_list[:-patience])
    return all(optimize_func([best_val, val]) == best_val 
              for val in val_list[-patience:]) 