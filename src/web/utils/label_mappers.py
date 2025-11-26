"""Label mapping functions for prediction results."""

import json
import pandas as pd
from typing import Dict, Optional, Any

from .constants import (
    DATASET_TO_TASK_MAP,
    REGRESSION_TASKS_FUNCTION,
    REGRESSION_TASKS_FUNCTION_MAX_MIN,
    LABEL_MAPPING_FUNCTION,
)


def map_labels(row: pd.Series, task: str) -> Any:
    """
    Map prediction labels to human-readable text for function prediction results.
    
    Args:
        row: DataFrame row containing prediction data
        task: Task name (e.g., "Solubility", "Subcellular Localization")
    
    Returns:
        Mapped label value (string or numeric)
    """
    current_task = DATASET_TO_TASK_MAP.get(row.get('Dataset', ''), task)
    
    # Handle regression tasks
    if current_task in REGRESSION_TASKS_FUNCTION: 
        scaled_value = row.get("prediction")
        if pd.notna(scaled_value) and scaled_value != 'N/A':
            try:
                scaled_value = float(scaled_value)
                if current_task in REGRESSION_TASKS_FUNCTION_MAX_MIN:
                    min_val, max_val = REGRESSION_TASKS_FUNCTION_MAX_MIN[current_task]
                    original_value = scaled_value * (max_val - min_val) + min_val
                    return round(original_value, 2)
                else:
                    return round(scaled_value, 2)
            except (ValueError, TypeError):
                return scaled_value
        return scaled_value

    # Handle SortingSignal special case
    if row.get('Dataset') == 'SortingSignal':
        predictions_str = row.get('predicted_class')
        if predictions_str:
            try:
                predictions = json.loads(predictions_str) if isinstance(predictions_str, str) else predictions_str
                if all(p == 0 for p in predictions):
                    return "No signal"
                signal_labels = ["CH", 'GPI', "MT", "NES", "NLS", "PTS", "SP", "TM", "TH"]
                active_labels = [signal_labels[i] for i, pred in enumerate(predictions) if pred == 1]
                return "_".join(active_labels) if active_labels else "None"
            except Exception:
                pass
    
    # Handle classification tasks
    labels_key = ("DeepLocMulti" if row.get('Dataset') == "DeepLocMulti" 
                 else "DeepLocBinary" if row.get('Dataset') == "DeepLocBinary" 
                 else current_task)
    labels = LABEL_MAPPING_FUNCTION.get(labels_key)
    
    pred_val = row.get("prediction", row.get("predicted_class"))
    if pred_val is None or pred_val == "N/A":
        return "N/A"
        
    try:
        pred_val = int(float(pred_val))
        if labels and 0 <= pred_val < len(labels): 
            return labels[pred_val]
    except (ValueError, TypeError):
        pass
    
    return str(pred_val)


def map_labels_individual(row: pd.Series, current_task: str, regression_tasks_max_min: Optional[Dict] = None) -> str:
    """
    Map prediction labels to text for individual function prediction.
    
    Args:
        row: DataFrame row containing prediction data
        current_task: Current task name
        regression_tasks_max_min: Optional dictionary with regression task min/max values
    
    Returns:
        Mapped label as string
    """
    # Define regression tasks max-min values for denormalization
    if regression_tasks_max_min is None:
        regression_tasks_max_min = {
            "Stability": [40.1995166, 66.8968874],
            "Optimum temperature": [2, 120]
        }
    
    # For regression tasks, return the numeric value with denormalization
    if current_task in REGRESSION_TASKS_FUNCTION: 
        scaled_value = row.get("prediction", row.get("predicted_class"))
        if pd.notna(scaled_value) and scaled_value != 'N/A':
            try:
                scaled_value = float(scaled_value)
                
                # Apply denormalization for specific tasks
                if current_task == "Stability" and current_task in regression_tasks_max_min:
                    min_val, max_val = regression_tasks_max_min["Stability"]
                    original_value = scaled_value * (max_val - min_val) + min_val
                    return f"{round(original_value, 2)}"
                elif current_task == "Optimum temperature" and current_task in regression_tasks_max_min:
                    min_val, max_val = regression_tasks_max_min["Optimum temperature"]
                    original_value = scaled_value * (max_val - min_val) + min_val
                    return f"{round(original_value, 1)}°C"
                else:
                    return f"{round(scaled_value, 3)}"
            
            except (ValueError, TypeError):
                return str(scaled_value)

        return str(scaled_value)

    # Handle multi-label classification tasks
    if row.get('Dataset') == 'SortingSignal':
        predictions_str = row['predicted_class']
        predictions = json.loads(predictions_str)
        if all(p == 0 for p in predictions):
            return "No signal"
        # Get labels for SortingSignal
        signal_labels = ["CH", 'GPI', "MT", "NES", "NLS", "PTS", "SP", "TM", "TH"]
        active_labels = []
        # Find indices where prediction is 1 (active labels)
        for i, pred in enumerate(predictions):
            if pred == 1:
                active_labels.append(signal_labels[i])
        # Return concatenated labels or "None" if no active labels
        return "_".join(active_labels) if active_labels else "None"

    # For regular classification tasks, map to text labels
    labels_key = ("DeepLocMulti" if row.get('Dataset') == "DeepLocMulti" 
                else "DeepLocBinary" if row.get('Dataset') == "DeepLocBinary" 
                else current_task)
    labels = LABEL_MAPPING_FUNCTION.get(labels_key)
    
    pred_val = row.get("predicted_class")
    if pred_val is None or pred_val == "N/A":
        return "N/A"  
    try:
        pred_val = int(float(pred_val))
        if labels and 0 <= pred_val < len(labels): 
            return labels[pred_val]
    except (ValueError, TypeError):
        pass
    
    return str(pred_val)

