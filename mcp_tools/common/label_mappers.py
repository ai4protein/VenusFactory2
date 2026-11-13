"""Label mapping functions for prediction results."""

import json
import pandas as pd
from typing import Dict, Optional, Any

from .constants import (
    DATASET_TO_TASK_MAP,
    REGRESSION_TASKS_FUNCTION,
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
                # Display the inference value unchanged in scale. Only a
                # verified checkpoint transform can restore target units.
                return round(scaled_value, 2)
            except (ValueError, TypeError):
                return scaled_value
        return scaled_value

    # Handle SortingSignal special case
    if row.get('Dataset') == 'SortingSignal':
        # Try to get prediction from either 'prediction' or 'predicted_class' column
        predictions_str = row.get('prediction', row.get('predicted_class'))
        if predictions_str is not None:
            try:
                # Parse JSON if it's a string, otherwise use directly
                predictions = json.loads(predictions_str) if isinstance(predictions_str, str) else predictions_str
                # Handle list of predictions
                if isinstance(predictions, list):
                    if all(p == 0 for p in predictions):
                        return "No signal"
                    signal_labels = LABEL_MAPPING_FUNCTION["Sortingsignal"]
                    active_labels = [signal_labels[i] for i, pred in enumerate(predictions) if pred == 1]
                    return "_".join(active_labels) if active_labels else "None"
            except Exception as e:
                print(f"Error parsing SortingSignal prediction: {e}, value: {predictions_str}")
                pass

    # Handle classification tasks
    labels_key = ("DeepLocMulti" if row.get('Dataset') == "DeepLocMulti"
                 else "DeepLocBinary" if row.get('Dataset') == "DeepLocBinary"
                 else current_task)
    labels = LABEL_MAPPING_FUNCTION.get(labels_key) or LABEL_MAPPING_FUNCTION.get(
        DATASET_TO_TASK_MAP.get(row.get("Dataset", ""), current_task)
    )

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
        regression_tasks_max_min: Deprecated compatibility argument; ignored.
            Target transforms belong to checkpoint inference, not label formatting.
    
    Returns:
        Mapped label as string
    """
    # Retain the old argument for caller compatibility, but never infer a
    # checkpoint's target transform from a task name or display-time range.
    if current_task in REGRESSION_TASKS_FUNCTION:
        value = row.get("prediction", row.get("predicted_class"))
        if pd.notna(value) and value != 'N/A':
            try:
                return str(round(float(value), 3))
            except (ValueError, TypeError):
                pass
        return str(value)

    # Handle multi-label classification tasks
    if row.get('Dataset') == 'SortingSignal':
        predictions_str = row['predicted_class']
        predictions = json.loads(predictions_str)
        if all(p == 0 for p in predictions):
            return "No signal"
        # Get labels for SortingSignal
        signal_labels = LABEL_MAPPING_FUNCTION["Sortingsignal"]
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
    labels = LABEL_MAPPING_FUNCTION.get(labels_key) or LABEL_MAPPING_FUNCTION.get(
        DATASET_TO_TASK_MAP.get(row.get("Dataset", ""), current_task)
    )

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
