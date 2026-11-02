"""Data processing utilities for protein prediction results."""

import re
import json
import numpy as np
import pandas as pd
from typing import Tuple, List
import gradio as gr

from .constants import (
    DATASET_MAPPING_FUNCTION,
    LABEL_MAPPING_FUNCTION,
    REGRESSION_TASKS_FUNCTION,
    REGRESSION_TASKS_FUNCTION_MAX_MIN,
    DATASET_TO_TASK_MAP
)


def get_total_residues_count(df: pd.DataFrame) -> int:
    """Get total number of unique residue positions from mutation data."""
    if 'mutant' not in df.columns:
        return 0
    
    try:
        positions = df['mutant'].str.extract(r'(\d+)').dropna()
        return positions[0].astype(int).nunique() if not positions.empty else 0
    except Exception:
        return 0


def update_dataset_choices(task: str) -> gr.CheckboxGroup:
    """Update dataset choices based on selected task."""
    datasets = DATASET_MAPPING_FUNCTION.get(task, [])
    return gr.update(choices=datasets, value=datasets)


def expand_residue_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """Expand residue-level predictions into per-residue format."""
    expanded_rows = []
    
    for _, row in df.iterrows():
        header = row['header']
        sequence = row['sequence']
        task = row['Task']
        dataset = row['Dataset']
        try:
            predictions = json.loads(row['predicted_class'])
            probabilities = json.loads(row['probabilities']) if isinstance(row['probabilities'], str) else row['probabilities']
            
            if isinstance(predictions[0], list):
                predictions = predictions[0]
            if isinstance(probabilities[0], list):
                probabilities = probabilities[0]
                
            for i, (residue, pred, prob) in enumerate(zip(sequence, predictions, probabilities)):
                if isinstance(prob, list):
                    max_prob = max(prob)
                    predicted_label = prob.index(max_prob)
                else:
                    max_prob = prob
                    predicted_label = pred
                
                expanded_rows.append({
                    'index': i,
                    'residue': residue,
                    'predicted_label': predicted_label,
                    'probability': max_prob,
                })
                
        except Exception as e:
            print(f"Error processing row for {header}: {e}")
            continue
    
    return pd.DataFrame(expanded_rows)


def prepare_top_residue_heatmap_data(df: pd.DataFrame) -> Tuple:
    """Prepare data for heatmap visualization."""
    score_col = next((col for col in df.columns if 'score' in col.lower()), None)
    rank_col = "Prediction Rank"
    if score_col is None:
        return (None,) * 5

    def is_valid_mutant(mutant):
        """Check if mutant string is valid (format: A123B where A != B)."""
        return isinstance(mutant, str) and re.match(r'^[A-Z]\d+[A-Z]$', mutant) and mutant[0] != mutant[-1]
    
    valid_df = df[df['mutant'].apply(is_valid_mutant)].copy()
    
    if valid_df.empty:
        return ([], [], np.array([[]]), np.array([[]]), score_col)

    min_score, max_score = valid_df[score_col].min(), valid_df[score_col].max()
    if max_score == min_score:
        valid_df['scaled_score'] = 0.0
    else:
        valid_df['scaled_score'] = -1 + 2 * (valid_df[score_col] - min_score) / (max_score - min_score)
    
    valid_df['position'] = valid_df['mutant'].str[1:-1].astype(int)

    effect_scores = valid_df.groupby('position')['scaled_score'].mean()
    sorted_positions = effect_scores.sort_values(ascending=False)
    top_positions = sorted_positions.head(20).index if len(sorted_positions) > 20 else sorted_positions.index
    top_df = valid_df[valid_df['position'].isin(top_positions)]
    x_labels = list("ACDEFGHIKLMNPQRSTVWY")
    x_map = {label: i for i, label in enumerate(x_labels)}
    wt_map = {pos: mut[0] for pos, mut in zip(top_df['position'], top_df['mutant'])}
    y_labels = [f"{wt_map.get(pos, '?')}{pos}" for pos in top_positions]
    y_map = {pos: i for i, pos in enumerate(top_positions)}
    
    z_data = np.full((len(y_labels), len(x_labels)), np.nan)
    score_matrix = np.full((len(y_labels), len(x_labels)), np.nan)
    rank_matrix = np.full((len(y_labels), len(x_labels)), np.nan)

    for _, row in top_df.iterrows():
        pos, mut_aa = row['position'], row['mutant'][-1]
        if pos in y_map and mut_aa in x_map:
            y_idx, x_idx = y_map[pos], x_map[mut_aa]
            z_data[y_idx, x_idx] = row['scaled_score']
            score_matrix[y_idx, x_idx] = round(row[score_col], 3)
            rank_matrix[y_idx, x_idx] = row[rank_col]
            
    return x_labels, y_labels, z_data, score_matrix, rank_matrix


def perform_soft_voting(group_df: pd.DataFrame, pred_col: str = 'prediction') -> Tuple:
    """Perform soft voting aggregation on multiple model predictions."""
    all_probs = []
    valid_rows = []
    
    for _, row in group_df.iterrows():
        prob_col = 'probabilities' if 'probabilities' in row else None
        if prob_col and pd.notna(row[prob_col]):
            try:
                if isinstance(row[prob_col], str):
                    prob_str = row[prob_col].strip('[]')
                    probs = [float(x.strip()) for x in prob_str.split(',')]
                elif isinstance(row[prob_col], list):
                    probs = row[prob_col]
                else:
                    probs = json.loads(str(row[prob_col]))
                
                if isinstance(probs, list) and len(probs) > 0:
                    all_probs.append(probs)
                    valid_rows.append(row)
            except (json.JSONDecodeError, ValueError, IndexError):
                continue
    
    if not all_probs:
        return None, None
    
    # Perform soft voting: average all probability distributions
    max_len = max(len(probs) for probs in all_probs)
    normalized_probs = []
    for probs in all_probs:
        if len(probs) < max_len:
            probs.extend([0.0] * (max_len - len(probs)))
        normalized_probs.append(probs)
    
    # Calculate average probabilities across all models
    avg_probs = np.mean(normalized_probs, axis=0)
    
    # Get the class with highest average probability
    voted_class = np.argmax(avg_probs)
    voted_confidence = avg_probs[voted_class]
    
    return voted_class, voted_confidence


def map_prediction_labels(row: pd.Series, task: str) -> str:
    """Map prediction values to human-readable labels."""
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
            except:
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

