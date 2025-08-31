import os
import json
import gradio as gr
from datasets import load_dataset

def load_html_template(name: str, **kwargs) -> str:
    """Load and format an HTML fragment from assets/html_fragments/
    Args:
        name: name of the HTML fragment
        **kwargs: keyword arguments to format the HTML fragment
    Returns:
        formatted HTML fragment
    """
    path = os.path.join("src", "web", "assets", "html_fragments", name)
    with open(path, "r", encoding="utf-8") as f:
        html = f.read()
    if kwargs:
        return html.format(**kwargs)
    return html

def format_metrics_table(metrics_dict, priority_metrics=None):
    """Format metrics dictionary into HTML table rows
    Args:
        metrics_dict: dictionary of metrics
        priority_metrics: list of priority metric names
    Returns:
        formatted HTML table rows
    """
    if priority_metrics is None:
        priority_metrics = ['loss', 'accuracy', 'f1', 'precision', 'recall', 'auroc', 'mcc']
    
    def get_priority(item):
        name = item[0]
        if name in priority_metrics:
            return priority_metrics.index(name)
        return len(priority_metrics)
    
    sorted_metrics = sorted(metrics_dict.items(), key=get_priority)
    metrics_rows = ""
    
    for metric_name, metric_value in sorted_metrics:
        # use bold for priority metrics
        is_priority = metric_name in priority_metrics
        name_class = 'priority-metric' if is_priority else ''
        
        # convert metric name: abbreviations in uppercase, non-abbreviations in capitalize
        display_name = metric_name
        if metric_name.lower() in ['f1', 'mcc', 'auroc']:
            display_name = metric_name.upper()
        else:
            display_name = metric_name.capitalize()
        
        metrics_rows += load_html_template("metrics_row.html", 
                                         name_class=name_class,
                                         display_name=display_name,
                                         metric_value=metric_value)
    
    return metrics_rows

def generate_prediction_status_html(stage, status):
    """Generate prediction status HTML
    Args:
        stage: current stage
        status: status (running, completed, failed)
    Returns:
        formatted HTML
    """
    # Determine status color and icon
    if status == "running":
        status_color = "#4285f4"  # Blue
        icon = "⏳"
        animation = """
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.05); }
            100% { transform: scale(1); }
        }
        """
        animation_style = "animation: pulse 1.5s infinite ease-in-out;"
    elif status == "completed":
        status_color = "#2ecc71"  # Green
        icon = "✅"
        animation = ""
        animation_style = ""
    else:  # failed
        status_color = "#e74c3c"  # Red
        icon = "❌"
        animation = ""
        animation_style = ""
    
    return load_html_template("prediction_status.html",
                            status_color=status_color,
                            icon=icon,
                            stage=stage,
                            status=status,
                            status_capitalized=status.capitalize(),
                            animation=animation,
                            animation_style=animation_style)

def generate_prediction_results_html(problem_type, prediction_data):
    """Generate prediction results HTML based on problem type
    Args:
        problem_type: type of prediction problem
        prediction_data: prediction results data
    Returns:
        formatted HTML
    """
    if problem_type == "regression":
        problem_type_title = "Regression Prediction Results"
        results_table = load_html_template("regression_results_table.html",
                                         predicted_value=prediction_data['prediction'])
        probabilities_table = ""
    elif problem_type == "single_label_classification":
        problem_type_title = "Single-Label Classification Results"
        results_table = load_html_template("classification_results_table.html",
                                         predicted_class=prediction_data['predicted_class'])
        
        # Create probability table
        prob_rows = ""
        if isinstance(prediction_data.get('probabilities'), list):
            prob_rows = "".join([
                f"<tr><td style='text-align:center'>Class {i}</td><td style='text-align:center'>{prob:.4f}</td></tr>"
                for i, prob in enumerate(prediction_data['probabilities'])
            ])
        elif isinstance(prediction_data.get('probabilities'), dict):
            prob_rows = "".join([
                f"<tr><td style='text-align:center'>Class {label}</td><td style='text-align:center'>{prob:.4f}</td></tr>"
                for label, prob in prediction_data['probabilities'].items()
            ])
        else:
            prob_value = prediction_data.get('probabilities', 0)
            prob_rows = f"<tr><td style='text-align:center'>Class 0</td><td style='text-align:center'>{prob_value:.4f}</td></tr>"
        
        probabilities_table = load_html_template("probabilities_table.html", prob_rows=prob_rows)
    elif problem_type == "residue_single_label_classification":
        problem_type_title = "Residue-Level Classification Results"
        
        # Get the amino acid sequence from the prediction data if available
        aa_seq = prediction_data.get('aa_seq', '')
        predicted_classes = prediction_data.get('predicted_classes', [])
        probabilities = prediction_data.get('probabilities', [])
        
        # Create residue-level table rows
        residue_rows = ""
        for pos, (aa, pred_class, probs) in enumerate(zip(aa_seq, predicted_classes, probabilities)):
            # Format probabilities for display
            prob_str = ", ".join([f"Class {i}: {prob:.3f}" for i, prob in enumerate(probs)])
            residue_rows += f"""
            <tr>
                <td style='text-align:center'>{pos + 1}</td>
                <td style='text-align:center'>{aa}</td>
                <td style='text-align:center'>{pred_class}</td>
                <td style='text-align:left; font-size: 0.9em;'>{prob_str}</td>
            </tr>
            """
        
        results_table = load_html_template("residue_results_table.html", residue_rows=residue_rows)
        probabilities_table = ""
    elif problem_type == "residue_regression":
        problem_type_title = "Residue-Level Regression Results"
        
        # Get the amino acid sequence from the prediction data if available
        aa_seq = prediction_data.get('aa_seq', '')
        predictions = prediction_data.get('predictions', [])
        
        # Create residue-level table rows
        residue_rows = ""
        for pos, (aa, pred_value) in enumerate(zip(aa_seq, predictions)):
            residue_rows += f"""
            <tr>
                <td style='text-align:center'>{pos + 1}</td>
                <td style='text-align:center'>{aa}</td>
                <td style='text-align:center'>{pred_value:.4f}</td>
            </tr>
            """
        
        results_table = load_html_template("residue_regression_results_table.html", residue_rows=residue_rows)
        probabilities_table = ""
    else:  # multi_label_classification
        problem_type_title = "Multi-Label Classification Results"
        # Create prediction table
        pred_rows = ""
        if 'predictions' in prediction_data and 'probabilities' in prediction_data:
            if (isinstance(prediction_data['predictions'], list) and 
                isinstance(prediction_data['probabilities'], list)):
                pred_rows = "".join([
                    f"<tr><td style='width:33.33%; text-align:center'>Label {i}</td><td style='width:33.33%; text-align:center'>{pred}</td><td style='width:33.33%; text-align:center'>{prob:.4f}</td></tr>"
                    for i, (pred, prob) in enumerate(zip(prediction_data['predictions'], prediction_data['probabilities']))
                ])
            elif (isinstance(prediction_data['predictions'], dict) and 
                  isinstance(prediction_data['probabilities'], dict)):
                pred_rows = "".join([
                    f"<tr><td style='width:33.33%; text-align:center'>Label {label}</td><td style='width:33.33%; text-align:center'>{pred}</td><td style='width:33.33%; text-align:center'>{prediction_data['probabilities'].get(label, 0):.4f}</td></tr>"
                    for label, pred in prediction_data['predictions'].items()
                ])
            else:
                pred = prediction_data['predictions'] if 'predictions' in prediction_data else "N/A"
                prob = prediction_data['probabilities'] if 'probabilities' in prediction_data else 0.0
                pred_rows = f"<tr><td style='width:33.33%; text-align:center'>Label 0</td><td style='width:33.33%; text-align:center'>{pred}</td><td style='width:33.33%; text-align:center'>{prob:.4f}</td></tr>"
        else:
            for key, value in prediction_data.items():
                if 'label' in key.lower() or 'class' in key.lower():
                    label_name = key
                    label_value = value
                    prob_value = prediction_data.get(f"{key}_prob", 0.0)
                    pred_rows += f"<tr><td style='width:33.33%; text-align:center'>{label_name}</td><td style='width:33.33%; text-align:center'>{label_value}</td><td style='width:33.33%; text-align:center'>{prob_value:.4f}</td></tr>"
        
        results_table = load_html_template("multilabel_results_table.html", pred_rows=pred_rows)
        probabilities_table = ""
    
    # Load residue prediction CSS if needed
    css_style = ""
    if problem_type in ["residue_single_label_classification", "residue_regression"]:
        try:
            css_path = os.path.join("src", "web", "assets", "css", "residue_prediction.css")
            with open(css_path, "r", encoding="utf-8") as f:
                css_style = f"<style>{f.read()}</style>"
        except FileNotFoundError:
            print("Warning: residue_prediction.css not found")
    
    return css_style + load_html_template("prediction_results.html",
                            problem_type_title=problem_type_title,
                            results_table=results_table,
                            probabilities_table=probabilities_table)

def generate_batch_prediction_results_html(df, problem_type):
    """Generate batch prediction results HTML
    Args:
        df: pandas DataFrame with results
        problem_type: type of prediction problem
    Returns:
        formatted HTML
    """
    # Create summary statistics based on problem type
    summary_stats = ""
    if problem_type == "regression":
        summary_stats = f"""
        <div class="summary-stats">
            <div class="stat-item">
                <div class="stat-value">{len(df)}</div>
                <div class="stat-label">Predictions</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{df['prediction'].mean():.4f}</div>
                <div class="stat-label">Mean</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{df['prediction'].min():.4f}</div>
                <div class="stat-label">Min</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{df['prediction'].max():.4f}</div>
                <div class="stat-label">Max</div>
            </div>
        </div>
        """
    elif problem_type == "residue_single_label_classification":
        # For residue classification, show summary of residue predictions
        total_residues = 0
        class_counts = {}
        
        for _, row in df.iterrows():
            if 'residue_predictions' in row and isinstance(row['residue_predictions'], list):
                total_residues += len(row['residue_predictions'])
                for pred in row['residue_predictions']:
                    class_counts[pred] = class_counts.get(pred, 0) + 1
        
        summary_stats = f"""
        <div class="summary-stats">
            <div class="stat-item">
                <div class="stat-value">{len(df)}</div>
                <div class="stat-label">Sequences</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{total_residues}</div>
                <div class="stat-label">Total Residues</div>
            </div>
        </div>
        """
        
        # Add class distribution if available
        if class_counts:
            class_dist = ", ".join([f"Class {k}: {v}" for k, v in sorted(class_counts.items())])
            summary_stats += f"""
            <div class="class-distribution">
                <h4>Class Distribution:</h4>
                <p>{class_dist}</p>
            </div>
            """
    elif problem_type == "residue_regression":
        # For residue regression, show summary of residue predictions
        total_residues = 0
        all_values = []
        
        for _, row in df.iterrows():
            if 'residue_predictions' in row and isinstance(row['residue_predictions'], list):
                total_residues += len(row['residue_predictions'])
                all_values.extend(row['residue_predictions'])
        
        if all_values:
            import numpy as np
            all_values = np.array(all_values)
            summary_stats = f"""
            <div class="summary-stats">
                <div class="stat-item">
                    <div class="stat-value">{len(df)}</div>
                    <div class="stat-label">Sequences</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{total_residues}</div>
                    <div class="stat-label">Total Residues</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{all_values.mean():.4f}</div>
                    <div class="stat-label">Mean Value</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{all_values.min():.4f}</div>
                    <div class="stat-label">Min Value</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{all_values.max():.4f}</div>
                    <div class="stat-label">Max Value</div>
                </div>
            </div>
            """
    elif problem_type == "single_label_classification":
        if 'predicted_class' in df.columns:
            class_counts = df['predicted_class'].value_counts()
            class_stats = "".join([
                f"""
                <div class="stat-item">
                    <div class="stat-value">{count}</div>
                    <div class="stat-label">Class {class_label}</div>
                </div>
                """
                for class_label, count in class_counts.items()
            ])
            
            summary_stats = f"""
            <div class="summary-stats">
                <div class="stat-item">
                    <div class="stat-value">{len(df)}</div>
                    <div class="stat-label">Predictions</div>
                </div>
                {class_stats}
            </div>
            """
    elif problem_type == "multi_label_classification":
        label_cols = [col for col in df.columns if col.startswith('label_') and not col.endswith('_prob')]
        if label_cols:
            label_stats = "".join([
                f"""
                <div class="stat-item">
                    <div class="stat-value">{df[col].sum()}</div>
                    <div class="stat-label">{col}</div>
                </div>
                """
                for col in label_cols
            ])
            
            summary_stats = f"""
            <div class="summary-stats">
                <div class="stat-item">
                    <div class="stat-value">{len(df)}</div>
                    <div class="stat-label">Predictions</div>
                </div>
                {label_stats}
            </div>
            """
    
    # Create table headers and rows
    table_headers = " ".join([f'<th style="text-align: center;">{col}</th>' for col in df.columns])
    table_rows = generate_table_rows(df)
    
    # Load residue prediction CSS if needed
    css_style = ""
    if problem_type in ["residue_single_label_classification", "residue_regression"]:
        try:
            css_path = os.path.join("src", "web", "assets", "css", "residue_prediction.css")
            with open(css_path, "r", encoding="utf-8") as f:
                css_style = f"<style>{f.read()}</style>"
        except FileNotFoundError:
            print("Warning: residue_prediction.css not found")
    
    return css_style + load_html_template("batch_prediction_results.html",
                            summary_stats=summary_stats,
                            table_headers=table_headers,
                            table_rows=table_rows)

def generate_table_rows(df, max_rows=100):
    """Generate HTML table rows with special handling for sequence data
    Args:
        df: pandas DataFrame
        max_rows: maximum number of rows to display
    Returns:
        formatted HTML table rows
    """
    rows = []
    for i, row in df.iterrows():
        if i >= max_rows:
            break
        
        cells = []
        for col in df.columns:
            value = row[col]
            # Special handling for sequence type columns
            if col in ['aa_seq', 'foldseek_seq', 'ss8_seq'] and isinstance(value, str) and len(value) > 30:
                # Add title attribute to show full sequence on hover
                cell = f'<td title="{value}" style="padding: 15px; font-size: 14px; border: 1px solid #ddd; font-family: monospace; text-align: center; vertical-align: middle; display: table-cell; text-align: center;">{value[:30]}...</td>'
            # Special handling for residue predictions (lists)
            elif col in ['residue_predictions', 'aa_seq_residues'] and isinstance(value, list):
                if len(value) > 10:
                    # Show first 10 elements and indicate there are more
                    display_value = str(value[:10]) + "..."
                    full_value = str(value)
                    cell = f'<td title="{full_value}" style="padding: 15px; font-size: 12px; border: 1px solid #ddd; text-align: center; max-width: 200px; overflow: hidden;">{display_value}</td>'
                else:
                    cell = f'<td style="padding: 15px; font-size: 12px; border: 1px solid #ddd; text-align: center; max-width: 200px; overflow: hidden;">{value}</td>'
            # Format numeric values to 4 decimal places
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                formatted_value = f"{value:.4f}" if isinstance(value, float) else value
                cell = f'<td style="padding: 15px; font-size: 14px; border: 1px solid #ddd; text-align: center;">{formatted_value}</td>'
            else:
                cell = f'<td style="padding: 15px; font-size: 14px; border: 1px solid #ddd; text-align: center;">{value}</td>'
            cells.append(cell)
        
        # Add alternating row background color
        bg_color = "#f9f9f9" if i % 2 == 1 else "white"
        rows.append(f'<tr style="background-color: {bg_color};">{" ".join(cells)}</tr>')
    
    if len(df) > max_rows:
        cols_count = len(df.columns)
        rows.append(f'<tr><td colspan="{cols_count}" style="text-align:center; font-style:italic; padding: 15px; font-size: 14px; border: 1px solid #ddd;">Showing {max_rows} of {len(df)} rows</td></tr>')
    
    return '\n'.join(rows)
