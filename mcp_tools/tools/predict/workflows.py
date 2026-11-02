from mcp_tools.models.factory.runners.finetuned import run_cli as run_finetuned_cli
"""Calculation workflows consumed by the API; no Gradio dependency."""
import os
import sys
import subprocess
import time
import json
import logging
import shutil
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Generator, Optional, Tuple
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from mcp_tools.common.constants import COLOR_MAP_FUNCTION, DATASET_MAPPING_FUNCTION, DATASET_TO_TASK_MAP, LABEL_MAPPING_FUNCTION, LLM_MODELS, MODEL_ADAPTER_MAPPING_FUNCTION, MODEL_MAPPING_FUNCTION, REGRESSION_TASKS_FUNCTION, RESIDUE_MAPPING_FUNCTION
from mcp_tools.common.common_utils import create_tar_archive, ensure_project_ckpt, get_save_path
from mcp_tools.common.llm_helpers import LLMConfig, call_llm_api, format_expert_response, generate_ai_summary_prompt, generate_expert_analysis_prompt_residue, get_api_key, get_chat_base_url
from mcp_tools.common.prediction_runners import PROJECT_ROOT
from mcp_tools.common.label_mappers import map_labels
from mcp_tools.common.report_fragments import load_html_template
from mcp_tools.common.results import ignore_progress, result_patch
from mcp_tools.common.common_utils import get_project_root
PROJECT_ROOT = get_project_root()

def truncate_sequence(seq):
    """Truncate sequence if it's a string longer than 30 characters."""
    if isinstance(seq, str) and len(seq) > 30:
        return seq[:]
    return seq

def format_confidence(row, task):
    """Format confidence score for display."""
    score = row["Confidence Score"]
    predicted_class = row["Predicted Class"]

    if isinstance(score, (float, int)) and score != 'N/A':
        return round(float(score), 2)
    elif isinstance(score, str) and score not in ['N/A', '']:
        try:
            if score.startswith('[') and score.endswith(']'):
                prob_str = score.strip('[]')
                probs = [float(x.strip()) for x in prob_str.split(',')]

                current_task = DATASET_TO_TASK_MAP.get(row.get('Dataset', ''), task)

                if current_task in REGRESSION_TASKS_FUNCTION:
                    return predicted_class
                else:
                    labels_key = ("DeepLocMulti" if row.get('Dataset') == "DeepLocMulti"
                                 else "DeepLocBinary" if row.get('Dataset') == "DeepLocBinary"
                                 else current_task)
                    labels = LABEL_MAPPING_FUNCTION.get(labels_key, [])

                    if labels and predicted_class in labels:
                        pred_index = labels.index(predicted_class)
                        if 0 <= pred_index < len(probs):
                            return round(probs[pred_index], 2)

                    return round(max(probs), 2)
            else:
                return round(float(score), 2)
        except (ValueError, IndexError):
            return score
    return score

def generate_plots_for_all_results(results_df: pd.DataFrame) -> go.Figure:
    """Generate plots for function prediction results with consistent Dardana font and academic styling."""
    # Filter data
    def is_non_regression_task(dataset):
        """Check if dataset is not a regression task."""
        return DATASET_TO_TASK_MAP.get(dataset) not in REGRESSION_TASKS_FUNCTION

    plot_df = results_df[
        (results_df['header'] != "ERROR") &
        (results_df['Dataset'].apply(is_non_regression_task))
    ].copy()

    if plot_df.empty:
        return go.Figure().update_layout(
            title_text="<b>No Visualization Available</b>",
            title_x=0.5,
            font_family="Dardana",
            title_font_size=14,
            xaxis={"visible": False},
            yaxis={"visible": False},
            annotations=[{
                "text": "This task does not support visualization.",
                "xref": "paper", "yref": "paper",
                "x": 0.5, "y": 0.5,
                "showarrow": False,
                "font": {"family": "Dardana", "size": 12}
            }]
        )

    sequences = plot_df['header'].unique()
    datasets = plot_df['Dataset'].unique()
    n_seq, n_data = len(sequences), len(datasets)

    titles = [
        f"{seq[:15]}<br>{ds[:20]}" if n_seq > 1 else f"{ds[:25]}"
        for seq in sequences for ds in datasets
    ]

    fig = make_subplots(
        rows=n_seq, cols=n_data,
        subplot_titles=titles,
        vertical_spacing=0.25 if n_seq > 1 else 0.15
    )

    FONT_STYLE = dict(family="Dardana", size=12)
    AXIS_TITLE_STYLE = dict(family="Dardana", size=11)
    TICK_FONT_STYLE = dict(family="Dardana", size=10)
    BAR_WIDTH = 0.7

    for r_idx, seq in enumerate(sequences, 1):
        for c_idx, ds in enumerate(datasets, 1):
            row_data = plot_df[(plot_df['header'] == seq) & (plot_df['Dataset'] == ds)]
            if row_data.empty:
                continue

            row = row_data.iloc[0]
            prob_col = next((col for col in row.index if 'probab' in col.lower()), None)

            if not prob_col or pd.isna(row[prob_col]):
                continue

            try:
                confidences = (json.loads(row[prob_col]) if isinstance(row[prob_col], str)
                             else row[prob_col])

                if not isinstance(confidences, list):
                    continue

                task = DATASET_TO_TASK_MAP.get(ds)
                labels_key = ("DeepLocMulti" if ds == "DeepLocMulti"
                            else "DeepLocBinary" if ds == "DeepLocBinary"
                            else task)

                labels = LABEL_MAPPING_FUNCTION.get(
                    labels_key,
                    [f"Class {k}" for k in range(len(confidences))]
                )

                colors = [
                    COLOR_MAP_FUNCTION.get(lbl, COLOR_MAP_FUNCTION["Default"])
                    for lbl in labels
                ]

                def get_confidence(item):
                    """Get confidence score from plot data item."""
                    return item[1]

                plot_data = sorted(
                    zip(labels, confidences, colors),
                    key=get_confidence,
                    reverse=True
                )
                sorted_labels, sorted_conf, sorted_colors = zip(*plot_data)

                fig.add_trace(
                    go.Bar(
                        x=sorted_labels,
                        y=sorted_conf,
                        marker_color=sorted_colors,
                        width=BAR_WIDTH,
                        marker_line_width=0,
                        hovertemplate="<b>%{x}</b><br>Confidence: %{y:.2f}<extra></extra>"
                    ),
                    row=r_idx, col=c_idx
                )

                # Axis styling
                fig.update_xaxes(
                    tickfont=TICK_FONT_STYLE,
                    row=r_idx, col=c_idx
                )
                fig.update_yaxes(
                    range=[0, 1],
                    title_text="Confidence" if c_idx == 1 else "",
                    title_font=AXIS_TITLE_STYLE,
                    tickfont=TICK_FONT_STYLE,
                    row=r_idx, col=c_idx
                )

            except Exception as e:
                print(f"Plotting error for {seq}/{ds}: {e}")

    # Global layout adjustments
    main_title = "<b>Prediction Confidence Scores</b>"
    if n_seq == 1:
        main_title += f"<br><sub>{sequences[0][:80]}</sub>"

    fig.update_layout(
        title=dict(
            text=main_title,
            x=0.5,
            font=dict(family="Dardana", size=16)
        ),
        showlegend=False,
        font=FONT_STYLE,
        height=max(400, 300 * n_seq + 100),
        margin=dict(l=50, r=50, b=80, t=100),
        plot_bgcolor='white',
        paper_bgcolor='white',
        bargap=0.2
    )

    # Subplot title styling
    for annotation in fig['layout']['annotations']:
        annotation['font'] = dict(family="Dardana", size=12)

    return fig

def handle_protein_function_prediction_advance(
    task: str,
    fasta_file: Any,
    enable_ai: bool,
    llm_model: str,
    user_api_key: Optional[str] = None,
    model_name: Optional[str] = None,
    datasets: Optional[List[str]] = None,
    progress=ignore_progress
    ) -> Generator:
    """Handle protein function prediction workflow."""
    model = model_name if model_name else "ESM2-650M"

    # Built-in model logic
    if datasets is not None and len(datasets) > 0:
        final_datasets = datasets
    else:
        final_datasets = DATASET_MAPPING_FUNCTION.get(task, [])
    if not all([task, datasets, fasta_file]):
        yield (
            "❌ Error: Task, Datasets, and FASTA file are required.",
            pd.DataFrame(), None, result_patch(visible=False),
            "Please provide all required inputs."
        )
        return
    progress(0.1, desc="Running prediction...")
    yield (
        f"🚀 Starting predictions with {model}...",
        pd.DataFrame(), None, result_patch(visible=False),
        "AI analysis will appear here..."
    )

    all_results_list = []

    timestamp = str(int(time.time()))
    function_dir = get_save_path("Protein_Function", "Result")

    for i, dataset in enumerate(final_datasets):
        yield (
            f"⏳ Running prediction...",
            pd.DataFrame(), None, result_patch(visible=False),
            "AI analysis will appear here..."
        )

        try:
            model_key = MODEL_MAPPING_FUNCTION.get(model)
            if not model_key:
                raise ValueError(f"Model key not found for {model}")

            adapter_key = MODEL_ADAPTER_MAPPING_FUNCTION[model_key]
            script_path = PROJECT_ROOT / 'mcp_tools/models/factory/prediction' / f"{model_key}.py"
            adapter_path = ensure_project_ckpt(PROJECT_ROOT / "ckpt" / dataset / adapter_key)
            output_file = function_dir / f"temp_{dataset}_{model}_{timestamp}.csv"

            if not script_path.exists() or not adapter_path.exists():
                raise FileNotFoundError(f"Required files not found: Script={script_path}, Adapter={adapter_path}")

            if isinstance(fasta_file, str):
                file_path = fasta_file
            else:
                file_path = fasta_file.name
            cmd = [sys.executable, str(script_path), "--fasta_file", str(Path(file_path)), "--adapter_path", str(adapter_path), "--output_csv", str(output_file)]
            run_finetuned_cli(cmd)

            if output_file.exists():
                df = pd.read_csv(output_file)
                df["Dataset"] = dataset
                all_results_list.append(df)
                os.remove(output_file)
        except Exception as e:
            error_detail = e.stderr if isinstance(e, subprocess.CalledProcessError) else str(e)
            print(f"Failed to process '{dataset}': {error_detail}")
            all_results_list.append(pd.DataFrame([{"Dataset": dataset, "header": "ERROR", "sequence": error_detail}]))

    if not all_results_list:
        yield "⚠️ No results generated.", pd.DataFrame(), None, result_patch(visible=False), "No results to analyze."
        return
    progress(0.7, desc="Processing results...")
    final_df = pd.concat(all_results_list, ignore_index=True).fillna('N/A')

    plot_fig = generate_plots_for_all_results(final_df)
    display_df = final_df.copy()

    display_df["predicted_class"] = display_df.apply(lambda row: map_labels(row, task), axis=1)
    if 'prediction' in display_df.columns:
        display_df.drop(columns=['prediction'], inplace=True)

    rename_map = {
        'header': "Protein Name",
        'sequence': "Sequence",
        'predicted_class': "Predicted Class",
        'probabilities': "Confidence Score",
        'Dataset': "Dataset"
    }
    display_df.rename(columns=rename_map, inplace=True)

    if "Sequence" in display_df.columns:
        display_df["Sequence"] = display_df["Sequence"].apply(truncate_sequence)

    if "Confidence Score" in display_df.columns and "Predicted Class" in display_df.columns:
        display_df["Confidence Score"] = display_df.apply(lambda row: format_confidence(row, task), axis=1)

    ai_summary = "AI Analysis disabled. Enable in settings to generate a report."
    expert_analysis = load_html_template("ai_expert_placeholder.html")

    if enable_ai:
        progress(0.8, desc="Generating AI summary...")
        yield "🤖 Generating AI summary...", display_df, plot_fig, result_patch(visible=False), expert_analysis
        api_key = get_api_key(llm_model, user_api_key)
        if not api_key:
            ai_summary = f"❌ No API key found for {llm_model}."
        else:
            llm_config = LLMConfig(api_key, llm_model, get_chat_base_url(), LLM_MODELS.get(llm_model, ""))
            prompt = generate_ai_summary_prompt(display_df, task, model)
            ai_summary = call_llm_api(llm_config, prompt)
            expert_analysis = format_expert_response(ai_summary)
        progress(0.9, desc="Finalizing AI analysis...")
    else:
        progress(1.0, desc="Complete!")

    tar_path_str = ""
    try:
        timestamp = str(int(time.time()))
        archive_dir = get_save_path("Protein_Function", "Downloads_zip")

        processed_df_for_save = display_df.copy()
        result_path = archive_dir / f"Result_{timestamp}.csv"
        processed_df_for_save.to_csv(result_path, index=False)

        if plot_fig and hasattr(plot_fig, 'data') and plot_fig.data:
            plot_path = archive_dir / "results_plot.html"
            plot_fig.write_html(str(plot_path))
        else:
            plot_path = None

        if not ai_summary.startswith("❌") and not ai_summary.startswith("AI Analysis"):
            report_path = archive_dir / f"AI_Report_{timestamp}.md"
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(f"# AI Report\n\n{ai_summary}")
        else:
            report_path = None

        files_to_tar = {str(result_path): result_path.name}
        if plot_path and plot_path.exists():
            files_to_tar[str(plot_path)] = plot_path.name
        if report_path and report_path.exists():
            files_to_tar[str(report_path)] = report_path.name

        tar_path = function_dir / f"func_pred_{timestamp}.tar.gz"
        tar_path_str = create_tar_archive(files_to_tar, str(tar_path))
    except Exception as e:
        print(f"Error creating tar.gz file: {e}")
        tar_path_str = ""

    final_status = "✅ All predictions completed!"
    if enable_ai and not ai_summary.startswith("❌"):
        final_status += " AI analysis included."
    progress(1.0, desc="Complete!")
    yield final_status, display_df, plot_fig, result_patch(visible=True, value=tar_path_str), expert_analysis

def generate_plots_for_residue_results(results_df: pd.DataFrame, task: str = "Functional Prediction") -> go.Figure:
    """Generate plots for residue prediction results with consistent styling."""
    if results_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No data to visualize",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, family="Arial")
        )
        return fig

    residues = results_df['residue'].tolist()
    probabilities = results_df['probability'].tolist()
    positions = results_df['index'].tolist() if 'index' in results_df.columns else list(range(1, len(residues) + 1))
    predicted_labels = results_df['predicted_label'].tolist() if 'predicted_label' in results_df.columns else [1 if p >= 0.5 else 0 for p in probabilities]
    functional_residues = [i for i, label in enumerate(predicted_labels) if label == 1]

    fig = go.Figure()

    sequence_length = len(residues)
    y_position = 0

    fig.add_trace(go.Scatter(
        x=[1, sequence_length],
        y=[y_position, y_position],
        mode='lines',
        line=dict(color='lightgray', width=12),
        showlegend=False,
        hoverinfo='skip'
    ))

    if functional_residues:
        segments = []
        current_segment = [functional_residues[0]]

        for i in range(1, len(functional_residues)):
            if functional_residues[i] == functional_residues[i-1] + 1:
                current_segment.append(functional_residues[i])
            else:
                segments.append(current_segment)
                current_segment = [functional_residues[i]]
        segments.append(current_segment)

        for i, segment in enumerate(segments):
            start_pos = positions[segment[0]]
            end_pos = positions[segment[-1]]

            segment_residues = [residues[j] for j in segment]
            segment_probs = [probabilities[j] for j in segment]
            hover_text = f"Task: {task}<br>Range: {start_pos}-{end_pos}<br>Residues: {''.join(segment_residues)}<br>Avg Probability: {np.mean(segment_probs):.3f}"

            fig.add_trace(go.Scatter(
                x=[start_pos, end_pos],
                y=[y_position, y_position],
                mode='lines',
                line=dict(color='red', width=12),
                showlegend=False,
                hovertemplate=hover_text + "<extra></extra>",
                name=f"Functional Region {i+1}"
            ))

    interval = max(1, sequence_length // 10)
    position_labels = list(range(1, sequence_length + 1, interval))
    if position_labels[-1] != sequence_length:
        position_labels.append(sequence_length)

    fig.update_layout(
        title=dict(
            text=f"<b>Functional Residue </b> - {task}",
            x=0.02, y=0.95, xanchor='left', yanchor='top',
            font=dict(size=14, family="Arial", color="black")
        ),
        xaxis=dict(
            title="Residue Position",
            tickmode='array',
            tickvals=position_labels,
            ticktext=[str(pos) for pos in position_labels],
            tickfont=dict(size=10),
            showgrid=False,
            zeroline=False,
            range=[0, sequence_length + 1]
        ),
        yaxis=dict(
            title="",
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            range=[-0.5, 0.5]
        ),
        showlegend=False,
        height=150,
        margin=dict(l=50, r=50, t=50, b=40),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family="Arial", size=10),
        hovermode='closest'
    )
    return fig

def expand_residue_predictions(df):
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

def handle_protein_residue_function_prediction(
    task: str,
    fasta_file: Any,
    enable_ai: bool,
    llm_model: str,
    user_api_key: Optional[str] = None,
    model_name: Optional[str] = None,
    progress=ignore_progress
) -> Generator:

    """Handle protein residue function prediction workflow."""
    model = model_name if model_name else "ESM2-650M"

    if not all([task, fasta_file]):
        yield(
           "❌ Error: Task and FASTA file are required.",
            pd.DataFrame(), None,
            result_patch(visible=False),
            "Please provide all required inputs.",
            "AI Analysis disabled."
        )
        return

    all_results_list = []
    timestamp = str(int(time.time()))
    residue_save_dir = get_save_path("Protein_Function", "Residue_save")

    # Run predictions for each dataset
    progress(0.2, desc="Running Predicrtion...")
    yield(
        f"⏳ Running prediction...",
        pd.DataFrame(), None,
        result_patch(visible=False),
        "AI analysis will appear here...",
        "AI Analysis disabled."
    )

    try:
        model_key = MODEL_MAPPING_FUNCTION.get(model)
        if not model_key:
            raise ValueError(f"Model key not found for {model}")

        adapter_key = MODEL_ADAPTER_MAPPING_FUNCTION[model_key]

        # Get residue task dataset
        datasets = RESIDUE_MAPPING_FUNCTION.get(task, [])
        if not datasets:
            raise ValueError(f"No datasets found for task: {task}")

        for dataset in datasets:
            script_path = PROJECT_ROOT / 'mcp_tools/models/factory/prediction' / f"{model_key}.py"
            adapter_path = ensure_project_ckpt(PROJECT_ROOT / "ckpt" / dataset / adapter_key)
            output_file = residue_save_dir/ f"{dataset}_{model}_{timestamp}.csv"

            if not script_path.exists() or not adapter_path.exists():
                raise FileNotFoundError(f"Required files not found: Script={script_path}, Adapter={adapter_path}")
            if isinstance(fasta_file, str):
                file_path = fasta_file
            else:
                file_path = fasta_file.name

            cmd = [sys.executable, str(script_path),
                 "--fasta_file", str(Path(file_path)),
                 "--adapter_path", str(adapter_path),
                 "--output_csv", str(output_file)]
            run_finetuned_cli(cmd)

            if output_file.exists():
                df = pd.read_csv(output_file)
                df["Task"] = task
                df["Dataset"] = dataset
                all_results_list.append(df)
                os.remove(output_file)

    except Exception as e:
        error_detail = e.stderr if isinstance(e, subprocess.CalledProcessError) else str(e)
        print(f"Failed to process '{task}': {error_detail}")
        all_results_list.append(pd.DataFrame([{"Task": task, "header": "ERROR", "residue": error_detail, "probability": 0}]))
        yield (
            f"❌ Prediction failed: {error_detail}",
            pd.DataFrame(), None,
            result_patch(visible=False),
            "An error occurred during prediction.",
            "AI Analysis disabled.",
        )
        return

    progress(0.7, desc="Processing results...")
    if all_results_list:
        combined_df = pd.concat(all_results_list, ignore_index=True)
        final_df = expand_residue_predictions(combined_df)
    else:
        final_df = pd.DataFrame()
    download_path = residue_save_dir / f"prediction_results_{timestamp}.csv"
    final_df.to_csv(download_path, index=False)

    progress(0.8, desc="Creating visualization...")

    try:
        residue_plot = generate_plots_for_residue_results(final_df, task)
        plot_update = result_patch(value=residue_plot, visible=True)
    except Exception as e:
        print(f"Visualization error: {e}")
        plot_update = result_patch(visible=False)

    display_df = final_df.copy()
    column_rename = {
        'index': 'Position',
        'residue': 'Residue',
        'predicted_label': 'Predicted Label',
        'probability': 'Probability',
    }
    display_df.rename(columns=column_rename, inplace=True)
    if 'Probability' in display_df.columns:
        display_df['Probability'] = display_df['Probability'].round(3)
    ai_summary = "AI Analysis disabled. Enable in settings to generate a report."
    ai_response = "AI Analysis disabled."  # Initialize ai_response variable

    expert_analysis = load_html_template("ai_expert_placeholder.html")

    if enable_ai:
        progress(0.8, desc="Generate AI summary...")
        yield (
            "🤖 Expert is analyzing results...",
            display_df, None,
            result_patch(visible=False),
            expert_analysis,
            "AI Analysis in progress..."
        )
        api_key = get_api_key(llm_model, user_api_key)
        if not api_key:
            ai_summary = f"❌ No API key found for {llm_model}."
        else:
            llm_config = LLMConfig(api_key, llm_model, get_chat_base_url(), LLM_MODELS.get(llm_model, ""))
            prompt = generate_expert_analysis_prompt_residue(display_df, task)
            ai_response = call_llm_api(llm_config, prompt)
            expert_analysis = format_expert_response(ai_response)
        progress(0.9, desc="Finalizing AI analysis...")
    else:
        progress(1.0, desc="Complete!")

    final_status = "✅ All predictions completed!"
    progress(1.0, desc="Complete!")
    yield(
        final_status,
        display_df,
        residue_plot,
        result_patch(visible=True, value=download_path),
        expert_analysis,
        ai_response
    )
