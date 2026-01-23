import gradio as gr
import pandas as pd
import os
import sys
import subprocess
import time
from pathlib import Path
from typing import Dict, Any, List, Generator, Optional, Tuple
import plotly.graph_objects as go
import numpy as np
import json
from dotenv import load_dotenv

from web.utils.constants import *
from web.utils.common_utils import *
from web.utils.file_handlers import *
from web.utils.ai_helpers import *
from web.utils.data_processors import *
from web.utils.visualization import *
from web.utils.prediction_runners import *
from web.utils.label_mappers import map_labels

load_dotenv()

def handle_mutation_prediction_base(
    function_selection: str, 
    file_obj: Any, 
    enable_ai: bool, 
    ai_model: str, 
    user_api_key: Optional[str] = None, 
    model_name: Optional[str] = None,
    progress=gr.Progress()
) -> Generator:
    try:
        import requests
        requests.post("/api/stats/track", json={"module": "function_analysis"})
    except Exception:
        pass
    """Handle mutation prediction workflow."""
    if not file_obj or not function_selection:
        yield (
            "❌ Error: Function and file are required.", 
            None, None, gr.update(visible=False), None, 
            gr.update(visible=False), None, 
            "Please select a function and upload a file."
        )
        return

    if isinstance(file_obj, str):
        file_path = file_obj
    else:
        file_path = file_obj.name

    if file_path.lower().endswith((".fasta", ".fa")):
        if model_name and model_name in ["VenusPLM", "ESM2-650M", "ESM-1v", "ESM-1b"]:
            model_type = "sequence"
        else:
            model_name, model_type = "ESM2-650M", "sequence"

        processed_file_path = process_fasta_file(file_path)
        if processed_file_path != file_path:
            file_path = processed_file_path
            yield (
                "⚠️ Multi-sequence FASTA detected. Using only the first sequence for prediction.",
                None, None, gr.update(visible=False), None,
                gr.update(visible=False), None,
                "Processing first sequence only..."
            )
    elif file_path.lower().endswith(".pdb"):
        if model_name and model_name in ["ESM-IF1", "SaProt", "MIF-ST", "ProSST-2048", "ProtSSN", "VenusREM (foldseek-based)"]:
            model_type = "structure"
        else:
            model_name, model_type = "ESM-IF1", "structure"
    else:
        yield (
            "❌ Error: Unsupported file type.", 
            None, None, gr.update(visible=False), None, 
            gr.update(visible=False), None, 
            "Please upload a .fasta, .fa, or .pdb file."
        )
        return

    yield (
        f"⏳ Running prediction...", 
        None, None, gr.update(visible=False), None, 
        gr.update(visible=False), None, 
        "Prediction in progress..."
    )
    progress(0.1, desc="Running prediction...")
    status, raw_df = run_zero_shot_prediction(model_type, model_name, file_path)
    progress(0.7, desc="Processing results...")
    if raw_df.empty:
        yield (
            status, 
            go.Figure(layout={'title': 'No results generated'}), 
            pd.DataFrame(), gr.update(visible=False), None, 
            gr.update(visible=False), None, 
            "No results to analyze."
        )
        return
    
    score_col = next((c for c in raw_df.columns if 'score' in c.lower()), raw_df.columns[1])
    
    display_df = pd.DataFrame()
    display_df['Mutant'] = raw_df['mutant']
    display_df['Prediction Rank'] = range(1, len(raw_df) + 1)
    
    min_s, max_s = raw_df[score_col].min(), raw_df[score_col].max()
    if max_s == min_s:
        scaled_scores = pd.Series([0.0] * len(raw_df))
    else:
        scaled_scores = -1 + 2 * (raw_df[score_col] - min_s) / (max_s - min_s)
    display_df['Prediction Score'] = scaled_scores.round(2)

    df_for_heatmap = raw_df.copy()
    df_for_heatmap['Prediction Rank'] = range(1, len(df_for_heatmap) + 1)

    total_residues = get_total_residues_count(df_for_heatmap)
    data_tuple = prepare_top_residue_heatmap_data(df_for_heatmap)
    
    if data_tuple[0] is None:
        yield (
            status, 
            go.Figure(layout={'title': 'Score column not found'}), 
            display_df, gr.update(visible=False), None, 
            gr.update(visible=False), display_df, 
            "Score column not found."
        )
        return

    summary_fig = generate_plotly_heatmap(*data_tuple[:4])
    
    ai_summary = "AI Analysis disabled. Enable in settings to generate a report."

    expert_analysis = "<div style='height: 300px; display: flex; align-items: center; justify-content: center; color: #666;'>Analysis will appear here once prediction is complete...</div>"

    if enable_ai:
        progress(0.8, desc="Generating AI summary...")
        yield (
            f"🤖 Expert is analyzing results...", 
            summary_fig, display_df, gr.update(visible=False), None, 
            gr.update(visible=total_residues > 20), display_df, 
            expert_analysis
        )
        
        api_key = get_api_key(ai_model, user_api_key)
        if not api_key:
            ai_summary = "❌ No API key found."
        else:
            ai_config = AIConfig(
                api_key, ai_model, 
                AI_MODELS[ai_model]["api_base"], 
                AI_MODELS[ai_model]["model"]
            )
            prompt = generate_mutation_ai_prompt(display_df, model_name, function_selection)
            ai_summary = call_ai_api(ai_config, prompt)
            expert_analysis = format_expert_response(ai_summary)
        progress(0.9, desc="Finalizing AI analysis...")
    else:
        progress(1.0, desc="Complete!")
    
    timestamp = str(int(time.time()))
    heatmap_dir = get_save_path("Zero_shot", "HeatMap")
    csv_path = heatmap_dir / f"mut_res_{timestamp}.csv"
    heatmap_path = heatmap_dir / f"mut_map_{timestamp}.html"
    
    display_df.to_csv(csv_path, index=False)
    summary_fig.write_html(heatmap_path)
    
    files_to_tar = {
        str(csv_path): f"prediction_results_{timestamp}.csv", 
        str(heatmap_path): f"prediction_heatmap_{timestamp}.html"
    }
    
    if not ai_summary.startswith("❌") and not ai_summary.startswith("AI Analysis"):
        report_path = heatmap_dir / f"ai_report_{timestamp}.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(ai_summary)
        files_to_tar[str(report_path)] = f"AI_Analysis_Report_{timestamp}.md"

    tar_path = heatmap_dir / f"pred_mut_{timestamp}.tar.gz"
    tar_path_str = create_tar_archive(files_to_tar, str(tar_path))

    final_status = status if not enable_ai else "✅ Prediction and AI analysis complete!"
    progress(1.0, desc="Complete!")
    yield (
        final_status, summary_fig, display_df, 
        gr.update(visible=True, value=tar_path_str), tar_path_str, 
        gr.update(visible=total_residues > 20), display_df, expert_analysis
    )

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

def handle_protein_function_prediction(
    task: str, 
    fasta_file: Any, 
    enable_ai: bool, 
    ai_model: str, 
    user_api_key: Optional[str] = None, 
    model_name: Optional[str] = None, 
    datasets: Optional[List[str]] = None,
    progress=gr.Progress()
    ) -> Generator:
    try:
        import requests
        requests.post("/api/stats/track", json={"module": "function_analysis"})
    except Exception:
        pass
    """Handle protein function prediction workflow."""
    model = model_name if model_name else "ESM2-650M"
    datasets = (datasets if datasets is not None 
               else DATASET_MAPPING_FUNCTION.get(task, []))

    if not all([task, datasets, fasta_file]):
        yield (
            "❌ Error: Task, Datasets, and FASTA file are required.", 
            pd.DataFrame(), 
            gr.update(visible=False), 
            "Please provide all required inputs.",
            "AI Analysis disabled."
        )
        return

    yield (
        f"🚀 Starting predictions with {model}...", 
        pd.DataFrame(), 
        gr.update(visible=False), 
        "AI analysis will appear here...",
        "AI Analysis disabled."
    )
    
    all_results_list = []
    timestamp = str(int(time.time()))
    function_dir = get_save_path("Protein_Function", "Result")

    # Run predictions for each dataset
    progress(0.1, desc="Running prediction...")
    for i, dataset in enumerate(datasets):
        yield (
            f"⏳ Running prediction...", 
            pd.DataFrame(), 
            gr.update(visible=False), 
            "AI analysis will appear here...",
            "AI Analysis disabled."
        )
        
        try:
            model_key = MODEL_MAPPING_FUNCTION.get(model)
            if not model_key:
                raise ValueError(f"Model key not found for {model}")
            
            adapter_key = MODEL_ADAPTER_MAPPING_FUNCTION[model_key]
            script_path = Path("src") / "property" / f"{model_key}.py"
            adapter_path = Path("ckpt") / dataset / adapter_key
            output_file = function_dir / f"{dataset}_{model}_{timestamp}.csv"
            
            if not script_path.exists() or not adapter_path.exists():
                raise FileNotFoundError(f"Required files not found: Script={script_path}, Adapter={adapter_path}")
            if isinstance(fasta_file, str):
                file_path = fasta_file
            else:
                file_path = fasta_file.name
            cmd = [sys.executable, str(script_path), "--fasta_file", str(Path(file_path)), "--adapter_path", str(adapter_path), "--output_csv", str(output_file)]
            subprocess.run(cmd, capture_output=True, text=True, check=True, encoding='utf-8', errors='ignore')
            
            if output_file.exists():
                df = pd.read_csv(output_file) 
                df["Dataset"] = dataset
                all_results_list.append(df)
                os.remove(output_file)
        except Exception as e:
            error_detail = e.stderr if isinstance(e, subprocess.CalledProcessError) else str(e)
            print(f"Failed to process '{dataset}': {error_detail}")
            all_results_list.append(pd.DataFrame([{"Dataset": dataset, "header": "ERROR", "sequence": error_detail}]))
    progress(0.7, desc="Processing results...")
    if not all_results_list:
        yield (
            "⚠️ No results generated.", 
            pd.DataFrame(), 
            gr.update(visible=False), 
            "No results to analyze.",
            "AI Analysis disabled."
        )
        return
    
    raw_final_df = pd.concat(all_results_list, ignore_index=True).fillna('N/A')
    
    final_df = raw_final_df.copy()
    non_voting_tasks = REGRESSION_TASKS_FUNCTION
    non_voting_datasets = ["DeepLocMulti", "DeepLocBinary"]
    is_voting_run = task not in non_voting_tasks and not any(ds in raw_final_df['Dataset'].unique() for ds in non_voting_datasets) and len(raw_final_df['Dataset'].unique()) > 1

    if is_voting_run:
        yield (
            "🤝 Performing soft voting on prediction results...", 
            pd.DataFrame(), 
            gr.update(visible=False), 
            "Aggregating results...",
            "AI Analysis disabled."
        )
        voted_results = []
        for header, group in raw_final_df.groupby('header'):
            if group.empty: continue
            
            pred_col = 'prediction' if 'prediction' in group.columns else 'predicted_class'
            if pred_col not in group.columns: continue
            
            all_probs = []
            valid_rows = []
            
            for _, row in group.iterrows():
                prob_col = 'probabilities' if 'probabilities' in row else None
                if prob_col and pd.notna(row[prob_col]):
                    try:
                        # Handle string representation of list
                        if isinstance(row[prob_col], str):
                            # Remove brackets and split by comma
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
                continue
            
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
            
            # Create result row
            result_row = group.iloc[0].copy()
            result_row[pred_col] = voted_class
            if 'probabilities' in result_row:
                result_row['probabilities'] = voted_confidence
            
            voted_results.append(result_row.to_frame().T)
        
        if voted_results:
            final_df = pd.concat(voted_results, ignore_index=True)
            final_df = final_df.drop(columns=['Dataset'], errors='ignore')
    display_df = final_df.copy()
    # print(f"Using simplified processing, final_df shape: {final_df.shape}")
    # map_labels is now imported from utils.label_mappers
    
    # Apply label mapping
    display_df["predicted_class"] = display_df.apply(lambda row: map_labels(row, task), axis=1)

    # Remove raw prediction column if it exists
    if 'prediction' in display_df.columns and 'predicted_class' in display_df.columns:
        display_df.drop(columns=['prediction'], inplace=True)
    # Simple column renaming
    if 'header' in display_df.columns:
        display_df.rename(columns={'header': 'Protein Name'}, inplace=True)
    if 'sequence' in display_df.columns:
        display_df.rename(columns={'sequence': 'Sequence'}, inplace=True)
    if 'predicted_class' in display_df.columns:
        display_df.rename(columns={'predicted_class': 'Predicted Class'}, inplace=True)
    if 'probabilities' in display_df.columns:
        display_df.rename(columns={'probabilities': 'Confidence Score'}, inplace=True)
    if 'Dataset' in display_df.columns:
        first_dataset = display_df['Dataset'].iloc[0] if len(display_df) > 0 else None
        first_task = DATASET_TO_TASK_MAP.get(first_dataset) if first_dataset else None
        
        if first_task in REGRESSION_TASKS_FUNCTION:
            if 'prediction' in display_df.columns:
                display_df.rename(columns={'prediction': 'Predicted Value'}, inplace=True)
                display_df['Predicted Value'] = display_df['Predicted Value'].round(2)
        else:
            if 'predicted_class' in display_df.columns:
                display_df.rename(columns={'predicted_class': 'Predicted Class'}, inplace=True)
            if 'probabilities' in display_df.columns:
                display_df.rename(columns={'probabilities': 'Confidence Score'}, inplace=True)
        
        display_df.rename(columns={'Dataset': 'Dataset'}, inplace=True)
    
    # print(f"After simple rename, display_df columns: {list(display_df.columns)}")
    # print(f"Display_df shape: {display_df.shape}")

    ai_summary = "AI Analysis disabled. Enable in settings to generate a report."
    ai_response = "AI Analysis disabled." 

    expert_analysis = "<div style='height: 300px; display: flex; align-items: center; justify-content: center; color: #666;'>Analysis will appear here once prediction is complete...</div>"

    if enable_ai:
        progress(0.8, desc="Generating AI summary...")
        yield (
            "🤖 Expert is analyzing results...", 
            display_df, 
            gr.update(visible=False), 
            expert_analysis,
            "AI Analysis in progress..."
        )
        api_key = get_api_key(ai_model, user_api_key)
        if not api_key: 
            ai_summary = f"❌ No API key found for {ai_model}."
        else:
            ai_config = AIConfig(api_key, ai_model, AI_MODELS[ai_model]["api_base"], AI_MODELS[ai_model]["model"])
            prompt = generate_expert_analysis_prompt(display_df, task)
            ai_response = call_ai_api(ai_config, prompt)
            expert_analysis = format_expert_response(ai_response)
        progress(0.9, desc="Finalizing AI analysis...")
    else:
        progress(1.0, desc="Complete!")
    
    tar_path_str = ""
    try:
        timestamp = str(int(time.time()))
        tar_dir = get_save_path("Protein_Function", "Result")

        # Save only the processed results
        processed_df_for_save = display_df.copy()
        csv_path = tar_dir / f"Result_{timestamp}.csv"
        processed_df_for_save.to_csv(csv_path, index=False)
        # print(f"Saved CSV to: {csv_path}")

        # Create simple tar.gz with just the CSV file
        files_to_tar = {str(csv_path): csv_path.name}
        tar_path = tar_dir / f"func_pred_{timestamp}.tar.gz"
        tar_path_str = create_tar_archive(files_to_tar, str(tar_path))
        # print(f"Created tar.gz file: {tar_path_str}")
    except Exception as e: 
        print(f"Error creating tar.gz file: {e}")
        tar_path_str = ""

    final_status = "✅ All predictions completed!"
    if is_voting_run: 
        final_status += " Results were aggregated using soft voting."
    if enable_ai and not ai_summary.startswith("❌"): 
        final_status += " AI analysis included."
    
    # print(f"Final status: {final_status}")
    # print(f"Display DF shape: {display_df.shape}")
    # print(f"Archive path: {tar_path_str}")
    # print(f"About to yield final results...")
    
    progress(1.0, desc="Complete!")
    # print("Progress set to 100%, about to yield...")
    
    # Simple yield without try-catch to avoid any issues
    yield (
        final_status, 
        display_df, 
        gr.update(visible=True, value=tar_path_str) if tar_path_str else gr.update(visible=False), 
        expert_analysis, ai_response
    )
    # print("Final yield completed successfully!")


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
            text=f"<b>Functional Residue Prediction</b> - {task}",
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
    ai_model: str,
    user_api_key: Optional[str] = None,
    model_name: Optional[str] = None,
    progress=gr.Progress()
) -> Generator:
    try:
        import requests
        requests.post("/api/stats/track", json={"module": "function_analysis"})
    except Exception:
        pass

    """Handle protein residue function prediction workflow."""
    model = model_name if model_name else "ESM2-650M"

    if not all([task, fasta_file]):
        yield(
           "❌ Error: Task and FASTA file are required.", 
            pd.DataFrame(), None,
            gr.update(visible=False), 
            "Please provide all required inputs.",
            "AI Analysis disabled." 
        )
        return

    all_results_list = []
    timestamp = str(int(time.time()))
    residue_save_dir = get_save_path("Residue_Prediction", "Result")

    # Run predictions for each dataset
    progress(0.2, desc="Running Predicrtion...")
    yield(
        f"⏳ Running prediction...", 
        pd.DataFrame(), None,
        gr.update(visible=False), 
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
            script_path = Path("src") / "property" / f"{model_key}.py"
            adapter_path = Path("ckpt") / dataset / adapter_key
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
            subprocess.run(cmd, capture_output=True, text=True, check=True, encoding='utf-8', errors='ignore')
            
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
            gr.update(visible=False),
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
        plot_update = gr.update(value=residue_plot, visible=True)
    except Exception as e:
        print(f"Visualization error: {e}")
        plot_update = gr.update(visible=False)
    
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

    expert_analysis = "<div style='height: 300px; display: flex; align-items: center; justify-content: center; color: #666;'>Analysis will appear here once prediction is complete...</div>"

    if enable_ai:
        progress(0.8, desc="Generate AI summary...")
        yield (
            "🤖 Expert is analyzing results...", 
            display_df, None,
            gr.update(visible=False), 
            expert_analysis,
            "AI Analysis in progress..."
        )
        api_key = get_api_key(ai_model, user_api_key)
        if not api_key: 
            ai_summary = f"❌ No API key found for {ai_model}."
        else:
            ai_config = AIConfig(api_key, ai_model, AI_MODELS[ai_model]["api_base"], AI_MODELS[ai_model]["model"])
            prompt = generate_expert_analysis_prompt_residue(display_df, task)
            ai_response = call_ai_api(ai_config, prompt)
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
        gr.update(visible=True, value=download_path),
        expert_analysis,
        ai_response
    )


def run_protein_properties_prediction(task_type: str, file_path: str) -> Tuple[str, str]:
    """Run protein properties prediction"""
    try:
        timestamp = str(int(time.time()))
        properties_dir = get_save_path("Protein_Properties", "Result")
        output_json = properties_dir / f"{task_type.replace(' ', '_').replace('(', '').replace(')', '')}_{timestamp}.json"
        script_name = PROTEIN_PROPERTIES_MAP_FUNCTION.get(task_type)
        if not script_name:
           return "", f"Error: Task '{task_type}' is not allowed"
       
        script_path = f"src/property/{script_name}.py"
        if not os.path.exists(script_path):
           return "", f"Script not found: {script_path}"

        # Determine file argument based on file extension
        file_argument = "--fasta_file" if file_path.lower().endswith((".fasta", ".fa")) else "--pdb_file"
       
        cmd_save = [
           sys.executable, script_path,
           file_argument, file_path,
           "--chain_id", "A",
           "--output_file", str(output_json)
        ]
       
        # Run the command and capture output for debugging
        result = subprocess.run(
           cmd_save,
           capture_output=True,
           text=True,
           encoding="utf-8",
           errors="ignore"
        )

        # Check if the command failed
        if result.returncode != 0:
            error_msg = f"Script execution failed (return code: {result.returncode})"
            if result.stderr:
                error_msg += f"\nStderr: {result.stderr}"
            if result.stdout:
                error_msg += f"\nStdout: {result.stdout}"
            return "", error_msg

        # Check if output file was created
        if os.path.exists(output_json):
            return output_json, ""
        else:
            # Provide more detailed error information
            error_msg = "JSON output file was not generated"
            if result.stdout:
                error_msg += f"\nScript output: {result.stdout}"
            if result.stderr:
                error_msg += f"\nScript errors: {result.stderr}"
            return "", error_msg

    except Exception as e:
        return "", f"Error: {str(e)}"

def format_physical_chemical_properties(data: dict) -> str:
    """Format physical and chemical properties results"""
    result = ""
    result += f"Sequence length: {data['sequence_length']} aa\n"
    result += f"Molecular weight: {data['molecular_weight'] / 1000:.2f} kDa\n"
    result += f"Theoretical pI: {data['theoretical_pI']}\n"
    result += f"Aromaticity: {data['aromaticity']}\n"
    result += f"Instability index: {data['instability_index']}\n"
    
    if data['instability_index'] > 40:
        result += "  ⚠️ Predicted as unstable protein\n"
    else:
        result += "  ✅ Predicted as stable protein\n"
    
    result += f"GRAVY: {data['gravy']}\n"
    
    ssf = data['secondary_structure_fraction']
    result += f"Secondary structure prediction: Helix={ssf['helix']}, Turn={ssf['turn']}, Sheet={ssf['sheet']}\n"
    
    return result


def format_rsa_results(data: dict) -> str:
    """Format RSA calculation results"""
    result = ""
    result += f"Exposed residues: {data['exposed_residues']}\n"
    result += f"Buried residues: {data['buried_residues']}\n"
    result += f"Total residues: {data['total_residues']}\n"
    
    # Sort residues by number for display
    def get_residue_number(item):
        """Get residue number from item for sorting."""
        return int(item[0])
    
    try:
        sorted_residues = sorted(data['residue_rsa'].items(), key=get_residue_number)
    except ValueError:
        sorted_residues = sorted(data['residue_rsa'].items())
    
    for res_id, res_data in sorted_residues:
        aa = res_data['aa']
        rsa = res_data['rsa']
        location = "Exposed (surface)" if rsa >= 0.25 else "Buried (core)"
        result += f"  Residue {res_id} ({aa}): RSA = {rsa:.3f} ({location})\n"
    return result


def format_sasa_results(data: dict) -> str:
    """Format SASA calculation results"""
    result = ""
    result += f"{'Chain':<6} {'Residue':<12} {'SASA (Ų)':<15}\n"
    
    for chain_id, chain_data in sorted(data['chains'].items()):
        result += f"--- Chain {chain_id} (Total SASA: {chain_data['total_sasa']:.2f} Ų) ---\n"
        
        # Sort residues by number for display
        def get_residue_number_sasa(item):
            """Get residue number from item for sorting."""
            return int(item[0])
        
        try:
            sorted_residues = sorted(chain_data['residue_sasa'].items(), key=get_residue_number_sasa)
        except ValueError:
            sorted_residues = sorted(chain_data['residue_sasa'].items())
        
        for res_num, res_data in sorted_residues:
            res_id_str = f"{res_data['resname']}{res_num}"
            result += f"{chain_id:<6} {res_id_str:<12} {res_data['sasa']:<15.2f}\n"

    return result


def format_secondary_structure_results(data: dict) -> str:
    """Format secondary structure calculation results"""
    result = f"Successfully calculated secondary structure for chain '{data['chain_id']}'\n"
    result += f"Sequence length: {len(data['aa_sequence'])}\n"
    result += f"Helix (H): {data['ss_counts']['helix']} ({data['ss_counts']['helix']/len(data['aa_sequence'])*100:.1f}%)\n"
    result += f"Sheet (E): {data['ss_counts']['sheet']} ({data['ss_counts']['sheet']/len(data['aa_sequence'])*100:.1f}%)\n"
    result += f"Coil (C): {data['ss_counts']['coil']} ({data['ss_counts']['coil']/len(data['aa_sequence'])*100:.1f}%)\n"
    
    # Sort residues by number for display
    def get_residue_number_ss(item):
        """Get residue number from item for sorting."""
        return int(item[0])
    
    try:
        sorted_residues = sorted(data['residue_ss'].items(), key=get_residue_number_ss)
    except ValueError:
        sorted_residues = sorted(data['residue_ss'].items())
    
    for res_id, res_data in sorted_residues:
        result += f"  Residue {res_id} ({res_data['aa_seq']}): ss8: {res_data['ss8_seq']} ({res_data['ss8_name']}), ss3: {res_data['ss3_seq']}\n"
    
    result += f"aa_seq: {data['aa_sequence']}\n"
    result += f"ss8_seq: {data['ss8_sequence']}\n"
    result += f"ss3_seq: {data['ss3_sequence']}\n"
    
    return result


def handle_protein_properties_generation(
    task: str,
    file_obj: Any,
    progress = gr.Progress()
    ) -> Generator:
    """
    Handle protein properties generation with progress updates
    """
    try:
        if not file_obj or not task:
            yield (
                "❌ Error: Function and file are required.", 
                None, gr.update(visible=False), None, 
                "Please select a function and upload a file."
            )
            return
        
        if isinstance(file_obj, str):
            file_path = file_obj
        else:
            file_path = file_obj.name
        
        if file_path.lower().endswith((".fasta", ".fa")):
            if "PDB only" in task:
                yield (
                    "❌ Error: Unsupported task.", 
                    None, gr.update(visible=False), None, 
                    f"Task '{task}' requires a PDB file, but a FASTA file was provided."
                )
                return
        
        yield (
            f"⏳ Running prediction...", 
            None, gr.update(visible=False), None, 
            "Prediction in progress..."     
        )
        progress(0.1, desc="Running prediction...")
        output_json_path, error_message = run_protein_properties_prediction(task, file_path)
        progress(0.7, desc="Processing results...")
        
        if error_message:
            yield (f"❌ Error: {error_message}", None, gr.update(visible=False),
                None, error_message
            )
            return
        
        progress(0.9, desc="Finalizing results...")
        
        # Load and format results based on task type
        formatted_result = ""
        if output_json_path and os.path.exists(output_json_path):
            try:
                with open(output_json_path, 'r') as f:
                    results_data = json.load(f)
                
                # Format results based on the task type
                if task == "Physical and chemical properties":
                    formatted_result = format_physical_chemical_properties(results_data)
                elif task == "Relative solvent accessible surface area (PDB only)":
                    formatted_result = format_rsa_results(results_data)
                elif task == "SASA value (PDB only)":
                    formatted_result = format_sasa_results(results_data)
                elif task == "Secondary structure (PDB only)":
                    formatted_result = format_secondary_structure_results(results_data)
                else:
                    formatted_result = f"Results saved to: {os.path.basename(output_json_path)}"
                    
            except Exception as e:
                formatted_result = f"Results saved to: {os.path.basename(output_json_path)}\n(Error formatting display: {str(e)})"
        
        download_visible = bool(output_json_path and os.path.exists(output_json_path))
        download_update = gr.update(visible=download_visible,
                                    value = output_json_path if download_visible else None)
        final_status = f"✅ {task} completed successfully!"
        if output_json_path:
            final_status += f" Results saved to {os.path.basename(output_json_path)}"
        
        progress(1.0, desc="Complete!")

        yield (final_status, formatted_result, download_update,
                output_json_path if download_visible else None,
                "Prediction completed successfully.")

    except Exception as e:
        yield (
            f"❌ Unexpected error: {str(e)}", None, gr.update(visible=False),
            None, f"An unexpected error occurred: {str(e)}"
        )



def create_quick_tool_tab(constant: Dict[str, Any]) -> Dict[str, Any]:
    zero_shot_model = list(MODEL_MAPPING_ZERO_SHOT.keys())
    with gr.Blocks() as demo:
        with gr.Tabs():
            with gr.TabItem("Intelligent Directed Evolution"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=2):
                        gr.Markdown("### Model Configuration")
                        zero_shot_function_dd = gr.Dropdown(choices=DATASET_MAPPING_ZERO_SHOT, label="Select Protein Function", value=DATASET_MAPPING_ZERO_SHOT[0])
                        zero_shot_model_dd = gr.Dropdown(choices=zero_shot_model, label="Select Structure-based Model", visible=False)
                        gr.Markdown("**Data Input**")
                        with gr.Tabs():
                            with gr.TabItem("Upload Protein File (.fasta, .fa, .pdb)"):
                                easy_zshot_file_upload = gr.File(label="Upload Protein File (.fasta, .fa, .pdb)", file_types=[".fasta", ".fa", ".pdb"])
                                easy_zshot_file_example = gr.Examples(examples=[["./download/P60002.fasta"]], inputs=easy_zshot_file_upload, label="Click example to load")
                            with gr.TabItem("Paste Protein Content"):
                                easy_zshot_paste_content_input =  gr.Textbox(label="Paste Protein Content", placeholder="Paste protein content here...", lines=8, max_lines=15)
                                with gr.Row():
                                    easy_zshot_paste_content_btn = gr.Button("🔍 Detect Content", variant="primary", size="m")
                                    easy_zshot_paste_clear_btn = gr.Button("🗑️ Clear", variant="primary", size="m")

                        easy_zshot_protein_display = gr.Textbox(label="Uploaded Protein Sequence", interactive=False, lines=3, max_lines=7)
                        easy_zshot_sequence_selector = gr.Dropdown(label="Select Chain", choices=["Sequence_1"], value="Sequence_1", visible=False, allow_custom_value=True)
                        easy_zshot_original_file_path_state = gr.State("")
                        easy_zshot_original_paste_content_state = gr.State("")
                        easy_zshot_selected_sequence_state = gr.State("Sequence_1")
                        easy_zshot_sequence_state = gr.State({})
                        easy_zshot_current_file_state = gr.State("")
                        
                        gr.Markdown("### Configure AI Analysis (Optional)")
                        with gr.Accordion("AI Settings", open=True):
                            enable_ai_zshot = gr.Checkbox(label="Enable AI Summary", value=False)
                            with gr.Group(visible=False) as ai_box_zshot:
                                ai_model_dd_zshot = gr.Dropdown(
                                    choices=list(AI_MODELS.keys()), 
                                    value="DeepSeek", 
                                    label="Select AI Model"
                                )
                                ai_status_zshot = gr.Markdown(
                                    value="✓ Using provided API Key" if os.getenv("DEEPSEEK_API_KEY") else "⚠ No API Key found in .env file",
                                    visible=True
                                )
                                api_key_in_zshot = gr.Textbox(
                                    label="API Key", 
                                    type="password", 
                                    placeholder="Enter your API Key if needed",
                                    visible=not bool(os.getenv("DEEPSEEK_API_KEY"))
                                )
                        easy_zshot_predict_btn = gr.Button("🚀 Start Prediction", variant="primary")

                    with gr.Column(scale=3):
                        gr.Markdown("### Results")
                        zero_shot_status_box = gr.Textbox(label="Status", interactive=False)
                        
                        with gr.Tabs():
                            with gr.TabItem("📊 Raw Results"):
                                zero_shot_df_out = gr.DataFrame(label="Raw Data")
                            with gr.TabItem("📈 Prediction Heatmap"):
                                zero_shot_plot_out = gr.Plot(label="Heatmap")
                            with gr.TabItem("👨‍🔬 AI Expert Analysis"):
                                # function_results_plot = gr.Plot(label="Confidence Scores")
                                zero_shot_ai_expert_html = gr.HTML(
                                    value="<div style='height: 300px; display: flex; align-items: center; justify-content: center; color: #666;'>AI analysis will appear here...</div>",
                                    label="👨‍🔬 AI Expert Analysis"
                                )                        
                        zero_shot_download_btn = gr.DownloadButton("💾 Download Results", visible=False)
                        zero_shot_download_path_state = gr.State()
                        zero_shot_view_controls = gr.State() # Placeholder for potential future controls
                        zero_shot_full_data_state = gr.State()

            with gr.TabItem("Protein Function Prediction"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=2):
                        gr.Markdown("**Model Configuration**")
                        easy_func_task_dd = gr.Dropdown(choices=list(DATASET_MAPPING_FUNCTION.keys()), label="Select Task", value="Solubility")
                        gr.Markdown("**Data Input**")
                        with gr.Tabs():
                            with gr.TabItem("Upload FASTA File"):
                                base_function_fasta_upload = gr.File(label="Upload FASTA file", file_types=[".fasta", ".fa"])
                                base_function_file_example = gr.Examples(examples=[["./download/P60002.fasta"]], inputs=base_function_fasta_upload, label="Click example to load")
                            with gr.TabItem("Paste FASTA Content"):
                                base_func_paste_content_input = gr.Textbox(label="Paste FASTA Content", placeholder="Paste FASTA content here...", lines=8, max_lines=15)
                                with gr.Row():
                                    base_func_paste_content_btn = gr.Button("🔍 Detect & Save Content", variant="primary", size="m")
                                    base_func_paste_clear_btn = gr.Button("🗑️ Clear", variant="primary", size="m")
                        
                        base_function_protein_display = gr.Textbox(label="Uploaded Protein Sequence", interactive=False, lines=3, max_lines=7)
                        base_function_selector = gr.Dropdown(label="Select Chain", choices=["Sequence_1"], value="Sequence_1", visible=False, allow_custom_value=True)
                        base_function_original_file_path_state = gr.State("")
                        base_function_original_paste_content_state = gr.State("")
                        base_function_selected_sequence_state = gr.State("Sequence_1")
                        base_function_sequence_state = gr.State({})
                        base_function_current_file_state = gr.State("")

                        gr.Markdown("### Configure AI Analysis (Optional)")
                        with gr.Accordion("AI Settings", open=True):
                            enable_ai_func = gr.Checkbox(label="Enable AI Summary", value=False)
                            with gr.Accordion("AI Settings", open=True):
                                with gr.Group(visible=False) as ai_box_func:
                                    ai_model_dd_func = gr.Dropdown(
                                        choices=list(AI_MODELS.keys()), 
                                        label="Select AI Model", 
                                        value="DeepSeek"
                                    )
                                    ai_status_func = gr.Markdown(
                                        value="✓ Using provided API Key" if os.getenv("DEEPSEEK_API_KEY") else "⚠ No API Key found in .env file",
                                        visible=True
                                    )
                                    api_key_in_func = gr.Textbox(
                                        label="API Key", 
                                        type="password", 
                                        placeholder="Enter your API Key if needed",
                                        visible=not bool(os.getenv("DEEPSEEK_API_KEY"))
                                    )
                        easy_func_predict_btn = gr.Button("🚀 Start Prediction", variant="primary")
                
                    with gr.Column(scale=3):
                        gr.Markdown("### Results")
                        function_status_textbox = gr.Textbox(label="Status", interactive=False)
                        with gr.Tabs():
                            with gr.TabItem("📊 Raw Results"):
                                function_results_df = gr.DataFrame(label="Prediction Data", column_widths=["20%", "20%", "20%", "20%", "20%"])
                            with gr.TabItem("👨‍🔬 AI Expert Analysis"):
                                function_ai_expert_html = gr.HTML(
                                    value="<div style='height: 300px; display: flex; align-items: center; justify-content: center; color: #666;'>AI analysis will appear here...</div>",
                                    label="👨‍🔬 AI Expert Analysis"
                                )
                        function_download_btn = gr.DownloadButton("💾 Download Results", visible=False)

            with gr.TabItem("Functional Residue Prediction"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=2):
                        gr.Markdown("### Model Configuration")
                        base_residue_function_task_dd = gr.Dropdown(choices=list(RESIDUE_MAPPING_FUNCTION.keys()), label="Select Task", value="Activity Site")
                        gr.Markdown("**Data Input**")
                        with gr.Tabs():
                            with gr.TabItem("Upload FASTA File"):
                                base_residue_function_fasta_upload = gr.File(label="Upload Fasta file", file_types=[".fasta", ".fa"])
                                base_residue_function_file_exmaple = gr.Examples(examples=[["./download/P60002.fasta"]], inputs=base_residue_function_fasta_upload, label="Click example to load")
                            with gr.TabItem("Paste FASTA Content"):
                                base_residue_function_paste_content_input = gr.Textbox(label="Paste FASTA Content", placeholder="Paste FASTA content here...", lines=8, max_lines=15)
                                with gr.Row():
                                    base_residue_function_paste_content_btn = gr.Button("🔍 Detect & Save Content", variant="primary", size="m")
                                    base_residue_function_paste_clear_btn = gr.Button("🗑️ Clear", variant="primary", size="m")
                        
                        base_residue_function_protein_display = gr.Textbox(label="Uploaded Protein Sequence", interactive=False, lines=3, max_lines=7)
                        base_residue_function_selector = gr.Dropdown(label="Select Chain", choices=["Sequence_1"], value="Sequence_1", visible=False, allow_custom_value=True)
                        base_residue_function_original_file_path_state = gr.State("")
                        base_residue_function_original_paste_content_state = gr.State("")
                        base_residue_function_selected_sequence_state = gr.State("Sequence_1")
                        base_residue_function_sequence_state = gr.State({})
                        base_residue_function_current_file_state = gr.State("")

                        gr.Markdown("### Configure AI  Analysis (Optional)")
                        with gr.Accordion("AI Settings", open=True):
                            enable_ai_residue_function = gr.Checkbox(label="Enable AI Summary", value=False)
                            with gr.Accordion("AI  Settings", open=True):
                                with gr.Group(visible=False) as ai_box_residue_function:
                                    ai_model_dd_residue_function = gr.Dropdown(
                                        choices=list(AI_MODELS.keys()),
                                        label="Select AI Model",
                                        value="DeepSeek"
                                    )
                                    ai_status_residue_function = gr.Markdown(
                                        value="✓ Using provided API Key" if os.getenv("DEEPSEEK_API_KEY") else "⚠ No API Key found in .env file",
                                        visible=True
                                    )
                                    api_key_in_residue_function = gr.Textbox(
                                        label="API Key",
                                        type="password",
                                        placeholder="Ener your API Key if needed",
                                        visible=not bool(os.getenv("DEEPSEEK_API_KEY"))
                                    )
                        base_residue_function_predict_btn = gr.Button("🚀 Start Prediction", variant="primary")
                    
                    with gr.Column(scale=3):
                        gr.Markdown("### Results")
                        base_residue_function_status_textbox = gr.Textbox(label="Status", interactive=False)
                        with gr.Tabs():
                            with gr.TabItem("📊 Raw Results"):
                                base_residue_function_results_df = gr.DataFrame(label="Prediction Data", column_widths=["20%", "20%", "20%", "20%", "20%"])
                            with gr.TabItem("📈 Prediction Heatmap"):
                                base_residue_function_plot_out = gr.Plot(label="Heatmap")
                            with gr.TabItem("👨‍🔬 AI Expert Analysis"):
                                base_residue_function_ai_expert_html = gr.HTML(
                                    value="<div style='height: 300px; display: flex; align-items: center; justify-content: center; color: #666;'>AI analysis will appear here...</div>",
                                    label="👨‍🔬 AI Expert Analysis"
                                )
                        base_residue_function_download_btn = gr.DownloadButton("💾 Download Results", visible=False)


            with gr.TabItem("Physicochemical Property Analysis"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=2):
                        gr.Markdown("**Task Configuration**")
                        protein_properties_choices = list(PROTEIN_PROPERTIES_FUNCTION)
                        protein_properties_task_dd = gr.Dropdown(choices=protein_properties_choices, label="Select Properties of Protein", value=protein_properties_choices[0])
                        
                        gr.Markdown("**Data Input**")
                        with gr.Tabs():
                            with gr.TabItem("Upload Protein File (.fasta, .fa, .pdb)"):
                                protein_properties_file_upload = gr.File(label="Upload Protein File (.fasta, .fa, .pdb)", file_types=[".fasta", ".fa", ".pdb"])
                                
                                protein_properties_file_example = gr.Examples(examples=[["./download/P60002.fasta"]], inputs=protein_properties_file_upload, label="Click example to load")

                            with gr.TabItem("Paste Protein Content"):
                                protein_properties_paste_content_input = gr.Textbox(label="Pasta Protein Content", placeholder="Paste protein content here...", lines=8, max_lines=15)
                                with gr.Row():
                                    protein_properties_paste_content_btn = gr.Button("🔍 Detect Content", variant="primary", size="m")
                                    protein_properties_paste_clear_btn = gr.Button("🗑️ Clear", variant="primary", size="m")
                        
                        
                        protein_properties_protein_display = gr.Textbox(label="Uploaded Protein Sequence", interactive=False, lines=3, max_lines=7)
                        protein_properties_sequence_selector = gr.Dropdown(label="Select Chain", choices=["Sequence_1"], value="Sequence_1", visible=False, allow_custom_value=True)
                        protein_properties_original_file_path_state = gr.State("")
                        protein_properties_original_paste_content_state = gr.State("")
                        protein_properties_selected_sequence_state = gr.State("Sequence_1")
                        protein_properties_sequence_state = gr.State({})
                        protein_properties_current_file_state = gr.State("")

                        protein_properties_predict_btn = gr.Button("🚀 Start Prediction", variant="primary")
                    with gr.Column(scale=3):
                        gr.Markdown("### Results")
                        protein_properties_status_box = gr.Textbox(label="Status", interactive=False)
                        
                        with gr.Tabs():
                            with gr.TabItem("📊 Raw Results"):
                                protein_properties_result_out = gr.Textbox(
                                                                label="Raw Data", 
                                                                interactive=False, 
                                                                lines=15
                                                            )
                        
                        protein_properties_download_btn = gr.DownloadButton("💾 Download Results", visible=False)
                        protein_properties_path_state = gr.State()
                        protein_properties_view_controls = gr.State()
                        protein_properties_full_data_state = gr.State()        
        enable_ai_zshot.change(fn=toggle_ai_section, inputs=enable_ai_zshot, outputs=ai_box_zshot)
        enable_ai_func.change(fn=toggle_ai_section, inputs=enable_ai_func, outputs=ai_box_func)

        ai_model_dd_zshot.change(
            fn=on_ai_model_change,
            inputs=ai_model_dd_zshot,
            outputs=[api_key_in_zshot, ai_status_zshot]
        )
        ai_model_dd_func.change(
            fn=on_ai_model_change,
            inputs=ai_model_dd_func,
            outputs=[api_key_in_func, ai_status_func]
        )
        enable_ai_residue_function.change(
            fn=toggle_ai_section, 
            inputs=enable_ai_residue_function, 
            outputs=ai_box_residue_function
        )

        ai_model_dd_residue_function.change(
            fn=on_ai_model_change,
            inputs=ai_model_dd_residue_function,
            outputs=[api_key_in_residue_function, ai_status_residue_function]
        )


        easy_zshot_file_upload.upload(
            fn=handle_file_upload, 
            inputs=easy_zshot_file_upload, 
            outputs=[easy_zshot_protein_display, easy_zshot_sequence_selector, easy_zshot_sequence_state, easy_zshot_selected_sequence_state, easy_zshot_original_file_path_state, easy_zshot_current_file_state]
        )
        easy_zshot_file_upload.change(
            fn=handle_file_upload, 
            inputs=easy_zshot_file_upload, 
            outputs=[easy_zshot_protein_display, easy_zshot_sequence_selector, easy_zshot_sequence_state, easy_zshot_selected_sequence_state, easy_zshot_original_file_path_state, easy_zshot_current_file_state]
        )
        easy_zshot_paste_clear_btn.click(
            fn=clear_paste_content_fasta,
            outputs=[easy_zshot_paste_content_input, easy_zshot_protein_display, easy_zshot_sequence_selector, easy_zshot_sequence_state, easy_zshot_selected_sequence_state, easy_zshot_original_file_path_state]
        )
        easy_zshot_paste_content_btn.click(
            fn=parse_fasta_paste_content,
            inputs=easy_zshot_paste_content_input,
            outputs=[easy_zshot_protein_display, easy_zshot_sequence_selector, easy_zshot_sequence_state, easy_zshot_selected_sequence_state, easy_zshot_original_file_path_state, easy_zshot_original_paste_content_state]
        )

        easy_zshot_sequence_selector.change(
            fn=handle_sequence_change_unified,
            inputs=[easy_zshot_sequence_selector, easy_zshot_sequence_state, easy_zshot_original_file_path_state, easy_zshot_original_paste_content_state],
            outputs=[easy_zshot_protein_display, easy_zshot_file_upload]
        )

        easy_zshot_predict_btn.click(
            fn=handle_mutation_prediction_base,
            inputs=[zero_shot_function_dd, easy_zshot_file_upload, enable_ai_zshot, ai_model_dd_zshot, api_key_in_zshot, zero_shot_model_dd],
            outputs=[zero_shot_status_box, zero_shot_plot_out, zero_shot_df_out, zero_shot_download_btn, zero_shot_download_path_state, zero_shot_view_controls, zero_shot_full_data_state, zero_shot_ai_expert_html],
            show_progress=True
        )

        base_function_fasta_upload.upload(
            fn=handle_file_upload, 
            inputs=base_function_fasta_upload, 
            outputs=[base_function_protein_display, base_function_selector, base_function_sequence_state, base_function_selected_sequence_state, base_function_original_file_path_state, base_function_current_file_state]
        )
        base_function_fasta_upload.change(
            fn=handle_file_upload, 
            inputs=base_function_fasta_upload, 
            outputs=[base_function_protein_display, base_function_selector, base_function_sequence_state, base_function_selected_sequence_state, base_function_original_file_path_state, base_function_current_file_state]
        )

        base_func_paste_clear_btn.click(
            fn=clear_paste_content_fasta,
            outputs=[base_func_paste_content_input, base_function_protein_display, base_function_selector, base_function_sequence_state, base_function_selected_sequence_state, base_function_original_file_path_state]
        )

        base_func_paste_content_btn.click(
            fn=parse_fasta_paste_content,
            inputs=base_func_paste_content_input,
            outputs=[base_function_protein_display, base_function_selector, base_function_sequence_state, base_function_selected_sequence_state, base_function_original_file_path_state, base_function_original_paste_content_state]
        )

        base_function_selector.change(
            fn=handle_sequence_change_unified,
            inputs=[base_function_selector, base_function_sequence_state, base_function_original_file_path_state, base_function_original_paste_content_state],
            outputs=[base_function_protein_display, base_function_fasta_upload]
        )

        easy_func_predict_btn.click(
            fn=handle_protein_function_prediction,
            inputs=[easy_func_task_dd, base_function_fasta_upload, enable_ai_func, ai_model_dd_func, api_key_in_func],
            outputs=[function_status_textbox, function_results_df, function_download_btn, function_ai_expert_html, gr.State()], 
            show_progress=True
        )

        base_residue_function_fasta_upload.upload(
            fn=handle_file_upload,
            inputs=base_residue_function_fasta_upload,
            outputs=[base_residue_function_protein_display, base_residue_function_selector, base_residue_function_sequence_state, base_residue_function_selected_sequence_state, base_residue_function_original_file_path_state, base_residue_function_current_file_state]
        )
        base_residue_function_fasta_upload.change(
            fn=handle_file_upload,
            inputs=base_residue_function_fasta_upload,
            outputs=[base_residue_function_protein_display, base_residue_function_selector, base_residue_function_sequence_state, base_residue_function_selected_sequence_state, base_residue_function_original_file_path_state, base_residue_function_current_file_state]
        )
        base_residue_function_paste_clear_btn.click(
            fn=clear_paste_content_fasta,
            outputs=[base_residue_function_paste_content_input, base_residue_function_protein_display, base_residue_function_selector, base_residue_function_sequence_state, base_residue_function_selected_sequence_state, base_residue_function_original_file_path_state]
        )
        base_residue_function_paste_content_btn.click(
            fn=parse_fasta_paste_content,
            inputs=base_residue_function_paste_content_input,
            outputs=[base_residue_function_protein_display, base_residue_function_selector, base_residue_function_sequence_state, base_residue_function_selected_sequence_state, base_residue_function_original_file_path_state, base_residue_function_original_paste_content_state]
        )
        base_residue_function_selector.change(
            fn=handle_sequence_change_unified,
            inputs=[base_residue_function_selector, base_residue_function_sequence_state, base_residue_function_original_file_path_state, base_residue_function_original_paste_content_state],
            outputs=[base_residue_function_protein_display, base_residue_function_fasta_upload]
        )
        base_residue_function_predict_btn.click(
            fn=handle_protein_residue_function_prediction,
            inputs=[base_residue_function_task_dd, base_residue_function_fasta_upload, enable_ai_residue_function, ai_model_dd_residue_function, api_key_in_residue_function],
            outputs=[base_residue_function_status_textbox, base_residue_function_results_df, base_residue_function_plot_out,base_residue_function_download_btn, base_residue_function_ai_expert_html, gr.State()],
            show_progress = True
        )

        protein_properties_file_upload.upload(
            fn=handle_file_upload,
            inputs=protein_properties_file_upload,
            outputs=[protein_properties_protein_display, protein_properties_sequence_selector, protein_properties_sequence_state, protein_properties_selected_sequence_state, protein_properties_original_file_path_state, protein_properties_current_file_state]
        )
        protein_properties_file_upload.change(
            fn=handle_file_upload,
            inputs=protein_properties_file_upload,
            outputs=[protein_properties_protein_display, protein_properties_sequence_selector, protein_properties_sequence_state, protein_properties_selected_sequence_state, protein_properties_original_file_path_state, protein_properties_current_file_state]
        )
        protein_properties_paste_clear_btn.click(
            fn=clear_paste_content_fasta,
            outputs=[protein_properties_paste_content_input, protein_properties_protein_display, protein_properties_sequence_selector, protein_properties_sequence_state, protein_properties_selected_sequence_state, protein_properties_original_file_path_state]
        )
        protein_properties_paste_content_btn.click(
            fn=parse_fasta_paste_content,
            inputs=protein_properties_paste_content_input,
            outputs=[protein_properties_protein_display, protein_properties_sequence_selector, protein_properties_sequence_state, protein_properties_selected_sequence_state, protein_properties_original_file_path_state, protein_properties_original_paste_content_state]
        )

        protein_properties_sequence_selector.change(
            fn=handle_sequence_change_unified,
            inputs=[protein_properties_sequence_selector, protein_properties_sequence_state, protein_properties_original_file_path_state, protein_properties_original_paste_content_state],
            outputs=[protein_properties_protein_display, protein_properties_file_upload]
        )

        protein_properties_predict_btn.click(
            fn=handle_protein_properties_generation,
            inputs=[protein_properties_task_dd, protein_properties_file_upload],
            outputs=[protein_properties_status_box, protein_properties_result_out, protein_properties_download_btn, protein_properties_path_state, gr.State()],
            show_progress=True
        )

    return demo