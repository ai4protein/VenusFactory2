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
from mcp_tools.common.constants import LLM_MODELS
from mcp_tools.common.common_utils import create_tar_archive, get_save_path
from mcp_tools.common.file_handlers import process_fasta_file
from mcp_tools.common.llm_helpers import LLMConfig, call_llm_api, format_expert_response, generate_mutation_ai_prompt, get_api_key, get_chat_base_url
from mcp_tools.common.data_processors import get_total_residues_count, prepare_top_residue_heatmap_data
from mcp_tools.common.visualization import generate_plotly_heatmap
from mcp_tools.common.prediction_runners import run_zero_shot_prediction
from mcp_tools.common.report_fragments import load_html_template
from mcp_tools.common.results import ignore_progress, result_patch
from mcp_tools.common.common_utils import get_project_root
PROJECT_ROOT = get_project_root()

def handle_mutation_prediction_advance(
    function_selection: Optional[str],
    file_obj: Any,
    enable_ai: bool,
    llm_model: str,
    user_api_key: str,
    model_name:str,
    progress=ignore_progress
) -> Generator:
    """Handle mutation prediction workflow."""
    effective_function_selection = (function_selection or "Model-guided optimization").strip() or "Model-guided optimization"

    if not file_obj:
        yield (
            "❌ Error: Input file is required.",
            None, None, result_patch(visible=False), None,
            result_patch(visible=False), None,
            "Please upload a valid sequence or structure file."
        )
        return

    if isinstance(file_obj, str):
        file_path = file_obj
    else:
        file_path = file_obj.name
    if file_path.lower().endswith(".pdb"):
        if model_name:
            model_type = "structure"
        else:
            model_name, model_type = "ESM-IF1", "structure"
    elif file_path.lower().endswith((".fasta", ".fa")):
        if model_name:
            model_type = "sequence"
        else:
            model_name, model_type = "ESM2-650M", "sequence"

        processed_file_path = process_fasta_file(file_path)
        if processed_file_path != file_path:
            file_path = processed_file_path
            yield (
                "⚠️ Multi-sequence FASTA detected. Using only the first sequence for prediction.",
                None, None, result_patch(visible=False), None,
                result_patch(visible=False), None,
                "Processing first sequence only..."
            )
    else:
        yield (
            "❌ Error: Unsupported file type.",
            None, None, result_patch(visible=False), None,
            result_patch(visible=False), None,
            "Please upload a .fasta, .fa, or .pdb file."
        )
        return

    progress(0.1, desc="Running prediction...")
    yield (
        f"⏳ Running prediction...",
        None, None, result_patch(visible=False), None,
        result_patch(visible=False), None,
        "Prediction in progress..."
    )

    status, raw_df = run_zero_shot_prediction(model_type, model_name, file_path)
    progress(0.7, desc="Processing results...")
    if raw_df.empty:
        yield (
            status,
            go.Figure(layout={'title': 'No results generated'}),
            pd.DataFrame(), result_patch(visible=False), None,
            result_patch(visible=False), None,
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
            display_df, result_patch(visible=False), None,
            result_patch(visible=False), display_df,
            "Score column not found."
        )
        return

    summary_fig = generate_plotly_heatmap(*data_tuple[:4])
    expert_analysis = load_html_template("ai_expert_placeholder.html")

    ai_summary = "AI Analysis disabled. Enable in settings to generate a report."
    if enable_ai:
        progress(0.8, desc="Generating AI summary...")
        yield (
            f"✅ Prediction complete. 🤖 Generating AI summary...",
            summary_fig, display_df, result_patch(visible=False), None,
            result_patch(visible=total_residues > 20), display_df,
            expert_analysis
        )

        api_key = get_api_key(llm_model, user_api_key)
        if not api_key:
            ai_summary = "❌ No API key found."
        else:
            llm_config = LLMConfig(
                api_key, llm_model,
                get_chat_base_url(),
                LLM_MODELS.get(llm_model, "")
            )
            prompt = generate_mutation_ai_prompt(display_df, model_name, effective_function_selection)
            ai_summary = call_llm_api(llm_config, prompt)
            expert_analysis = format_expert_response(ai_summary)
        progress(0.9, desc="Finalizing AI analysis...")
    else:
        progress(1.0, desc="Complete!")

    timestamp = str(int(time.time()))
    session_dir = get_save_path("Zero_Shot", "Result")
    csv_path = session_dir/ f"mut_res_{timestamp}.csv"
    heatmap_path = session_dir/ f"mut_map_{timestamp}.html"

    display_df.to_csv(csv_path, index=False)
    summary_fig.write_html(heatmap_path)

    files_to_tar = {
        str(csv_path): "prediction_results.csv",
        str(heatmap_path): "prediction_heatmap.html"
    }

    if not ai_summary.startswith("❌") and not ai_summary.startswith("AI Analysis"):
        report_path = session_dir / f"ai_report_{timestamp}.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(ai_summary)
        files_to_tar[str(report_path)] = "AI_Analysis_Report.md"

    tar_path = session_dir / f"pred_mut_{timestamp}.tar.gz"
    tar_path_str = create_tar_archive(files_to_tar, str(tar_path))

    final_status = status if not enable_ai else "✅ Prediction and AI analysis complete!"
    progress(1.0, desc="Complete!")
    yield (
        final_status, summary_fig, display_df,
        result_patch(visible=True, value=tar_path_str), tar_path_str,
        result_patch(visible=total_residues > 20), display_df, expert_analysis
    )
