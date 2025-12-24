"""UI helper functions for Gradio components."""

import gradio as gr
from typing import Tuple

from .file_handlers import parse_fasta_paste_content, parse_pdb_paste_content


def create_progress_html(message: str = "Processing...") -> str:
    """Create a breathing progress indicator HTML."""
    return f"""
    <div style="text-align: center; padding: 20px;">
        <div style="display: inline-block;">
            <div style="width: 60px; height: 60px; border: 4px solid #f3f3f3; border-top: 4px solid #3498db; 
                        border-radius: 50%; animation: spin 1s linear infinite;"></div>
            <p style="margin-top: 10px; color: #666; font-weight: 500;">{message}</p>
        </div>
        <style>
            @keyframes spin {{
                0% {{ transform: rotate(0deg); }}
                100% {{ transform: rotate(360deg); }}
            }}
            @keyframes breathe {{
                0%, 100% {{ opacity: 0.6; }}
                50% {{ opacity: 1; }}
            }}
        </style>
    </div>
    """


def create_status_html(status: str, color: str = "#666") -> str:
    """Create a status indicator HTML."""
    return f"<div style='text-align: center; padding: 10px;'><span style='color: {color}; font-weight: 500;'>{status}</span></div>"


def handle_paste_fasta_detect(fasta_content: str) -> Tuple:
    """Handle paste FASTA content detection with file upload component update."""
    result = parse_fasta_paste_content(fasta_content)
    if len(result) >= 5:
        temp_file_path = result[4]  # The temporary file path
        return result + (temp_file_path,)
    return result + ("",)


def handle_paste_pdb_detect(pdb_content: str) -> Tuple:
    """Handle paste PDB content detection."""
    result = parse_pdb_paste_content(pdb_content)
    return result + (pdb_content, )


def update_dataset_choices_fixed(task: str):
    """Update dataset choices based on selected task."""
    from .constants import DATASET_MAPPING_FUNCTION
    choices = DATASET_MAPPING_FUNCTION.get(task, [])
    return gr.CheckboxGroup(choices=choices, value=choices)

