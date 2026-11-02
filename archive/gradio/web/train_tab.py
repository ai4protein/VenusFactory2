import os
import json
import gradio as gr
import time
from pathlib import Path
from datasets import load_dataset
from typing import Any, Dict, Generator, List
from dataclasses import dataclass
from .utils.command import preview_command, save_arguments, build_command_list
from .utils.monitor import TrainingMonitor
from .utils.html_ui import load_html_template
import traceback
import tempfile
import subprocess

@dataclass
class TrainingArgs:
    def __init__(self, args_dict: dict, plm_models: dict, dataset_configs: dict):
        # Basic parameters
        self.plm_model = plm_models[args_dict["plm_model"]]
        
        # Process dataset selection
        self.dataset_selection = args_dict["dataset_selection"]  # "Custom" 或 "Pre-defined"
        if self.dataset_selection == "Pre-defined":
            self.dataset_config = dataset_configs[args_dict["dataset_config"]]
            self.dataset_custom = None
            # load dataset config
            with open(self.dataset_config, 'r') as f:
                config = json.load(f)
            self.problem_type = config.get("problem_type", "single_label_classification")
            self.num_labels = config.get("num_labels", 2)
            self.metrics = config.get("metrics", "accuracy,mcc,f1,precision,recall,auroc")
            self.sequence_column_name = config.get("sequence_column_name", "aa_seq")
            self.label_column_name = config.get("label_column_name", "label")
        else:
            self.dataset_config = None
            self.dataset_custom = args_dict["dataset_custom"]  # Custom dataset path
            self.problem_type = args_dict["problem_type"]
            self.num_labels = args_dict["num_labels"]
            self.metrics = args_dict["metrics"]
            self.sequence_column_name = args_dict["sequence_column_name"]
            self.label_column_name = args_dict["label_column_name"]
            # if metrics is a list, convert to comma-separated string
            if isinstance(self.metrics, list):
                self.metrics = ",".join(self.metrics)
            
        # Training method parameters
        self.training_method = args_dict["training_method"]
        self.pooling_method = args_dict["pooling_method"]
        
        # Batch processing parameters
        self.batch_mode = args_dict["batch_mode"]
        if self.batch_mode == "Batch Size Mode":
            self.batch_size = args_dict["batch_size"]
        else:
            self.batch_token = args_dict["batch_token"]
        
        # Training parameters
        self.learning_rate = args_dict["learning_rate"]
        self.num_epochs = args_dict["num_epochs"]
        self.max_seq_len = args_dict["max_seq_len"]
        self.gradient_accumulation_steps = args_dict["gradient_accumulation_steps"]
        self.warmup_steps = args_dict["warmup_steps"]
        self.scheduler = args_dict["scheduler"]

        # Output parameters (ensure output_model_name ends with .pt)
        raw_name = args_dict["output_model_name"]
        self.output_model_name = raw_name if str(raw_name).strip().lower().endswith(".pt") else f"{str(raw_name).strip()}.pt"
        od = args_dict.get("output_dir", "demo")
        if od and not str(od).startswith("ckpt") and not os.path.isabs(str(od)):
            od = "ckpt/" + str(od)
        self.output_dir = od
        
        # Wandb parameters
        self.wandb_enabled = args_dict["wandb_enabled"]
        if self.wandb_enabled:
            self.wandb_project = args_dict["wandb_project"]
            self.wandb_entity = args_dict["wandb_entity"]
        
        # Other parameters
        self.patience = args_dict["patience"]
        self.num_workers = args_dict["num_workers"]
        self.max_grad_norm = args_dict["max_grad_norm"]
        self.structure_seq = args_dict["structure_seq"]
        self.pdb_dir = args_dict.get("pdb_dir", "") or ""

        # LoRA parameters
        self.lora_r = args_dict["lora_r"]
        self.lora_alpha = args_dict["lora_alpha"]
        self.lora_dropout = args_dict["lora_dropout"]
        self.lora_target_modules = [m.strip() for m in args_dict["lora_target_modules"].split(",")] if args_dict["lora_target_modules"] else []

        # Monitored parameters
        self.monitored_metrics = args_dict["monitored_metrics"]
        self.monitored_strategy = args_dict["monitored_strategy"]

    def to_dict(self) -> Dict[str, Any]:
        args_dict = {
            "plm_model": self.plm_model,
            "training_method": self.training_method,
            "pooling_method": self.pooling_method,
            "learning_rate": self.learning_rate,
            "num_epochs": self.num_epochs,
            "max_seq_len": self.max_seq_len,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "warmup_steps": self.warmup_steps,
            "scheduler": self.scheduler,
            "output_model_name": self.output_model_name,
            "output_dir": self.output_dir,
            "patience": self.patience,
            "num_workers": self.num_workers,
            "max_grad_norm": self.max_grad_norm,
            "monitor": self.monitored_metrics,
            "monitor_strategy": self.monitored_strategy
        }

        if self.training_method == "ses-adapter" and self.structure_seq:
            args_dict["structure_seq"] = ",".join(self.structure_seq)
        if self.pdb_dir and self.pdb_dir.strip():
            args_dict["pdb_dir"] = self.pdb_dir.strip()

        # add dataset related parameters
        if self.dataset_selection == "Pre-defined":
            args_dict["dataset_config"] = self.dataset_config
        else:
            args_dict["dataset"] = self.dataset_custom
            args_dict["problem_type"] = self.problem_type
            args_dict["num_labels"] = self.num_labels
            args_dict["metrics"] = self.metrics
        
        # Add column name parameters for both predefined and custom datasets
        args_dict["sequence_column_name"] = self.sequence_column_name
        args_dict["label_column_name"] = self.label_column_name

        # Add LoRA parameters
        if self.training_method in ["plm-lora", "plm-qlora", "plm-adalora", "plm-dora", "plm-ia3"]:
            args_dict.update({
                "lora_r": self.lora_r,
                "lora_alpha": self.lora_alpha,
                "lora_dropout": self.lora_dropout,
                "lora_target_modules": self.lora_target_modules
            })

        # Add batch processing parameters
        if self.batch_mode == "Batch Size Mode":
            args_dict["batch_size"] = self.batch_size
        else:
            args_dict["batch_token"] = self.batch_token

        # Add wandb parameters
        if self.wandb_enabled:
            args_dict["wandb"] = True
            if self.wandb_project:
                args_dict["wandb_project"] = self.wandb_project
            if self.wandb_entity:
                args_dict["wandb_entity"] = self.wandb_entity

        return args_dict

def create_train_tab(constant: Dict[str, Any]) -> Dict[str, Any]:
    # Create training monitor
    monitor = TrainingMonitor()

    # Add missing variable declarations
    is_training = False
    current_process = None
    stop_thread = False
    process_aborted = False

    plm_models = constant["plm_models"]
    dataset_configs = constant["dataset_configs"]

    # Create reverse mapping: full_path -> display_name
    plm_models_reverse = {v: k for k, v in plm_models.items()}

    # Helper functions for transfer learning model selection
    def scan_folders_under(root: str, max_depth: int = 5) -> list:
        """Scan for subdirectories under root."""
        if not root or not os.path.isdir(root):
            return []
        result = []
        root_path = Path(root)
        for path in sorted(root_path.rglob("*")):
            if not path.is_dir():
                continue
            try:
                rel = path.relative_to(root_path)
            except ValueError:
                continue
            if not rel.parts or len(rel.parts) > max_depth:
                continue
            path_str = str(path)
            result.append((path_str, path_str))
        return result

    def scan_models_in_folder(folder_path: str) -> list:
        """Scan folder for .pt model files."""
        if not folder_path or not os.path.isdir(folder_path):
            return []
        result = []
        folder = Path(folder_path)
        for pt_file in sorted(folder.rglob("*.pt")):
            name = pt_file.stem
            # Skip LoRA auxiliary files
            if any(s in name for s in ("_lora", "_qlora", "_dora", "_adalora", "_ia3")):
                continue
            result.append((str(pt_file.relative_to(folder)), str(pt_file)))
        return result

    def load_model_config(model_path: str) -> dict:
        """Load config from model's .json file."""
        if not model_path:
            return {}
        model_path = Path(model_path)
        config_path = model_path.parent / f"{model_path.stem}.json"
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    # Model and Dataset Selection
    gr.Markdown("## Model and Dataset Configuration")

    with gr.Accordion("📥 Import Training Configuration", open=False) as config_import_accordion:
        with gr.Row():
            with gr.Column(scale=4):
                config_path_input = gr.Textbox(
                    label="Configuration File Path",
                    placeholder="e.g., ./config.json",
                    value=""
                )
            with gr.Column(scale=1):
                import_config_button = gr.Button(
                    "Import Config",
                    variant="primary",
                    elem_classes=["import-config-btn"]
                )

    # Scan available model folders for Continue Training
    _folder_choices = [(base, base) for base in ("ckpt",) if os.path.isdir(base)] + scan_folders_under("ckpt")

    # Main training interface
    with gr.Group():
        # Training Mode and Model Selection
        with gr.Row():
            training_mode = gr.Radio(
                choices=["From Scratch", "Continue Training"],
                label="Training Mode",
                value="From Scratch",
                scale=1
            )

            plm_model = gr.Dropdown(
                choices=list(plm_models.keys()),
                label="Protein Language Model",
                value=list(plm_models.keys())[0],
                visible=True,
                scale=2
            )

            transfer_model_folder = gr.Dropdown(
                label="Model Folder Path",
                choices=_folder_choices,
                value=None,
                allow_custom_value=False,
                visible=False,
                scale=1
            )
            transfer_model_dropdown = gr.Dropdown(
                label="Model Name",
                choices=[],
                value=None,
                allow_custom_value=False,
                visible=False,
                scale=1
            )

        # Model Info Display (visible when Continue Training)
        with gr.Accordion("📊 Model Info", open=False, visible=False) as transfer_settings_row:
            model_info_display = gr.JSON(
                label="Configuration",
                value={}
            )

        # Dataset Selection
        with gr.Row():
            with gr.Column(scale=4):
                with gr.Row():
                    is_custom_dataset = gr.Radio(
                        choices=["Custom", "Pre-defined"],
                        label="Dataset",
                        value="Pre-defined",
                        scale=1
                    )

                    dataset_config = gr.Dropdown(
                        choices=list(dataset_configs.keys()),
                        label="Dataset Configuration",
                        value=list(dataset_configs.keys())[0],
                        visible=True,
                        scale=2
                    )

                    dataset_custom = gr.Textbox(
                        label="Dataset Path",
                        placeholder="Local path or Huggingface dataset",
                        visible=False,
                        scale=2
                    )

            with gr.Column(scale=1, min_width=120, elem_classes="preview-button-container"):
                dataset_preview_button = gr.Button(
                    "Preview Dataset",
                    variant="primary",
                    size="lg",
                    elem_classes=["preview-button"]
                )

    # Dataset configuration details (always visible, collapsed by default for Pre-defined)
    with gr.Accordion("⚙️ Dataset Configuration Details", open=False) as custom_dataset_settings:
            with gr.Row():
                problem_type = gr.Dropdown(
                    choices=["single_label_classification", "multi_label_classification", "regression", "residue_single_label_classification", "residue_regression"],
                    label="Problem Type",
                    value="single_label_classification",
                    scale=23,
                    interactive=False   
                )
                num_labels = gr.Number(
                    value=2,
                    label="Number of Labels",
                    scale=11,
                    interactive=False
                )
                metrics = gr.Dropdown(
                    choices=["accuracy", "recall", "precision", "f1", "mcc", "auroc", "aupr", "f1_max", "f1_positive", "f1_negative", "spearman_corr", "mse"],
                    label="Metrics",
                    value=["accuracy", "mcc", "f1", "precision", "recall", "auroc"],
                    scale=101,
                    multiselect=True,
                    interactive=False
                )
            
            with gr.Row():
                sequence_column_name = gr.Textbox(
                    label="Amino Acid Sequence Column Name",
                    value="aa_seq",
                    placeholder="Name of the amino acid sequence column in dataset",
                    scale=10,
                    interactive=False
                )
                label_column_name = gr.Textbox(
                    label="Target Label Column Name",
                    value="label",
                    placeholder="Name of the target label column in dataset",
                    scale=10,
                    interactive=False
                )
                
                monitored_metrics = gr.Dropdown(
                    choices=["accuracy", "recall", "precision", "f1", "mcc", "auroc", "aupr", "f1_max", "f1_positive", "f1_negative", "spearman_corr", "mse"],
                    label="Monitored Metrics",
                    value="accuracy",
                    scale=10,
                    multiselect=False,
                    interactive=False
                )
                monitored_strategy = gr.Dropdown(
                    choices=["max", "min"],
                    label="Monitored Strategy",
                    value="max",
                    scale=10,
                    interactive=False
                )

    with gr.Row():
        structure_seq = gr.Dropdown(
            label="Structure Sequence",
            choices=["foldseek_seq", "ss8_seq"],
            value=["foldseek_seq", "ss8_seq"],
            multiselect=True,
            visible=False
        )

    with gr.Row(visible=False) as pdb_dir_row:
        pdb_dir = gr.Textbox(
            label="PDB Directory",
            placeholder="Path to PDB files (for ProSST, SaProt, ProtSSN, ses-adapter when dataset lacks structure)",
            value="",
            scale=3
        )

    # LoRA parameters (shown when using LoRA-based training methods)
    with gr.Row(visible=False) as lora_params_row:
        with gr.Column():
            lora_r = gr.Number(
                value=8,
                label="LoRA Rank",
                precision=0,
                minimum=1,
                maximum=128,
            )
        with gr.Column():
            lora_alpha = gr.Number(
                value=32,
                label="LoRA Alpha",
                precision=0,
                minimum=1,
                maximum=128
            )
        with gr.Column():
            lora_dropout = gr.Number(
                value=0.1,
                label="LoRA Dropout",
                minimum=0.0,
                maximum=1.0
            )
        with gr.Column():
            lora_target_modules = gr.Textbox(
                value="query,key,value",
                label="LoRA Target Modules",
                placeholder="Comma-separated list of target modules",
            )

    # put data statistics and table into accordion panel
    with gr.Row():
        with gr.Accordion("🔍 Dataset Preview", open=False) as preview_accordion:
            # data statistics area
            with gr.Row():
                dataset_stats_md = gr.HTML("", elem_classes=["dataset-stats"], padding=True)
            
            # table area
            with gr.Row():
                preview_table = gr.Dataframe(
                    headers=["Status", "Message", "Action"],
                    value=[["No dataset selected", "Please select a dataset to preview", "Click Preview Dataset"]],
                    wrap=True,
                    interactive=False,
                    row_count=3,
                    elem_classes=["preview-table"]
                )

    
    # Hyperparameter Settings
    gr.Markdown("## Hyperparameter Settings")
    with gr.Accordion("Click here to expand (If you are confused about the parameters, please check the documentation or take a screenshot and ask ChatGPT/Gemini for help)", open=False):
        # Batch Processing Configuration
        gr.Markdown("### Batch Processing Configuration")
        with gr.Group():
            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    batch_mode = gr.Radio(
                        choices=["Batch Size Mode", "Batch Token Mode"],
                        label="Batch Processing Mode",
                        value="Batch Size Mode"
                    )
                
                with gr.Column(scale=2):
                    batch_size = gr.Slider(
                        minimum=1, maximum=128, value=1,
                        step=1, label="Batch Size", visible=True
                    )
                    
                    batch_token = gr.Slider(
                        minimum=1000, maximum=50000, value=2000,
                        step=1000, label="Tokens per Batch", visible=False
                    )

        def update_train_tab_batch_inputs_UI(mode):
            """
            Update batch or token input visibility for Gradio UI
            Args:
                mode: batch mode
            Returns:
                batch_size: batch size input gr.Slider (visible or not)
                batch_token: batch token input gr.Slider (visible or not)
            """
            return {
                batch_size: gr.update(visible=mode == "Batch Size Mode"),
                batch_token: gr.update(visible=mode == "Batch Token Mode")
            }

        # Update visibility when mode changes
        batch_mode.change(
            fn=update_train_tab_batch_inputs_UI,
            inputs=[batch_mode],
            outputs=[batch_size, batch_token]
        )

        # Training Parameters
        gr.Markdown("### Training Parameters")
        with gr.Group():
            # First row: Basic training parameters
            with gr.Row(equal_height=True):
                with gr.Column(scale=1, min_width=150):
                    training_method = gr.Dropdown(
                        choices=["full", "freeze", "ses-adapter", "plm-lora", "plm-qlora", "plm-adalora", "plm-dora", "plm-ia3"],
                        label="Training Method",
                        value="freeze"
                    )
                with gr.Column(scale=1, min_width=150):
                    learning_rate = gr.Slider(
                        minimum=1e-8, maximum=1e-2, value=5e-4, step=1e-6,
                        label="Learning Rate"
                    )
                with gr.Column(scale=1, min_width=150):
                    num_epochs = gr.Slider(
                        minimum=1, maximum=200, value=20, step=1,
                        label="Number of Epochs"
                    )
                with gr.Column(scale=1, min_width=150):
                    patience = gr.Slider(
                        minimum=1, maximum=50, value=10, step=1,
                        label="Early Stopping Patience"
                    )
                with gr.Column(scale=1, min_width=150):
                    max_seq_len = gr.Slider(
                        minimum=-1, maximum=2048, value=None, step=32,
                        label="Max Sequence Length (-1 for unlimited)"
                    )
            
            def _plm_needs_structure(plm: str) -> bool:
                """True if PLM requires structure (ProSST, SaProt, ProtSSN)."""
                return bool(plm and (str(plm).startswith("ProSST") or str(plm).startswith("SaProt") or str(plm).startswith("ProtSSN")))

            def update_train_tab_training_method_UI(method, plm):
                """Update visibility for structure_seq, pdb_dir_row, lora_params_row."""
                show_structure = (method == "ses-adapter") or _plm_needs_structure(plm)
                return {
                    structure_seq: gr.update(visible=method == "ses-adapter"),
                    pdb_dir_row: gr.update(visible=show_structure),
                    lora_params_row: gr.update(visible=method in ["plm-lora", "plm-qlora", "plm-adalora", "plm-dora", "plm-ia3"])
                }

            # Add training_method and plm_model change events
            training_method.change(
                fn=update_train_tab_training_method_UI,
                inputs=[training_method, plm_model],
                outputs=[structure_seq, pdb_dir_row, lora_params_row]
            )
            plm_model.change(
                fn=update_train_tab_training_method_UI,
                inputs=[training_method, plm_model],
                outputs=[structure_seq, pdb_dir_row, lora_params_row]
            )

            # Second row: Advanced training parameters
            with gr.Row(equal_height=True):
                with gr.Column(scale=1, min_width=150):
                    pooling_method = gr.Dropdown(
                        choices=["mean", "attention1d", "light_attention"],
                        label="Pooling Method",
                        value="mean"
                    )
                
                with gr.Column(scale=1, min_width=150):
                    scheduler_type = gr.Dropdown(
                        choices=["linear", "cosine", "step", None],
                        label="Scheduler Type",
                        value=None
                    )
                with gr.Column(scale=1, min_width=150):
                    warmup_steps = gr.Slider(
                        minimum=0, maximum=1000, value=0, step=10,
                        label="Warmup Steps"
                    )
                with gr.Column(scale=1, min_width=150):
                    gradient_accumulation_steps = gr.Slider(
                        minimum=1, maximum=32, value=1, step=1,
                        label="Gradient Accumulation Steps"
                    )
                with gr.Column(scale=1, min_width=150):
                    max_grad_norm = gr.Slider(
                        minimum=0.1, maximum=10.0, value=0.1, step=0.1,
                        label="Max Gradient Norm"
                    )
                with gr.Column(scale=1, min_width=150):
                    num_workers = gr.Slider(
                        minimum=0, maximum=16, value=4, step=1,
                        label="Number of Workers"
                    )

    # Output and Training Control
    with gr.Row():
        # Left side: Output and Logging Settings
        with gr.Column(scale=1):
            gr.Markdown("## Output and Logging Settings")
            with gr.Row():
                output_dir = gr.Textbox(
                    label="Save Directory",
                    value="ckpt/demo",
                    placeholder="e.g. ckpt/demo or ckpt/your_folder",
                )
                output_model_name = gr.Textbox(
                    label="Output Model Name",
                    value="demo.pt",
                    placeholder="Name of the output model file"
                )

            wandb_logging = gr.Checkbox(
                label="Enable W&B Logging",
                value=False
            )

            wandb_project = gr.Textbox(
                label="W&B Project Name",
                value=None,
                visible=False
            )

            wandb_entity = gr.Textbox(
                label="W&B Entity",
                value=None,
                visible=False
            )

        # Right side: Training Control
        with gr.Column(scale=1):
            gr.Markdown("## Training Control")
            preview_button = gr.Button("Preview Command", elem_classes=["preview-command-btn"])
            train_button = gr.Button("Start Training", variant="primary", elem_classes=["train-btn"])
            abort_button = gr.Button("Abort Training", variant="stop", elem_classes=["abort-btn"])
            
    
    with gr.Row():
        command_preview = gr.Code(
            label="Command Preview",
            language="shell",
            interactive=False,
            visible=False
        )
    
    # Model Statistics and Training Progress side by side
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## Model Statistics")
            model_stats = gr.Dataframe(
                headers=["Model Type", "Total Parameters", "Trainable Parameters", "Percentage"],
                value=[
                    ["Training Model", "-", "-", "-"],
                    ["Pre-trained Model", "-", "-", "-"],
                    ["Combined Model", "-", "-", "-"]
                ],
                interactive=False,
                elem_classes=["center-table-content"]
            )

        with gr.Column(scale=1):
            gr.Markdown("## Training Progress")
            with gr.Row():
                progress_status = gr.HTML(
                    value="""
                    <div style="background-color: #f8f9fa; border-radius: 10px; padding: 20px; margin-bottom: 15px; box-shadow: 0 2px 5px rgba(0,0,0,0.05);">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 12px;">
                            <div>
                                <span style="font-weight: 600; font-size: 16px;">Training Status: </span>
                                <span style="color: #1976d2; font-weight: 500; font-size: 16px;">Click Start to train your model</span>
                            </div>
                        </div>
                    </div>
                    """,
                    label="Status",
                    padding=True
                )

            with gr.Row():
                best_model_info = gr.Textbox(
                    value="Best Model: None",
                    label="Best Model and Performance",
                    interactive=False
                )

    def format_to_millions(value):
        """Converts a number or a string with 'K'/'M' suffix to millions."""
        if not isinstance(value, str) or value == '-':
            return str(value)
        value_str = value.strip().upper()
        
        try:
            if value_str.endswith('K'):
                number = float(value_str[:-1]) * 1_000
            elif value_str.endswith('M'):
                number = float(value_str[:-1]) * 1_000_000
            else:
                number = float(value_str)
                
            millions = number / 1_000_000
            return f"{millions:.2f}M"
            
        except (ValueError, TypeError):
            return value
            
    def update_model_stats(stats: Dict[str, str]) -> List[List[str]]:
        """Update model statistics in table format."""
        if not stats:
            return [
                ["Training Model", "-", "-", "-"],
                ["Pre-trained Model", "-", "-", "-"],
                ["Combined Model", "-", "-", "-"]
            ]
        
        adapter_total = stats.get('adapter_total', '-')
        adapter_trainable = stats.get('adapter_trainable', '-')
        pretrain_total = stats.get('pretrain_total', '-')
        pretrain_trainable = stats.get('pretrain_trainable', '-')
        combined_total = stats.get('combined_total', '-')
        combined_trainable = stats.get('combined_trainable', '-')
        trainable_percentage = stats.get('trainable_percentage', '-')
        
        adapter_total_m = format_to_millions(adapter_total)
        adapter_trainable_m = format_to_millions(adapter_trainable)
        pretrain_total_m = format_to_millions(pretrain_total)
        pretrain_trainable_m = format_to_millions(pretrain_trainable)
        combined_total_m = format_to_millions(combined_total)
        combined_trainable_m = format_to_millions(combined_trainable)
        
        return [
            ["Training Model", adapter_total_m, adapter_trainable_m, "-"],
            ["Pre-trained Model", pretrain_total_m, pretrain_trainable_m, "-"],
            ["Combined Model", combined_total_m, combined_trainable_m, str(trainable_percentage)+"%"]
        ]

    # Add test results HTML display
    with gr.Row():
        test_results_html = gr.HTML(
            value="",
            label="Test Results",
            visible=True,
            padding=True
        )
        
    with gr.Row():
        with gr.Column(scale=4):
            pass
        with gr.Column(scale=1):  # limit the maximum width of the column
            download_csv_btn = gr.DownloadButton(
                "Download CSV", 
                visible=False,
                size="lg",
                elem_classes=["download-btn"]
            )
        # add an empty column to occupy the remaining space
        with gr.Column(scale=4):
            pass

    # Training plot in a separate row for full width
    with gr.Row():
        with gr.Column():
            loss_plot = gr.Plot(
                label="Training and Validation Loss",
                elem_id="loss_plot"
            )
        with gr.Column():
            metrics_plot = gr.Plot(
                label="Validation Metrics",
                elem_id="metrics_plot"
            )

    def update_progress(progress_info):
        # If progress_info is empty or None, use completely fresh empty state
        if not progress_info or not any(progress_info.values()):
            fresh_status_html = load_html_template("status_empty.html")
            return (
                fresh_status_html,
                "Best Model: None",
                gr.update(value="", visible=False),
                None,
                None,
                gr.update(visible=False)
            )
        
        # Reset values if stage is "Waiting" or "Error"
        if progress_info.get('stage', '') == 'Waiting' or progress_info.get('stage', '') == 'Error':
            # If this is an error stage, show error styling
            if progress_info.get('stage', '') == 'Error':
                error_status_html = load_html_template("status_error.html", error_title="Training failed", error_message="Failed", error_details="")
                return (
                    error_status_html,
                    "Training failed",
                    gr.update(value="", visible=False),
                    None,
                    None,
                    gr.update(visible=False)
                )
            else:
                return (
                    load_html_template("status_waiting.html"),
                    "Best Model: None",
                    gr.update(value="", visible=False),
                    None,
                    None,
                    gr.update(visible=False)
                )
        
        current = progress_info.get('current', 0)
        total = progress_info.get('total', 100)
        epoch = progress_info.get('epoch', 0)
        stage = progress_info.get('stage', 'Waiting')
        progress_detail = progress_info.get('progress_detail', '')
        best_epoch = progress_info.get('best_epoch', 0)
        best_metric_name = progress_info.get('best_metric_name', 'accuracy')
        best_metric_value = progress_info.get('best_metric_value', 0.0)
        elapsed_time = progress_info.get('elapsed_time', '')
        remaining_time = progress_info.get('remaining_time', '')
        it_per_sec = progress_info.get('it_per_sec', 0.0)
        grad_step = progress_info.get('grad_step', 0)
        loss = progress_info.get('loss', 0.0)
        total_epochs = progress_info.get('total_epochs', 0)
        test_results_html = progress_info.get('test_results_html', '')
        test_metrics = progress_info.get('test_metrics', {})
        is_completed = progress_info.get('is_completed', False)
        
        # Test results HTML visibility is always True, but show message when content is empty
        if not test_results_html and stage == 'Testing':
            test_results_html = load_html_template("status_testing_empty.html")
        elif not test_results_html:
            test_results_html = load_html_template("status_test_results_empty.html")
        
        test_html_update = gr.update(value=test_results_html, visible=True)
        
        # process CSV download button
        if test_metrics and len(test_metrics) > 0:
            # create a temporary file to save CSV content
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv', prefix='metrics_results_') as temp_file:
                # write CSV header
                temp_file.write("Metric,Value\n")
                
                # sort metrics by priority
                priority_metrics = ['loss', 'accuracy', 'f1', 'precision', 'recall', 'auroc', 'mcc']
                
                def get_priority(item):
                    name = item[0]
                    if name in priority_metrics:
                        return priority_metrics.index(name)
                    return len(priority_metrics)
                
                # sort and add to CSV
                sorted_metrics = sorted(test_metrics.items(), key=get_priority)
                for metric_name, metric_value in sorted_metrics:
                    # Convert metric name: uppercase for abbreviations, capitalize for others
                    display_name = metric_name
                    if metric_name.lower() in ['f1', 'mcc', 'auroc']:
                        display_name = metric_name.upper()
                    else:
                        display_name = metric_name.capitalize()
                    temp_file.write(f"{display_name},{metric_value:.6f}\n")
                
                file_path = temp_file.name
            
            download_btn_update = gr.update(value=file_path, visible=True)
        else:
            download_btn_update = gr.update(visible=False)
        
        # calculate progress percentage
        progress_percentage = (current / total) * 100 if total > 0 else 0
        
        # create modern progress bar HTML
        if is_completed:
            # training complete status
            status_html = load_html_template("status_success.html")
        else:
            # training or validation stage
            epoch_total = total_epochs if total_epochs > 0 else 100
            
            status_html = f"""
            <div style="background-color: #f8f9fa; border-radius: 10px; padding: 20px; margin-bottom: 15px; box-shadow: 0 2px 5px rgba(0,0,0,0.05);">
                <div style="display: flex; justify-content: space-between; margin-bottom: 12px;">
                    <div>
                        <span style="font-weight: 600; font-size: 16px;">Training Status: </span>
                        <span style="color: #1976d2; font-weight: 500; font-size: 16px;">{stage} (Epoch {epoch}/{epoch_total})</span>
                    </div>
                    <div>
                        <span style="font-weight: 600; color: #333;">{progress_percentage:.1f}%</span>
                    </div>
                </div>
                
                <div style="margin-bottom: 15px; background-color: #e9ecef; height: 10px; border-radius: 5px; overflow: hidden;">
                    <div style="background-color: #4285f4; width: {progress_percentage}%; height: 100%; border-radius: 5px; transition: width 0.3s ease;"></div>
                </div>
                
                <div style="display: flex; flex-wrap: wrap; gap: 10px; font-size: 14px; color: #555;">
                    <div style="background-color: #e8f5e9; padding: 5px 10px; border-radius: 4px;"><span style="font-weight: 500;">Progress:</span> {current}/{total}</div>
                    {f'<div style="background-color: #fff8e1; padding: 5px 10px; border-radius: 4px;"><span style="font-weight: 500;">Time:</span> {elapsed_time}<{remaining_time}, {it_per_sec:.2f}it/s></div>' if elapsed_time and remaining_time else ''}
                    {f'<div style="background-color: #e3f2fd; padding: 5px 10px; border-radius: 4px;"><span style="font-weight: 500;">Loss:</span> {loss:.4f}</div>' if stage == 'Training' and loss > 0 else ''}
                    {f'<div style="background-color: #f3e5f5; padding: 5px 10px; border-radius: 4px;"><span style="font-weight: 500;">Grad steps:</span> {grad_step}</div>' if stage == 'Training' and grad_step > 0 else ''}
                </div>
            </div>
            """
        
        # build best model information
        if best_epoch >= 0 and best_metric_value > 0:
            best_info = f"Best model: Epoch {best_epoch} ({best_metric_name}: {best_metric_value:.4f})"
        else:
            best_info = "No best model found yet"
        
        # get and update charts
        loss_fig = monitor.get_loss_plot()
        metrics_fig = monitor.get_metrics_plot()
        
        # return updated components
        return status_html, best_info, test_html_update, loss_fig, metrics_fig, download_btn_update

    # DEPRECATED: This old handle_train function is shadowed by the new one with explicit parameters (line ~1439)
    # Kept for reference only - not executed
    def handle_train_old(*args) -> Generator:
        nonlocal is_training, current_process, stop_thread, process_aborted, monitor

        # If already training, return
        if is_training:
            yield None, None, None, None, None, None, None
            return
        
        # Force explicit state reset first thing
        monitor._reset_tracking()
        monitor._reset_stats()
        
        # Explicitly ensure stats are reset
        if hasattr(monitor, "stats"):
            monitor.stats = {}
        
        # Force override any cached state in monitor
        monitor.current_progress = {
            "current": 0,
            "total": 0,
            "epoch": 0,
            "stage": "Waiting",
            "progress_detail": "",
            "best_epoch": -1,
            "best_metric_name": "",
            "best_metric_value": 0.0,
            "elapsed_time": "",
            "remaining_time": "",
            "it_per_sec": 0.0,
            "grad_step": 0,
            "loss": 0.0,
            "test_results_html": "",
            "test_metrics": {},
            "is_completed": False,
            "lines": []
        }
        
        # Reset all monitoring data structures
        monitor.train_losses = []
        monitor.val_losses = []
        monitor.metrics = {}
        monitor.epochs = []
        if hasattr(monitor, "stats"):
            monitor.stats = {}
        
        # Reset flags for new training session
        process_aborted = False
        stop_thread = False
        
        # Initialize table state
        initial_stats = [
            ["Training Model", "-", "-", "-"],
            ["Pre-trained Model", "-", "-", "-"],
            ["Combined Model", "-", "-", "-"]
        ]
        
        # Initial UI state with "Initializing" message
        initial_status_html = load_html_template("status_initializing.html")
        
        # First yield to update UI with "initializing" state
        yield initial_stats, initial_status_html, "Best Model: None", gr.update(value="", visible=False), None, None, gr.update(visible=False), gr.update(visible=False), gr.update(value=""), gr.update(value="Custom"), gr.update(value=""), gr.update(value={}), gr.update(value="")
        
        try:
            # Parse training arguments
            training_args = TrainingArgs(args, plm_models, dataset_configs)
            
            if training_args.training_method != "ses-adapter":
                training_args.structure_seq = None
            
            args_dict = training_args.to_dict()
            
            # Save total epochs to monitor for use in progress_info
            total_epochs = args_dict.get('num_epochs', 100)
            monitor.current_progress['total_epochs'] = total_epochs
            
            # Update status to "Preparing dataset"
            preparing_status_html = load_html_template("status_preparing.html")
            yield initial_stats, preparing_status_html, "Best Model: None", gr.update(value="", visible=False), None, None, gr.update(visible=False), gr.update(visible=False), gr.update(value=""), gr.update(value="Custom"), gr.update(value=""), gr.update(value={}), gr.update(value="")

            # Save arguments to file
            save_arguments(args_dict, args_dict.get('output_dir', 'ckpt'))

            # Start training
            is_training = True
            process_aborted = False  # Reset abort flag
            monitor.start_training(args_dict)
            current_process = monitor.process  # Store the process reference

            starting_status_html = load_html_template("status_starting.html")
            yield initial_stats, starting_status_html, "Best Model: None", gr.update(value="", visible=False), None, None, gr.update(visible=False), gr.update(visible=False), gr.update(value=""), gr.update(value="Custom"), gr.update(value=""), gr.update(value={}), gr.update(value="")

            # Add delay to ensure enough time for parsing initial statistics
            for i in range(3):
                time.sleep(1)
                # Check if statistics are already available
                stats = monitor.get_stats()
                if stats and len(stats) > 0:
                    break
            
            update_count = 0
            while True:
                # Check if the process still exists and hasn't been aborted
                if process_aborted or (current_process and current_process.poll() is not None):
                    break
                    
                try:
                    update_count += 1
                    time.sleep(0.5)
                    
                    # Check process status
                    monitor.check_process_status()
                    
                    # Get latest progress info
                    progress_info = monitor.get_progress()
                    
                    # If process has ended, check if it's normal end or error
                    if not monitor.is_training:
                        # Check both monitor.process and current_process since they might be different objects
                        if (monitor.process and monitor.process.returncode != 0) or (current_process and current_process.poll() is not None and current_process.returncode != 0):
                            # Get the return code from whichever process object is available
                            return_code = monitor.process.returncode if monitor.process else current_process.returncode
                            # Get complete output log
                            error_output = "\n".join(progress_info.get("lines", []))
                            if not error_output:
                                error_output = "No output captured from the training process"
                            
                            # Ensure we set the is_completed flag to False for errors
                            progress_info['is_completed'] = False
                            monitor.current_progress['is_completed'] = False
                            
                            # Also set the stage to Error
                            progress_info['stage'] = 'Error'
                            monitor.current_progress['stage'] = 'Error'
                            
                            error_status_html = load_html_template("status_failed_code.html", return_code=return_code, error_output=error_output)
                            yield (
                                initial_stats,
                                error_status_html,
                                "Training failed",
                                gr.update(value="", visible=False),
                                None,
                                None,
                                gr.update(visible=False),
                                gr.update(visible=False),
                                gr.update(value=""),
                                gr.update(value="Custom"),
                                gr.update(value=""),
                                gr.update(value={}),
                                gr.update(value="")
                            )
                            return
                        else:
                            # Only set is_completed to True if there was a successful exit code
                            progress_info['is_completed'] = True
                            monitor.current_progress['is_completed'] = True
                    
                    # Update UI
                    stats = monitor.get_stats()
                    if stats:
                        model_stats = update_model_stats(stats)
                    else:
                        model_stats = initial_stats
                    
                    status_html, best_info, test_html_update, loss_fig, metrics_fig, download_btn_update = update_progress(progress_info)

                    # During training, keep registration panel hidden
                    yield model_stats, status_html, best_info, test_html_update, loss_fig, metrics_fig, download_btn_update, gr.update(visible=False), gr.update(value=""), gr.update(value="Custom"), gr.update(value=""), gr.update(value={}), gr.update(value="")

                except Exception as e:
                    # Get complete output log
                    error_output = "\n".join(progress_info.get("lines", []))
                    if not error_output:
                        error_output = "No output captured from the training process"
                    
                    error_status_html = load_html_template("status_error.html", error_title="Error during training", error_message=str(e))
                    print(f"Error updating UI: {str(e)}")
                    traceback.print_exc()
                    yield initial_stats, error_status_html, "Training error", gr.update(value="", visible=False), None, None, gr.update(visible=False), gr.update(visible=False), gr.update(value=""), gr.update(value="Custom"), gr.update(value=""), gr.update(value={}), gr.update(value="")
                    return
            
            # Check if aborted
            if process_aborted:
                is_training = False
                current_process = None
                aborted_status_html = load_html_template("status_aborted.html")
                yield initial_stats, aborted_status_html, "Training aborted", gr.update(value="", visible=False), None, None, gr.update(visible=False), gr.update(visible=False), gr.update(value=""), gr.update(value="Custom"), gr.update(value=""), gr.update(value={}), gr.update(value="")
                return
            
            # Final update after training ends (only for normal completion)
            if monitor.process and monitor.process.returncode == 0:
                try:
                    progress_info = monitor.get_progress()
                    progress_info['is_completed'] = True
                    monitor.current_progress['is_completed'] = True
                    
                    stats = monitor.get_stats()
                    if stats:
                        model_stats = update_model_stats(stats)
                    else:
                        model_stats = initial_stats
                    
                    status_html, best_info, test_html_update, loss_fig, metrics_fig, download_btn_update = update_progress(progress_info)
                    
                    yield model_stats, status_html, best_info, test_html_update, loss_fig, metrics_fig, download_btn_update
                except Exception as e:
                    error_output = "\n".join(progress_info.get("lines", []))
                    if not error_output:
                        error_output = "No output captured from the training process"
                    
                    error_status_html = load_html_template("status_error.html", error_title="Error in final update", error_message=str(e))
                    yield initial_stats, error_status_html, "Error in final update", gr.update(value="", visible=False), None, None, gr.update(visible=False)
            
        except Exception as e:
            # Initialization error, may not have output log
            error_status_html = load_html_template("status_error.html", error_title="Training initialization failed", error_message=str(e))
            yield initial_stats, error_status_html, "Training failed", gr.update(value="", visible=False), None, None, gr.update(visible=False)
        finally:
            is_training = False
            current_process = None
    
    def handle_train_tab_abort():
        """Handle abortion of the training process"""
        nonlocal is_training, current_process, stop_thread, process_aborted
        
        if not is_training or current_process is None:
            return (gr.HTML(load_html_template("status_empty.html")),
            [["Training Model", "-", "-", "-"], 
                ["Pre-trained Model", "-", "-", "-"], 
                ["Combined Model", "-", "-", "-"]],
            "Best Model: None",
            gr.update(value="", visible=False),
            None,
            None,
            gr.update(visible=False))
        
        try:
            # Set the abort flag before terminating the process
            process_aborted = True
            stop_thread = True
            
            # Use process.terminate() instead of os.killpg for safer termination
            # This avoids accidentally killing the parent WebUI process
            current_process.terminate()
            
            # Wait for process to terminate (with timeout)
            try:
                current_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # Only if terminate didn't work, use a stronger method
                # But do NOT use killpg which might kill the parent WebUI
                current_process.kill()
            
            # Create a completely fresh state - not just resetting
            monitor.is_training = False
            
            # Explicitly create a new dictionary instead of modifying the existing one
            monitor.current_progress = {
                "current": 0,
                "total": 0,
                "epoch": 0,
                "stage": "Waiting",
                "progress_detail": "",
                "best_epoch": -1,
                "best_metric_name": "",
                "best_metric_value": 0.0,
                "elapsed_time": "",
                "remaining_time": "",
                "it_per_sec": 0.0,
                "grad_step": 0,
                "loss": 0.0,
                "test_results_html": "",
                "test_metrics": {},
                "is_completed": False,
                "lines": []
            }
            
            # Explicitly clear stats by creating a new dictionary
            monitor.stats = {}
            
            if hasattr(monitor, "process") and monitor.process:
                monitor.process = None
                
            # Reset state variables
            is_training = False
            current_process = None
            
            # Explicitly reset tracking to clear all state
            monitor._reset_tracking()
            monitor._reset_stats()
            
            # Reset all plots and statistics with new empty lists
            monitor.train_losses = []
            monitor.val_losses = []
            monitor.metrics = {}
            monitor.epochs = []
            
            # Create entirely fresh UI components
            empty_model_stats = [["Training Model", "-", "-", "-"], 
                                ["Pre-trained Model", "-", "-", "-"], 
                                ["Combined Model", "-", "-", "-"]]
            
            success_html = load_html_template("status_success.html")
            
            # Return updates for all relevant components
            return (gr.HTML(success_html),
                    empty_model_stats,
                    "Best Model: None",
                    gr.update(value="", visible=False),
                    None,
                    None,
                    gr.update(visible=False))
        except Exception as e:
            # Still need to reset states even if there's an error
            is_training = False
            current_process = None
            process_aborted = False
            
            # Reset monitor state regardless of error
            monitor.is_training = False
            monitor.stats = {}
            if hasattr(monitor, "process") and monitor.process:
                monitor.process = None
            monitor._reset_tracking()
            monitor._reset_stats()
            
            # Fresh empty components
            empty_model_stats = [["Training Model", "-", "-", "-"], 
                                ["Pre-trained Model", "-", "-", "-"], 
                                ["Combined Model", "-", "-", "-"]]
            
            error_html = load_html_template("status_error.html", error_title="Failed to terminate training", error_message=str(e))
            
            # Return updates for all relevant components including empty model stats
            return (gr.HTML(error_html),
                    empty_model_stats,
                    "Best Model: None",
                    gr.update(value="", visible=False),
                    None,
                    None,
                    gr.update(visible=False))

    def update_wandb_visibility_UI(checkbox):
        """Update wandb project and entity visibility for Gradio UI
        Args:
            checkbox: wandb checkbox
        Returns:
            wandb_project: wandb project input gr.Textbox (visible or not)
            wandb_entity: wandb entity input gr.Textbox (visible or not)
        """
        return {
            wandb_project: gr.update(visible=checkbox),
            wandb_entity: gr.update(visible=checkbox)
        }

    # bind preview and train buttons
    def handle_train_tab_command_preview(
        plm_model, is_custom_dataset, dataset_config, dataset_custom, problem_type, num_labels,
        metrics, training_method, pooling_method, batch_mode, batch_size, batch_token,
        learning_rate, num_epochs, max_seq_len, gradient_accumulation_steps, warmup_steps, scheduler_type,
        output_model_name, output_dir, wandb_logging, wandb_project, wandb_entity,
        patience, num_workers, max_grad_norm, structure_seq, pdb_dir,
        lora_r, lora_alpha, lora_dropout, lora_target_modules, monitored_metrics, monitored_strategy,
        sequence_column_name, label_column_name,
        training_mode, transfer_model_path
    ):
        """Handle the preview command button click event
        Args:
            plm_model: plm model name
            is_custom_dataset: whether to Custom (Custom or Pre-defined)
            dataset_config: dataset config path
            dataset_custom: custom dataset path
            problem_type: problem type
            num_labels: number of labels
            metrics: metrics (accuracy, recall, precision, f1, mcc, auroc, f1_max, spearman_corr, mse)
            training_method: training method (ses-adapter, plm-lora, plm-qlora, plm-adalora, plm-dora, plm-ia3)
            pooling_method: pooling method (mean, attention1d, light_attention)
            batch_mode: batch mode (Batch Size Mode or Batch Token Mode)
            batch_size: batch size
            batch_token: batch token (tokens per batch)
            learning_rate: learning rate
            num_epochs: number of epochs
            max_seq_len: max sequence length
            gradient_accumulation_steps: gradient accumulation steps
            warmup_steps: warmup steps
            scheduler_type: scheduler type (linear, cosine, cosine_with_restarts, polynomial, constant)
            output_model_name: output model name
            output_dir: output directory
            wandb_logging: whether to use wandb
            wandb_project: wandb project name
            wandb_entity: wandb entity name
            patience: patience for early stopping
            num_workers: number of workers
            max_grad_norm: max gradient norm
            structure_seq: structure sequence (foldseek_seq, ss8_seq)
            lora_r: lora rank
            lora_alpha: lora alpha
            lora_dropout: lora dropout
            lora_target_modules: lora target modules
            monitored_metrics: monitored metric
            monitored_strategy: monitored strategy (max, min)
            sequence_column_name: name of the sequence column in dataset
            label_column_name: name of the label column in dataset
        Returns:
            command_preview: command preview
        """
        if command_preview.visible:
            return gr.update(visible=False)
        
        # Create args dictionary directly from named parameters
        # Create args dictionary directly from named parameters
        args_dict = {
            "plm_model": plm_model,
            "dataset_selection": is_custom_dataset,
            "dataset_config": dataset_config,
            "dataset_custom": dataset_custom,
            "problem_type": problem_type,
            "num_labels": num_labels,
            "metrics": metrics,
            "training_method": training_method,
            "pooling_method": pooling_method,
            "batch_mode": batch_mode,
            "batch_size": batch_size,
            "batch_token": batch_token,
            "learning_rate": learning_rate,
            "num_epochs": num_epochs,
            "max_seq_len": max_seq_len,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "warmup_steps": warmup_steps,
            "scheduler": scheduler_type,
            "output_model_name": output_model_name,
            "output_dir": output_dir,
            "wandb_enabled": wandb_logging,
            "wandb_project": wandb_project,
            "wandb_entity": wandb_entity,
            "patience": patience,
            "num_workers": num_workers,
            "max_grad_norm": max_grad_norm,
            "structure_seq": structure_seq,
            "pdb_dir": pdb_dir,
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "lora_target_modules": lora_target_modules,
            "monitored_metrics": monitored_metrics,
            "monitored_strategy": monitored_strategy,
            "sequence_column_name": sequence_column_name,
            "label_column_name": label_column_name,
        }
        
        training_args = TrainingArgs(args_dict, plm_models, dataset_configs)
        preview_text = preview_command(training_args.to_dict())
        return gr.update(value=preview_text, visible=True)

    def reset_train_tab_UI():
        """Reset the UI state before training starts
        Returns:
            empty_model_stats: empty model stats
            empty_progress_status: empty progress status
            best_model: best model
            command_preview: command preview
        """
        # Reset monitor state
        monitor._reset_tracking()
        monitor._reset_stats()
        
        # Explicitly ensure stats are reset
        if hasattr(monitor, "stats"):
            monitor.stats = {}
        
        # Create a completely fresh progress state
        monitor.current_progress = {
            "current": 0,
            "total": 0,
            "epoch": 0,
            "stage": "Waiting",
            "progress_detail": "",
            "best_epoch": -1,
            "best_metric_name": "",
            "best_metric_value": 0.0,
            "elapsed_time": "",
            "remaining_time": "",
            "it_per_sec": 0.0,
            "grad_step": 0,
            "loss": 0.0,
            "test_results_html": "",
            "test_metrics": {},
            "is_completed": False,
            "lines": []
        }
        
        # Reset all statistical data
        monitor.train_losses = []
        monitor.val_losses = []
        monitor.metrics = {}
        monitor.epochs = []
        
        # Force UI to reset by creating completely fresh components
        empty_model_stats = [["Training Model", "-", "-", "-"], 
                            ["Pre-trained Model", "-", "-", "-"], 
                            ["Combined Model", "-", "-", "-"]]
        
        empty_progress_status = load_html_template("status_empty.html")
        
        # Return exactly 13 values matching the 13 output components
        return (
            empty_model_stats,
            empty_progress_status,
            "Best Model: None",
            gr.update(value="", visible=False),
            None,  # loss_plot must be None, not a string
            None,  # metrics_plot must be None, not a string
            gr.update(visible=False)
        )

    def handle_train(
        plm_model, is_custom_dataset, dataset_config, dataset_custom, problem_type, num_labels,
        metrics, training_method, pooling_method, batch_mode, batch_size, batch_token,
        learning_rate, num_epochs, max_seq_len, gradient_accumulation_steps, warmup_steps, scheduler_type,
        output_model_name, output_dir, wandb_logging, wandb_project, wandb_entity,
        patience, num_workers, max_grad_norm, structure_seq, pdb_dir,
        lora_r, lora_alpha, lora_dropout, lora_target_modules,
        monitored_metrics, monitored_strategy, sequence_column_name, label_column_name,
        training_mode, transfer_model_path
    ) -> Generator:
        """Handle the train command button click event
        Args:
            plm_model: plm model name
            is_custom_dataset: whether to Custom (Custom or Pre-defined)
            dataset_config: dataset config path
            dataset_custom: custom dataset path
            problem_type: problem type
            num_labels: number of labels
            metrics: metrics (accuracy, recall, precision, f1, mcc, auroc, f1_max, spearman_corr, mse)
            training_method: training method (ses-adapter, plm-lora, plm-qlora, plm-adalora, plm-dora, plm-ia3)
            pooling_method: pooling method (mean, attention1d, light_attention)
            batch_mode: batch mode (Batch Size Mode or Batch Token Mode)
            batch_size: batch size
            batch_token: batch token (tokens per batch)
            learning_rate: learning rate
            num_epochs: number of epochs
            max_seq_len: max sequence length
            gradient_accumulation_steps: gradient accumulation steps
            warmup_steps: warmup steps
            scheduler_type: scheduler type (linear, cosine, cosine_with_restarts, polynomial, constant)
            output_model_name: output model name
            output_dir: output directory
            wandb_logging: whether to use wandb
            wandb_project: wandb project name
            wandb_entity: wandb entity name
            patience: patience for early stopping
            num_workers: number of workers
            max_grad_norm: max gradient norm
            structure_seq: structure sequence (foldseek_seq, ss8_seq)
            lora_r: lora rank
            lora_alpha: lora alpha
            lora_dropout: lora dropout
            lora_target_modules: lora target modules
            monitored_metrics: monitored metric
            monitored_strategy: monitored strategy (max, min)
            sequence_column_name: name of the sequence column in dataset
            label_column_name: name of the label column in dataset
        Returns:
            model_stats: model stats
            status_html: status html
            best_info: best info
        """
        nonlocal is_training, current_process, stop_thread, process_aborted, monitor
        
        # If already training, return
        if is_training:
            yield None, None, None, None, None, None, None
            return
        
        # Force explicit state reset first thing
        monitor._reset_tracking()
        monitor._reset_stats()
        
        # Explicitly ensure stats are reset
        if hasattr(monitor, "stats"):
            monitor.stats = {}
        
        # Force override any cached state in monitor
        monitor.current_progress = {
            "current": 0,
            "total": 0,
            "epoch": 0,
            "stage": "Waiting",
            "progress_detail": "",
            "best_epoch": -1,
            "best_metric_name": "",
            "best_metric_value": 0.0,
            "elapsed_time": "",
            "remaining_time": "",
            "it_per_sec": 0.0,
            "grad_step": 0,
            "loss": 0.0,
            "test_results_html": "",
            "test_metrics": {},
            "is_completed": False,
            "lines": []
        }
        
        # Reset all monitoring data structures
        monitor.train_losses = []
        monitor.val_losses = []
        monitor.metrics = {}
        monitor.epochs = []
        if hasattr(monitor, "stats"):
            monitor.stats = {}
        
        # Reset flags for new training session
        process_aborted = False
        stop_thread = False
        
        # Initialize table state
        initial_stats = [
            ["Training Model", "-", "-", "-"],
            ["Pre-trained Model", "-", "-", "-"],
            ["Combined Model", "-", "-", "-"]
        ]
        
        # Initial UI state with "Initializing" message
        initial_status_html = load_html_template("status_initializing.html")
        
        # First yield to update UI with "initializing" state
        yield initial_stats, initial_status_html, "Best Model: None", gr.update(value="", visible=False), None, None, gr.update(visible=False), gr.update(visible=False), gr.update(value=""), gr.update(value="Custom"), gr.update(value=""), gr.update(value={}), gr.update(value="")
        
        try:
            # Create args dictionary directly from named parameters
            args_dict = {
                "plm_model": plm_model,
                "dataset_selection": is_custom_dataset,
                "dataset_config": dataset_config,
                "dataset_custom": dataset_custom,
                "problem_type": problem_type,
                "num_labels": num_labels,
                "metrics": metrics,
                "training_method": training_method,
                "pooling_method": pooling_method,
                "batch_mode": batch_mode,
                "batch_size": batch_size,
                "batch_token": batch_token,
                "learning_rate": learning_rate,
                "num_epochs": num_epochs,
                "max_seq_len": max_seq_len,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "warmup_steps": warmup_steps,
                "scheduler": scheduler_type,
                "output_model_name": output_model_name,
                "output_dir": output_dir,
                "wandb_enabled": wandb_logging,
                "wandb_project": wandb_project,
                "wandb_entity": wandb_entity,
                "patience": patience,
                "num_workers": num_workers,
                "max_grad_norm": max_grad_norm,
                "structure_seq": structure_seq,
                "pdb_dir": pdb_dir,
                "lora_r": lora_r,
                "lora_alpha": lora_alpha,
                "lora_dropout": lora_dropout,
                "lora_target_modules": lora_target_modules,
                "monitored_metrics": monitored_metrics,
                "monitored_strategy": monitored_strategy,
                "sequence_column_name": sequence_column_name,
                "label_column_name": label_column_name,
            }

            # Parse training arguments
            training_args = TrainingArgs(args_dict, plm_models, dataset_configs)

            if training_args.training_method != "ses-adapter":
                training_args.structure_seq = None

            args_dict = training_args.to_dict()

            # Add transfer learning parameters AFTER to_dict() to ensure they are not lost
            if training_mode == "Continue Training" and transfer_model_path:
                # Load model config to check problem_type compatibility
                model_config = load_model_config(transfer_model_path)
                if model_config:
                    model_problem_type = model_config.get("problem_type", "")
                    dataset_problem_type = args_dict.get("problem_type", "")

                    if model_problem_type and dataset_problem_type and model_problem_type != dataset_problem_type:
                        error_msg = f"❌ Problem Type Mismatch!\n\nModel was trained with: {model_problem_type}\nNew dataset has: {dataset_problem_type}\n\nContinue training requires the same problem type."
                        error_status_html = load_html_template("status_error.html",
                                                               error_title="Problem Type Mismatch",
                                                               error_message=error_msg,
                                                               error_details="Please select a dataset with the same problem type as the model.")
                        yield initial_stats, error_status_html, "Training failed", gr.update(value="", visible=False), None, None, gr.update(visible=False), gr.update(visible=False), gr.update(value=""), gr.update(value="Custom"), gr.update(value=""), gr.update(value={}), gr.update(value="")
                        return

                args_dict["initial_model_path"] = transfer_model_path

            # Save total epochs to monitor for use in progress_info
            total_epochs = args_dict.get('num_epochs', 100)
            monitor.current_progress['total_epochs'] = total_epochs
            
            # Update status to "Preparing dataset"
            preparing_status_html = load_html_template("status_preparing.html")
            yield initial_stats, preparing_status_html, "Best Model: None", gr.update(value="", visible=False), None, None, gr.update(visible=False), gr.update(visible=False), gr.update(value=""), gr.update(value="Custom"), gr.update(value=""), gr.update(value={}), gr.update(value="")

            # Save arguments to file
            save_arguments(args_dict, args_dict.get('output_dir', 'ckpt'))

            # Start training
            is_training = True
            process_aborted = False  # Reset abort flag
            monitor.start_training(args_dict)
            current_process = monitor.process  # Store the process reference

            starting_status_html = load_html_template("status_starting.html")
            yield initial_stats, starting_status_html, "Best Model: None", gr.update(value="", visible=False), None, None, gr.update(visible=False), gr.update(visible=False), gr.update(value=""), gr.update(value="Custom"), gr.update(value=""), gr.update(value={}), gr.update(value="")

            # Add delay to ensure enough time for parsing initial statistics
            for i in range(3):
                time.sleep(1)
                # Check if statistics are already available
                stats = monitor.get_stats()
                if stats and len(stats) > 0:
                    break
            
            update_count = 0
            while True:
                # Check if the process still exists and hasn't been aborted
                if process_aborted or (current_process and current_process.poll() is not None):
                    break
                    
                try:
                    update_count += 1
                    time.sleep(0.5)
                    
                    # Check process status
                    monitor.check_process_status()
                    
                    # Get latest progress info
                    progress_info = monitor.get_progress()
                    
                    # If process has ended, check if it's normal end or error
                    if not monitor.is_training:
                        # Check both monitor.process and current_process since they might be different objects
                        if (monitor.process and monitor.process.returncode != 0) or (current_process and current_process.poll() is not None and current_process.returncode != 0):
                            # Get the return code from whichever process object is available
                            return_code = monitor.process.returncode if monitor.process else current_process.returncode
                            # Get complete output log
                            error_output = "\n".join(progress_info.get("lines", []))
                            if not error_output:
                                error_output = "No output captured from the training process"
                            
                            # Ensure we set the is_completed flag to False for errors
                            progress_info['is_completed'] = False
                            monitor.current_progress['is_completed'] = False
                            
                            # Also set the stage to Error
                            progress_info['stage'] = 'Error'
                            monitor.current_progress['stage'] = 'Error'
                            
                            error_status_html = load_html_template("status_failed_code.html", return_code=return_code, error_output=error_output)
                            yield (
                                initial_stats,
                                error_status_html,
                                "Training failed",
                                gr.update(value="", visible=False),
                                None,
                                None,
                                gr.update(visible=False),
                                gr.update(visible=False),
                                gr.update(value=""),
                                gr.update(value="Custom"),
                                gr.update(value=""),
                                gr.update(value={}),
                                gr.update(value="")
                            )
                            return
                        else:
                            # Only set is_completed to True if there was a successful exit code
                            progress_info['is_completed'] = True
                            monitor.current_progress['is_completed'] = True
                    
                    # Update UI
                    stats = monitor.get_stats()
                    if stats:
                        model_stats = update_model_stats(stats)
                    else:
                        model_stats = initial_stats
                    
                    status_html, best_info, test_html_update, loss_fig, metrics_fig, download_btn_update = update_progress(progress_info)

                    # During training, keep registration panel hidden
                    yield model_stats, status_html, best_info, test_html_update, loss_fig, metrics_fig, download_btn_update, gr.update(visible=False), gr.update(value=""), gr.update(value="Custom"), gr.update(value=""), gr.update(value={}), gr.update(value="")

                except Exception as e:
                    # Get complete output log
                    error_output = "\n".join(progress_info.get("lines", []))
                    if not error_output:
                        error_output = "No output captured from the training process"
                    
                    error_status_html = load_html_template("status_error.html", error_title="Error during training", error_message=str(e))
                    print(f"Error updating UI: {str(e)}")
                    traceback.print_exc()
                    yield initial_stats, error_status_html, "Training error", gr.update(value="", visible=False), None, None, gr.update(visible=False), gr.update(visible=False), gr.update(value=""), gr.update(value="Custom"), gr.update(value=""), gr.update(value={}), gr.update(value="")
                    return
            
            # Check if aborted
            if process_aborted:
                is_training = False
                current_process = None
                aborted_status_html = load_html_template("status_aborted.html")
                yield initial_stats, aborted_status_html, "Training aborted", gr.update(value="", visible=False), None, None, gr.update(visible=False), gr.update(visible=False), gr.update(value=""), gr.update(value="Custom"), gr.update(value=""), gr.update(value={}), gr.update(value="")
                return
            
            # Final update after training ends (only for normal completion)
            if monitor.process and monitor.process.returncode == 0:
                try:
                    progress_info = monitor.get_progress()
                    progress_info['is_completed'] = True
                    monitor.current_progress['is_completed'] = True
                    
                    stats = monitor.get_stats()
                    if stats:
                        model_stats = update_model_stats(stats)
                    else:
                        model_stats = initial_stats
                    
                    status_html, best_info, test_html_update, loss_fig, metrics_fig, download_btn_update = update_progress(progress_info)

                    # Training completed successfully
                    yield (
                        model_stats, status_html, best_info, test_html_update, loss_fig, metrics_fig, download_btn_update
                    )
                except Exception as e:
                    error_output = "\n".join(progress_info.get("lines", []))
                    if not error_output:
                        error_output = "No output captured from the training process"

                    error_status_html = load_html_template("status_error.html", error_title="Error in final update", error_message=str(e))
                    yield initial_stats, error_status_html, "Error in final update", gr.update(value="", visible=False), None, None, gr.update(visible=False)

        except Exception as e:
            # Initialization error, may not have output log
            error_status_html = load_html_template("status_error.html", error_title="Training initialization failed", error_message=str(e))
            yield initial_stats, error_status_html, "Training failed", gr.update(value="", visible=False), None, None, gr.update(visible=False)
        finally:
            is_training = False
            current_process = None

    def handle_config_import(config_path: str) -> List[gr.update]:
        """
        Loads a training configuration from a JSON file and correctly updates the UI components.
        """
        try:
            if not os.path.exists(config_path):
                gr.Warning(f"Configuration file not found: {config_path}")
                return [gr.update() for _ in input_components]
            
            with open(config_path, "r", encoding='utf-8') as f:
                config = json.load(f)
            
            update_dict = {comp: gr.update() for comp in input_components}
            def get_config_val(key, default):
                return config.get(key, default)

            update_dict[plm_model] = gr.update(value=get_config_val("plm_model", plm_model.value))
            
            dataset_selection_value = get_config_val("dataset_selection", "Custom")
            update_dict[is_custom_dataset] = gr.update(value=dataset_selection_value)
            
            if dataset_selection_value == "Pre-defined":
                dataset_config_value = get_config_val("dataset_config", "Demo_Solubility")
                # Ensure the value is in the choices list
                if dataset_config_value in list(dataset_configs.keys()):
                    update_dict[dataset_config] = gr.update(visible=True, value=dataset_config_value)
                else:
                    # If not in choices, use the first available choice
                    update_dict[dataset_config] = gr.update(visible=True, value=list(dataset_configs.keys())[0])
            else:
                # For custom dataset, hide dataset_config and don't set any value
                update_dict[dataset_config] = gr.update(visible=False)
                
            update_dict[dataset_custom] = gr.update(
                visible=(dataset_selection_value == "Custom"),
                value=get_config_val("dataset_custom", "")
            )

            is_interactive = (dataset_selection_value == "Custom")
            
            update_dict[custom_dataset_settings] = gr.update(visible=is_interactive)
            
            update_dict[problem_type] = gr.update(
                value=get_config_val("problem_type", "single_label_classification"), 
                interactive=is_interactive
            )
            update_dict[num_labels] = gr.update(
                value=get_config_val("num_labels", 2), 
                interactive=is_interactive
            )

            metrics_value = get_config_val("metrics", ["accuracy"])
            if isinstance(metrics_value, str):
                metrics_value = metrics_value.split(",")
            update_dict[metrics] = gr.update(
                value=metrics_value, 
                interactive=is_interactive
            )
            
            update_dict[monitored_metrics] = gr.update(
                value=get_config_val("monitored_metrics", "accuracy"), 
                interactive=is_interactive
            )
            update_dict[monitored_strategy] = gr.update(
                value=get_config_val("monitored_strategy", "max"), 
                interactive=is_interactive
            )
            
            training_method_value = get_config_val("training_method", "freeze")
            update_dict[training_method] = gr.update(value=training_method_value)
            
            update_dict[structure_seq] = gr.update(
                visible=(training_method_value == "ses-adapter"),
                value=get_config_val("structure_seq", ["foldseek_seq", "ss8_seq"])
            )
            plm_val = get_config_val("plm_model", "")
            update_dict[pdb_dir] = gr.update(value=get_config_val("pdb_dir", ""))
            update_dict[pdb_dir_row] = gr.update(
                visible=(training_method_value == "ses-adapter") or (plm_val and (str(plm_val).startswith("ProSST") or str(plm_val).startswith("SaProt") or str(plm_val).startswith("ProtSSN")))
            )
            update_dict[lora_params_row] = gr.update(
                visible=(training_method_value in ["plm-lora", "plm-qlora", "plm-adalora", "plm-dora", "plm-ia3"])
            )
            
            update_dict[pooling_method] = gr.update(value=get_config_val("pooling_method", "mean"))
            
            batch_mode_value = get_config_val("batch_mode", "Batch Size Mode")
            update_dict[batch_mode] = gr.update(value=batch_mode_value)
            update_dict[batch_size] = gr.update(
                visible=(batch_mode_value == "Batch Size Mode"), 
                value=get_config_val("batch_size", 1)
            )
            update_dict[batch_token] = gr.update(
                visible=(batch_mode_value == "Batch Token Mode"), 
                value=get_config_val("batch_token", 2000)
            )

            update_dict[learning_rate] = gr.update(value=get_config_val("learning_rate", 5e-4))
            update_dict[num_epochs] = gr.update(value=get_config_val("num_epochs", 20))
            update_dict[max_seq_len] = gr.update(value=get_config_val("max_seq_len", None))
            update_dict[gradient_accumulation_steps] = gr.update(value=get_config_val("gradient_accumulation_steps", 1))
            update_dict[warmup_steps] = gr.update(value=get_config_val("warmup_steps", 0))
            update_dict[scheduler_type] = gr.update(value=get_config_val("scheduler", None))
            
            update_dict[output_model_name] = gr.update(value=get_config_val("output_model_name", "demo.pt"))
            od_val = get_config_val("output_dir", "demo")
            if od_val and not str(od_val).startswith("ckpt") and not os.path.isabs(str(od_val)):
                od_val = "ckpt/" + str(od_val)
            update_dict[output_dir] = gr.update(value=od_val)
            
            wandb_enabled = get_config_val("wandb_enabled", False)
            update_dict[wandb_logging] = gr.update(value=wandb_enabled)
            update_dict[wandb_project] = gr.update(visible=wandb_enabled, value=get_config_val("wandb_project", ""))
            update_dict[wandb_entity] = gr.update(visible=wandb_enabled, value=get_config_val("wandb_entity", ""))
            
            update_dict[patience] = gr.update(value=get_config_val("patience", 10))
            update_dict[num_workers] = gr.update(value=get_config_val("num_workers", 4))
            update_dict[max_grad_norm] = gr.update(value=get_config_val("max_grad_norm", -1))
            
            update_dict[lora_r] = gr.update(value=get_config_val("lora_r", 8))
            update_dict[lora_alpha] = gr.update(value=get_config_val("lora_alpha", 32))
            update_dict[lora_dropout] = gr.update(value=get_config_val("lora_dropout", 0.1))
            update_dict[lora_target_modules] = gr.update(value=get_config_val("lora_target_modules", "query,key,value"))

            gr.Info(f"Configuration successfully imported from {config_path}")
            
            return [update_dict[comp] for comp in input_components]

        except (json.JSONDecodeError, KeyError) as e:
            gr.Warning(f"Error parsing configuration file: {str(e)}")
            return [gr.update() for _ in input_components]
        except Exception as e:
            gr.Warning(f"An unexpected error occurred during import: {str(e)}")
            return [gr.update() for _ in input_components]
 
    # define all input components
    input_components = [
        plm_model,
        is_custom_dataset,
        dataset_config,
        dataset_custom,
        problem_type,
        num_labels,
        metrics,
        training_method,
        pooling_method,
        batch_mode,
        batch_size,
        batch_token,
        learning_rate,
        num_epochs,
        max_seq_len,
        gradient_accumulation_steps,
        warmup_steps,
        scheduler_type,
        output_model_name,
        output_dir,
        wandb_logging,
        wandb_project,
        wandb_entity,
        patience,
        num_workers,
        max_grad_norm,
        structure_seq,
        pdb_dir,
        lora_r,
        lora_alpha,
        lora_dropout,
        lora_target_modules,
        monitored_metrics,
        monitored_strategy,
        sequence_column_name,
        label_column_name,
        training_mode,
        transfer_model_dropdown,
    ]
    import_config_button.click(
        fn=handle_config_import,
        inputs=[config_path_input],
        outputs=input_components
    )
    preview_button.click(
        fn=handle_train_tab_command_preview,
        inputs=input_components,
        outputs=[command_preview]
    )
    
    train_button.click(
        fn=reset_train_tab_UI,
        outputs=[
            model_stats, progress_status, best_model_info, test_results_html, loss_plot, metrics_plot, download_csv_btn
        ]
    ).then(
        fn=handle_train,
        inputs=input_components,
        outputs=[
            model_stats, progress_status, best_model_info, test_results_html, loss_plot, metrics_plot, download_csv_btn
        ]
    )

    # bind abort button
    abort_button.click(
        fn=handle_train_tab_abort,
        outputs=[progress_status, model_stats, best_model_info, test_results_html, loss_plot, metrics_plot, download_csv_btn]
    )
    
    wandb_logging.change(
        fn=update_wandb_visibility_UI,
        inputs=[wandb_logging],
        outputs=[wandb_project, wandb_entity]
    )

    def update_train_tab_dataset_preview_UI(dataset_type=None, dataset_name=None, custom_dataset=None):
        """Update dataset preview content for Gradio UI
        Args:
            dataset_type: dataset type (Custom or Pre-defined)
            dataset_name: predefined dataset name
            custom_dataset: custom dataset path
        Returns:
        """
        # Determine which dataset to use based on selection
        if dataset_type == "Custom" and custom_dataset:
            try:
                # Try to load custom dataset
                dataset = load_dataset(custom_dataset)
                stats_html = load_html_template(
                    "dataset_stats_table.html", 
                    dataset_name=custom_dataset, 
                    train_count=len(dataset["train"]) if "train" in dataset else 0, 
                    val_count=len(dataset["validation"]) if "validation" in dataset else 0, 
                    test_count=len(dataset["test"]) if "test" in dataset else 0
                )
                
                # Get sample data points
                split = "train" if "train" in dataset else list(dataset.keys())[0]
                samples = dataset[split].select(range(min(3, len(dataset[split]))))
                if len(samples) == 0:
                    return gr.update(value=stats_html), gr.update(value=[["No data available", "-", "-"]], headers=["Name", "Sequence", "Label"]), gr.update(open=True)
                
                # Get fields actually present in the dataset
                available_fields = list(samples[0].keys())
                capitalized_headers = [field.capitalize() for field in available_fields]
                
                # Build sample data - ensure consistent structure
                sample_data = []
                for i, sample in enumerate(samples):
                    row_data = []
                    for field in available_fields:
                        # Truncate long sequences for display
                        value = str(sample[field])
                        if len(value) > 100:
                            value = value[:100] + "..."
                        row_data.append(value)
                    sample_data.append(row_data)
                
                return gr.update(value=stats_html), gr.update(value=sample_data, headers=capitalized_headers), gr.update(open=True)
            except Exception as e:
                error_html = load_html_template("error_loading_dataset.html", error_message=str(e))
                return gr.update(value=error_html), gr.update(value=[["Error", str(e), "-"]], headers=["Error", "Message", "Status"]), gr.update(open=True)
        
        # Use predefined dataset
        elif dataset_type == "Pre-defined" and dataset_name:
            try:
                config_path = dataset_configs[dataset_name]
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                # Load dataset statistics
                dataset = load_dataset(config["dataset"])
                stats_html = load_html_template(
                    "dataset_stats_table.html", 
                    dataset_name=config["dataset"], 
                    train_count=len(dataset["train"]) if "train" in dataset else 0, 
                    val_count=len(dataset["validation"]) if "validation" in dataset else 0, 
                    test_count=len(dataset["test"]) if "test" in dataset else 0
                    )
                
                # Get sample data points and available fields
                samples = dataset["train"].select(range(min(3, len(dataset["train"]))))
                if len(samples) == 0:
                    return gr.update(value=stats_html), gr.update(value=[["No data available", "-", "-"]], headers=["Name", "Sequence", "Label"]), gr.update(open=True)
                
                # Get fields actually present in the dataset
                available_fields = list(samples[0].keys())
                capitalized_headers = [field.capitalize() for field in available_fields]
                
                # Build sample data - ensure consistent structure
                sample_data = []
                for i, sample in enumerate(samples):
                    row_data = []
                    for field in available_fields:
                        # Truncate long sequences for display
                        value = str(sample[field])
                        if len(value) > 100:
                            value = value[:100] + "..."
                        row_data.append(value)
                    sample_data.append(row_data)
                
                return gr.update(value=stats_html), gr.update(value=sample_data, headers=capitalized_headers), gr.update(open=True)
            except Exception as e:
                error_html = load_html_template("error_loading_dataset.html", error_message=str(e))
                return gr.update(value=error_html), gr.update(value=[["Error", str(e), "-"]], headers=["Error", "Message", "Status"]), gr.update(open=True)
        
        # If no valid dataset information provided
        return gr.update(value=""), gr.update(value=[["No dataset selected", "-", "-"]], headers=["Status", "Message", "Action"]), gr.update(open=True)

    # Preview button click event
    dataset_preview_button.click(
        fn=update_train_tab_dataset_preview_UI,
        inputs=[is_custom_dataset, dataset_config, dataset_custom],
        outputs=[dataset_stats_md, preview_table, preview_accordion]
    )

    # add function for custom dataset settings
    def update_train_tab_dataset_settings_UI(choice, dataset_name=None):
        """Update dataset settings for Gradio UI
        Args:
            choice: dataset type (Custom or Pre-defined)
            dataset_name: predefined dataset name
        Returns:
        """
        if choice == "Pre-defined":
            # Pre-defined: Accordion visible but collapsed, fields read-only
            result = {
                dataset_config: gr.update(visible=True),
                dataset_custom: gr.update(visible=False),
                custom_dataset_settings: gr.update(visible=True, open=False)  # Collapsed
            }

            # if a specific dataset is selected, automatically load the config
            if dataset_name and dataset_name in dataset_configs:
                with open(dataset_configs[dataset_name], 'r') as f:
                    config = json.load(f)

                # process metrics, convert string to list to fit multi-select components
                metrics_value = config.get("metrics", "accuracy,mcc,f1,precision,recall,auroc")
                if isinstance(metrics_value, str):
                    metrics_value = metrics_value.split(",")

                # process monitored_metrics, single select
                monitored_metrics_value = config.get("monitor", "accuracy")
                monitored_strategy_value = config.get("monitor_strategy", "max")
                result.update({
                    problem_type: gr.update(value=config.get("problem_type", "single_label_classification"), interactive=False),
                    num_labels: gr.update(value=config.get("num_labels", 2), interactive=False),
                    metrics: gr.update(value=metrics_value, interactive=False),
                    monitored_metrics: gr.update(value=monitored_metrics_value, interactive=False),
                    monitored_strategy: gr.update(value=monitored_strategy_value, interactive=False),
                    sequence_column_name: gr.update(value=config.get("sequence_column_name", "aa_seq"), interactive=False),
                    label_column_name: gr.update(value=config.get("label_column_name", "label"), interactive=False)
                })
            return result
        else:
            # Custom dataset: Accordion visible and open, fields editable
            default_metrics = ["accuracy", "mcc", "f1", "precision", "recall", "auroc"]
            default_monitored_metrics = ["accuracy"]
            default_monitored_strategy = ["max"]
            return {
                dataset_config: gr.update(visible=False),
                dataset_custom: gr.update(visible=True),
                custom_dataset_settings: gr.update(visible=True, open=True),  # Expanded
                problem_type: gr.update(value="single_label_classification", interactive=True),
                num_labels: gr.update(value=2, interactive=True),
                metrics: gr.update(value=default_metrics, interactive=True),
                monitored_metrics: gr.update(value=default_monitored_metrics, interactive=True),
                monitored_strategy: gr.update(value=default_monitored_strategy, interactive=True),
                sequence_column_name: gr.update(value="aa_seq", interactive=True),
                label_column_name: gr.update(value="label", interactive=True)
            }

    # bind dataset settings update event
    is_custom_dataset.change(
        fn=update_train_tab_dataset_settings_UI,
        inputs=[is_custom_dataset, dataset_config],
        outputs=[dataset_config, dataset_custom, custom_dataset_settings, problem_type, num_labels, metrics, monitored_metrics, monitored_strategy, sequence_column_name, label_column_name]
    )

    # Transfer Learning UI handlers
    def update_training_mode_ui(mode):
        """Update UI based on training mode selection"""
        is_continue = (mode == "Continue Training")
        return {
            plm_model: gr.update(visible=not is_continue),
            transfer_model_folder: gr.update(visible=is_continue),
            transfer_model_dropdown: gr.update(visible=is_continue),
            transfer_settings_row: gr.update(visible=is_continue),
        }

    def on_transfer_folder_change(folder_path):
        """Update model dropdown when folder changes"""
        if not folder_path:
            return gr.update(choices=[], value=None), {}

        choices = scan_models_in_folder(folder_path)
        if not choices:
            return gr.update(choices=[], value=None), {}

        # Auto-select first model and load its info
        first_model_path = choices[0][1]
        model_info = load_model_config(first_model_path)

        return (
            gr.update(choices=choices, value=first_model_path),
            model_info
        )

    def on_transfer_model_change(model_path):
        """Update model info and hyperparameters when model changes"""
        if not model_path:
            return [gr.update()] + [{}] + [gr.update()] * 15  # Return empty updates

        config = load_model_config(model_path)
        if not config:
            return [gr.update()] + [{}] + [gr.update()] * 15

        # Get PLM model from config (full path like "facebook/esm2_t30_150M_UR50D")
        plm_model_full_path = config.get("plm_model", "facebook/esm2_t6_8M_UR50D")

        # Reverse mapping: convert full path to display name (e.g., "ESM2-150M")
        plm_model_value = plm_models_reverse.get(plm_model_full_path, "ESM2-8M")

        # Determine batch mode based on config
        has_batch_size = config.get("batch_size") is not None
        has_batch_token = config.get("batch_token") is not None
        batch_mode_value = "Batch Size Mode" if has_batch_size else "Batch Token Mode"

        # Check if training method is LoRA-based
        training_method_value = config.get("training_method", "freeze")
        is_lora_method = training_method_value in ["plm-lora", "plm-qlora", "plm-adalora", "plm-dora", "plm-ia3"]

        return [
            gr.update(value=plm_model_value),  # plm_model (update value even though hidden)
            config,  # model_info_display
            gr.update(value=training_method_value, interactive=False),  # training_method (read-only)
            gr.update(value=config.get("pooling_method", "mean"), interactive=False),  # pooling_method (read-only)
            gr.update(value=config.get("learning_rate", 5e-4)),  # learning_rate
            gr.update(value=config.get("num_epochs", 20)),  # num_epochs
            gr.update(value=batch_mode_value),  # batch_mode
            gr.update(value=config.get("batch_size", 8)),  # batch_size
            gr.update(value=config.get("batch_token", 4096)),  # batch_token
            gr.update(value=config.get("max_seq_len", -1)),  # max_seq_len
            gr.update(value=config.get("gradient_accumulation_steps", 1)),  # gradient_accumulation_steps
            gr.update(value=config.get("warmup_steps", 0)),  # warmup_steps
            gr.update(value=config.get("scheduler", None)),  # scheduler_type
            gr.update(value=config.get("lora_r", 8) if is_lora_method else 8),  # lora_r
            gr.update(value=config.get("lora_alpha", 32) if is_lora_method else 32),  # lora_alpha
            gr.update(value=config.get("lora_dropout", 0.1) if is_lora_method else 0.1),  # lora_dropout
            gr.update(value=config.get("lora_target_modules", "query,key,value") if is_lora_method else "query,key,value"),  # lora_target_modules
        ]

    def browse_trained_models_old():
        """[DEPRECATED] Browse trained models in ckpt directory"""
        import os
        from pathlib import Path

        models = []
        ckpt_root = Path("ckpt")

        if ckpt_root.exists():
            # Search for best_model.pt files
            for model_file in ckpt_root.rglob("best_model.pt"):
                try:
                    # Get model directory info
                    model_dir = model_file.parent
                    relative_path = str(model_file.relative_to(ckpt_root.resolve().parent))

                    # Try to read config if exists
                    config_files = list(model_dir.glob("*.json"))
                    model_name = model_dir.name

                    if config_files:
                        with open(config_files[0], 'r') as f:
                            config = json.load(f)
                            dataset_name = config.get('dataset', 'Unknown')
                            model_name = f"{model_dir.name} ({dataset_name})"

                    models.append({
                        "name": model_name,
                        "path": relative_path,
                        "size_mb": round(model_file.stat().st_size / (1024*1024), 2),
                        "modified": time.strftime("%Y-%m-%d %H:%M", time.localtime(model_file.stat().st_mtime))
                    })
                except Exception:
                    continue

        # Sort by modification time (newest first)
        models.sort(key=lambda x: x.get("modified", ""), reverse=True)

        if models:
            # Return as formatted choices
            return gr.update(
                choices=[m["path"] for m in models],
                value=models[0]["path"] if models else ""
            )
        else:
            return gr.update(choices=[], value="")

    def load_model_info(model_path):
        """Load and display model information"""
        if not model_path or not os.path.exists(model_path):
            return gr.update(visible=False), {}

        try:
            import torch

            # Try to load checkpoint info
            checkpoint = torch.load(model_path, map_location='cpu')

            info = {
                "model_path": model_path,
                "file_size_mb": round(os.path.getsize(model_path) / (1024*1024), 2)
            }

            # Try to load config from same directory
            model_dir = Path(model_path).parent
            config_files = list(model_dir.glob("*.json"))

            if config_files:
                with open(config_files[0], 'r') as f:
                    config = json.load(f)
                    info.update({
                        "plm_model": config.get("plm_model", "Unknown"),
                        "training_method": config.get("training_method", "Unknown"),
                        "problem_type": config.get("problem_type", "Unknown"),
                        "num_labels": config.get("num_labels", "Unknown"),
                        "dataset": config.get("dataset", "Unknown")
                    })

            # Check for LoRA adapters
            has_adapter = False
            for suffix in ['_lora', '_qlora', '_dora', '_adalora', '_ia3']:
                adapter_path = model_path.replace('.pt', suffix)
                if os.path.exists(adapter_path):
                    has_adapter = True
                    info["adapter_path"] = adapter_path
                    break

            info["has_adapter"] = has_adapter

            return gr.update(visible=True), info

        except Exception as e:
            return gr.update(visible=False), {"error": str(e)}

    # Bind training mode events
    training_mode.change(
        fn=update_training_mode_ui,
        inputs=[training_mode],
        outputs=[plm_model, transfer_model_folder, transfer_model_dropdown, transfer_settings_row]
    )

    transfer_model_folder.change(
        fn=on_transfer_folder_change,
        inputs=[transfer_model_folder],
        outputs=[transfer_model_dropdown, model_info_display]
    )

    transfer_model_dropdown.change(
        fn=on_transfer_model_change,
        inputs=[transfer_model_dropdown],
        outputs=[
            plm_model,  # Update PLM model from config (even though hidden)
            model_info_display,
            training_method,
            pooling_method,
            learning_rate,
            num_epochs,
            batch_mode,
            batch_size,
            batch_token,
            max_seq_len,
            gradient_accumulation_steps,
            warmup_steps,
            scheduler_type,
            lora_r,
            lora_alpha,
            lora_dropout,
            lora_target_modules,
        ]
    )

    def handle_dataset_config_change(x):
        """Handle dataset config change event
        Args:
            x: dataset config
        Returns:
        """
        return update_train_tab_dataset_settings_UI("Pre-defined", x)

    dataset_config.change(
        fn=handle_dataset_config_change,
        inputs=[dataset_config],
        outputs=[dataset_config, dataset_custom, custom_dataset_settings, problem_type, num_labels, metrics, monitored_metrics, monitored_strategy, sequence_column_name, label_column_name]
    )

    # Return components that need to be accessed from outside
    return {
        "output_text": progress_status,
        "loss_plot": loss_plot,
        "metrics_plot": metrics_plot,
        "train_button": train_button,
        "monitor": monitor,
        "test_results_html": test_results_html,  # add test results HTML component
        "components": {
            "plm_model": plm_model,
            "dataset_config": dataset_config,
            "training_method": training_method,
            "pooling_method": pooling_method,
            "batch_mode": batch_mode,
            "batch_size": batch_size,
            "batch_token": batch_token,
            "learning_rate": learning_rate,
            "num_epochs": num_epochs,
            "max_seq_len": max_seq_len,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "warmup_steps": warmup_steps,
            "scheduler_type": scheduler_type,
            "output_model_name": output_model_name,
            "output_dir": output_dir,
            "wandb_logging": wandb_logging,
            "wandb_project": wandb_project,
            "wandb_entity": wandb_entity,
            "patience": patience,
            "num_workers": num_workers,
            "max_grad_norm": max_grad_norm,
            "structure_seq": structure_seq,
            "pdb_dir": pdb_dir,
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "lora_target_modules": lora_target_modules,
            "sequence_column_name": sequence_column_name,
            "label_column_name": label_column_name,
        }
    }
