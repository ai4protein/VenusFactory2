import os
import json
import gradio as gr
import time
from datasets import load_dataset
import pandas as pd
from typing import Any, Dict, Union, Optional, Generator, List
from dataclasses import dataclass
from .utils.command import preview_command, save_arguments, build_command_list
from .utils.monitor import TrainingMonitor
import traceback
import base64
import tempfile
import numpy as np
import queue
import subprocess
import sys
import threading

@dataclass
class TrainingArgs:
    def __init__(self, args: list, plm_models: dict, dataset_configs: dict):
        # Basic parameters
        self.plm_model = plm_models[args[0]]
        
        # 处理自定义数据集或预定义数据集
        self.dataset_selection = args[1]  # "Use Custom Dataset" 或 "Use Pre-defined Dataset"
        if self.dataset_selection == "Use Pre-defined Dataset":
            self.dataset_config = dataset_configs[args[2]]
            self.dataset_custom = None
            # 从配置加载问题类型等
            with open(self.dataset_config, 'r') as f:
                config = json.load(f)
            self.problem_type = config.get("problem_type", "single_label_classification")
            self.num_labels = config.get("num_labels", 2)
            self.metrics = config.get("metrics", "accuracy,mcc,f1,precision,recall,auroc")
        else:
            self.dataset_config = None
            self.dataset_custom = args[3]  # Custom dataset path
            self.problem_type = args[4]
            self.num_labels = args[5]
            self.metrics = args[6]
            # 如果metrics是列表，转换为逗号分隔的字符串
            if isinstance(self.metrics, list):
                self.metrics = ",".join(self.metrics)
            
        # Training method parameters
        self.training_method = args[7]
        self.pooling_method = args[8]
        
        # Batch processing parameters
        self.batch_mode = args[9]
        if self.batch_mode == "Batch Size Mode":
            self.batch_size = args[10]
        else:
            self.batch_token = args[11]
        
        # Training parameters
        self.learning_rate = args[12]
        self.num_epochs = args[13]
        self.max_seq_len = args[14]
        self.gradient_accumulation_steps = args[15]
        self.warmup_steps = args[16]
        self.scheduler = args[17]

        # Output parameters
        self.output_model_name = args[18]
        self.output_dir = args[19]
        
        # Wandb parameters
        self.wandb_enabled = args[20]
        if self.wandb_enabled:
            self.wandb_project = args[21]
            self.wandb_entity = args[22]
        
        # Other parameters
        self.patience = args[23]
        self.num_workers = args[24]
        self.max_grad_norm = args[25]
        self.structure_seq = args[26]

        # LoRA parameters
        self.lora_r = args[27]
        self.lora_alpha = args[28]
        self.lora_dropout = args[29]
        self.lora_target_modules = [m.strip() for m in args[30].split(",")] if args[30] else []

        # Monitored parameters
        self.monitored_metrics = args[31]
        self.monitored_strategy = args[32]

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

        # 添加数据集相关参数
        if self.dataset_selection == "Use Pre-defined Dataset":
            args_dict["dataset_config"] = self.dataset_config
        else:
            args_dict["dataset"] = self.dataset_custom
            args_dict["problem_type"] = self.problem_type
            args_dict["num_labels"] = self.num_labels
            args_dict["metrics"] = self.metrics

        # Add LoRA parameters
        if self.training_method in ["plm-lora", "plm-qlora", "plm_adalora", "plm_dora", "plm_ia3"]:
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

    with gr.Tab("Training"):
        # Model and Dataset Selection
        gr.Markdown("### Model and Dataset Configuration")

        # Original training interface components
        with gr.Group():
            with gr.Row():
                with gr.Column(scale=4):
                    with gr.Row():
                        plm_model = gr.Dropdown(
                            choices=list(plm_models.keys()),
                            label="Protein Language Model",
                            value=list(plm_models.keys())[0],
                            scale=2
                        )
                    
                        # 新增数据集选择方式
                        is_custom_dataset = gr.Radio(
                            choices=["Use Custom Dataset", "Use Pre-defined Dataset"],
                            label="Dataset Selection",
                            value="Use Pre-defined Dataset",
                            scale=3
                        )
                
                        dataset_config = gr.Dropdown(
                            choices=list(dataset_configs.keys()),
                            label="Dataset Configuration",
                            value=list(dataset_configs.keys())[0],
                            visible=True,
                            scale=2
                        )
                        
                        dataset_custom = gr.Textbox(
                            label="Custom Dataset Path",
                            placeholder="Huggingface Dataset eg: user/dataset",
                            visible=False,
                            scale=2
                        )
                
                # 将预览按钮放在单独的列中，并添加样式
                with gr.Column(scale=1, min_width=120, elem_classes="preview-button-container"):
                    dataset_preview_button = gr.Button(
                        "Preview Dataset", 
                        variant="primary", 
                        size="lg",
                        elem_classes="preview-button"
                    )
                
            # 自定义数据集的额外配置选项（单独一行）
            with gr.Group(visible=True) as custom_dataset_settings:
                with gr.Row():
                    problem_type = gr.Dropdown(
                        choices=["single_label_classification", "multi_label_classification", "regression"],
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
                        choices=["accuracy", "recall", "precision", "f1", "mcc", "auroc", "f1_max", "spearman_corr", "mse"],
                        label="Metrics",
                        value=["accuracy", "mcc", "f1", "precision", "recall", "auroc"],
                        scale=101,
                        multiselect=True,
                        interactive=False
                    )
                
                with gr.Row():
                    monitored_metrics = gr.Dropdown(
                        choices=["accuracy", "recall", "precision", "f1", "mcc", "auroc", "f1_max", "spearman_corr", "mse"],
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

            # ! add for plm-lora, plm-qlora, plm_adalora, plm_dora, plm_ia3
            with gr.Row(visible=False) as lora_params_row:
                # gr.Markdown("#### LoRA Parameters")
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
                        # info="LoRA will be applied to these modules"
                    )

        # 将数据统计和表格都放入折叠面板
        with gr.Row():
            with gr.Accordion("Dataset Preview", open=False) as preview_accordion:
                # 数据统计区域
                with gr.Row():
                    dataset_stats_md = gr.HTML("", elem_classes=["dataset-stats"])
                
                # 表格区域
                with gr.Row():
                    preview_table = gr.Dataframe(
                        headers=["Name", "Sequence", "Label"],
                        value=[["No dataset selected", "-", "-"]],
                        wrap=True,
                        interactive=False,
                        row_count=3,
                        elem_classes=["preview-table"]
                    )

        # Add CSS styles
        gr.HTML("""
        <style>
            /* 数据统计样式 */
            .dataset-stats {
                margin: 0 0 15px 0;
                padding: 0;
            }
            
            .dataset-stats table {
                width: 100%;
                border-collapse: collapse;
                font-size: 0.9em;
                box-shadow: 0 2px 4px rgba(0,0,0,0.05);
                border-radius: 8px;
                overflow: hidden;
                table-layout: fixed;
            }
            
            .dataset-stats th {
                background-color: #e0e0e0;
                font-weight: bold;
                padding: 6px 10px;
                text-align: center;
                border: 1px solid #ddd;
                font-size: 0.95em;
                white-space: nowrap;
                overflow: hidden;
                min-width: 120px;
            }
            
            .dataset-stats td {
                padding: 6px 10px;
                text-align: center;
                border: 1px solid #ddd;
            }
            
            .dataset-stats h2 {
                font-size: 1.1em;
                margin: 0 0 10px 0;
                text-align: center;
            }
            
            /* 表格样式 */
            .preview-table table {
                background-color: white !important;
                font-size: 0.9em !important;
                width: 100%;
                table-layout: fixed !important;
            }
            
            .preview-table .gr-block.gr-box {
                background-color: transparent !important;
            }
            
            .preview-table .gr-input-label {
                background-color: transparent !important;
            }

            /* 表格外观增强 */
            .preview-table table {
                margin-top: 0;
                border-radius: 8px;
                overflow: hidden;
                box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            }
            
            /* 表头样式 */
            .preview-table th {
                background-color: #e0e0e0 !important;
                font-weight: bold !important;
                padding: 6px !important;
                border-bottom: 1px solid #ccc !important;
                font-size: 0.95em !important;
                text-align: center !important;
                white-space: nowrap !important;
                min-width: 120px !important;
            }
            
            /* 单元格样式 */
            .preview-table td {
                padding: 4px 6px !important;
                max-width: 300px !important;
                overflow: hidden;
                text-overflow: ellipsis;
                white-space: nowrap;
                text-align: left !important;
            }
            
            /* 悬停效果 */
            .preview-table tr:hover {
                background-color: #f0f0f0 !important;
            }
            
            /* 折叠面板样式 */
            .gr-accordion {
                border: 1px solid #e0e0e0;
                border-radius: 8px;
                overflow: hidden;
                margin-bottom: 15px;
            }
            
            /* 折叠面板标题样式 */
            .gr-accordion .label-wrap {
                background-color: #f5f5f5;
                padding: 8px 15px;
                font-weight: bold;
            }
            
            .preview-button {
                height: 86px !important;
            }
            
            /* Center Model Statistics Table */
            .center-table-content td, .center-table-content th {
                text-align: center !important;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
                padding: 10px !important;
            }
            
            .center-table-content table {
                width: 100% !important;
                border-collapse: collapse !important;
                margin-bottom: 20px !important;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1) !important;
                border-radius: 8px !important;
                overflow: hidden !important;
            }
            
            .center-table-content th {
                background-color: #f0f4f8 !important;
                color: #2c3e50 !important;
                font-weight: 600 !important;
                border-bottom: 2px solid #ddd !important;
            }
            
            .center-table-content tr:nth-child(even) {
                background-color: #f9f9f9 !important;
            }
            
            .center-table-content tr:hover {
                background-color: #f0f7ff !important;
            }
            
            /* Improve readability of progress bars */
            .progress-container {
                margin-bottom: 20px !important;
            }
            
            .progress-bar {
                transition: width 0.5s ease-in-out !important;
            }
            
            .status-message {
                margin-bottom: 8px !important;
                font-weight: 500 !important;
            }
        </style>
        """, visible=True)

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
                        minimum=1,
                        maximum=128,
                        value=16,
                        step=1,
                        label="Batch Size",
                        visible=True
                    )
                    
                    batch_token = gr.Slider(
                        minimum=1000,
                        maximum=50000,
                        value=10000,
                        step=1000,
                        label="Tokens per Batch",
                        visible=False
                    )

        def update_batch_inputs(mode):
            return {
                batch_size: gr.update(visible=mode == "Batch Size Mode"),
                batch_token: gr.update(visible=mode == "Batch Token Mode")
            }

        # Update visibility when mode changes
        batch_mode.change(
            fn=update_batch_inputs,
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
                        choices=["full", "freeze", "ses-adapter", "plm-lora", "plm-qlora", "plm_adalora", "plm_dora", "plm_ia3"],
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
            
            def update_training_method(method):
                return {
                    structure_seq: gr.update(visible=method == "ses-adapter"),
                    lora_params_row: gr.update(visible=method in ["plm-lora", "plm-qlora", "plm_adalora", "plm_dora", "plm_ia3"])
                }

            # Add training_method change event
            training_method.change(
                fn=update_training_method,
                inputs=[training_method],
                outputs=[structure_seq, lora_params_row]
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
                        minimum=0.1, maximum=10.0, value=-1, step=0.1,
                        label="Max Gradient Norm (-1 for no clipping)"
                    )
                with gr.Column(scale=1, min_width=150):
                    num_workers = gr.Slider(
                        minimum=0, maximum=16, value=4, step=1,
                        label="Number of Workers"
                    )
                
        # Output and Logging Settings
        gr.Markdown("### Output and Logging Settings")
        with gr.Row():
            with gr.Column():
                output_dir = gr.Textbox(
                    label="Save Directory",
                    value="demo",
                    placeholder="Path to save training results"
                )
                
                output_model_name = gr.Textbox(
                    label="Output Model Name",
                    value="demo.pt",
                    placeholder="Name of the output model file"
                )

            with gr.Column():
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

        # Training Control and Output
        gr.Markdown("### Training Control")
        with gr.Row():
            preview_button = gr.Button("Preview Command")
            abort_button = gr.Button("Abort", variant="stop")
            train_button = gr.Button("Start", variant="primary")
        
        with gr.Row():
            command_preview = gr.Code(
                label="Command Preview",
                language="shell",
                interactive=False,
                visible=False
            )
        
        # Model Statistics Section
        gr.Markdown("### Model Statistics")
        with gr.Row():
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
            
            return [
                ["Training Model", str(adapter_total), str(adapter_trainable), "-"],
                ["Pre-trained Model", str(pretrain_total), str(pretrain_trainable), "-"],
                ["Combined Model", str(combined_total), str(combined_trainable), str(trainable_percentage)]
            ]

        # Training Progress
        gr.Markdown("### Training Progress")
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
                label="Status"
            )

        with gr.Row():
            best_model_info = gr.Textbox(
                value="Best Model: None",
                label="Best Performance",
                interactive=False
            )

        # Add test results HTML display
        with gr.Row():
            test_results_html = gr.HTML(
                value="",
                label="Test Results",
                visible=True
            )
            
        with gr.Row():
            with gr.Column(scale=4):
                pass
            with gr.Column(scale=1):  # 限制列的最大宽度
                download_csv_btn = gr.DownloadButton(
                    "Download CSV", 
                    visible=False,
                    size="lg"
                )
            # 添加一个空列来占据剩余空间
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
                fresh_status_html = """
                <div style="background-color: #f8f9fa; border-radius: 10px; padding: 20px; margin-bottom: 15px; box-shadow: 0 2px 5px rgba(0,0,0,0.05);">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 12px;">
                        <div>
                            <span style="font-weight: 600; font-size: 16px;">Training Status: </span>
                            <span style="color: #1976d2; font-weight: 500; font-size: 16px;">Click Start to train your model</span>
                        </div>
                    </div>
                </div>
                """
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
                    error_status_html = """
                    <div style="background-color: #ffebee; border-radius: 10px; padding: 20px; margin-bottom: 15px; box-shadow: 0 2px 5px rgba(0,0,0,0.05);">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 12px;">
                            <div>
                                <span style="font-weight: 600; font-size: 16px;">Training Status: </span>
                                <span style="color: #c62828; font-weight: 500; font-size: 16px;">Failed</span>
                            </div>
                        </div>
                    </div>
                    """
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
                        """
                        <div style="background-color: #f8f9fa; border-radius: 10px; padding: 20px; margin-bottom: 15px; box-shadow: 0 2px 5px rgba(0,0,0,0.05);">
                            <div style="display: flex; justify-content: space-between; margin-bottom: 12px;">
                                <div>
                                    <span style="font-weight: 600; font-size: 16px;">Training Status: </span>
                                    <span style="color: #1976d2; font-weight: 500; font-size: 16px;">Waiting to start...</span>
                                </div>
                            </div>
                        </div>
                        """,
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
            total_epochs = progress_info.get('total_epochs', 0)  # 获取总epoch数
            test_results_html = progress_info.get('test_results_html', '')  # 获取测试结果HTML
            test_metrics = progress_info.get('test_metrics', {})  # 获取测试指标
            is_completed = progress_info.get('is_completed', False)  # 检查训练是否完成
            
            # Test results HTML visibility is always True, but show message when content is empty
            if not test_results_html and stage == 'Testing':
                test_results_html = """
                <div style="text-align: center; padding: 20px; color: #666;">
                    <p>Testing in progress, please wait for results...</p>
                </div>
                """
            elif not test_results_html:
                test_results_html = """
                <div style="text-align: center; padding: 20px; color: #666;">
                    <p>Test results will be displayed after testing phase completes</p>
                </div>
                """
            
            test_html_update = gr.update(value=test_results_html, visible=True)
            
            # 处理CSV下载按钮
            if test_metrics and len(test_metrics) > 0:
                # 创建临时文件保存CSV内容
                with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv', prefix='metrics_results_') as temp_file:
                    # 写入CSV头部
                    temp_file.write("Metric,Value\n")
                    
                    # 按照优先级排序指标
                    priority_metrics = ['loss', 'accuracy', 'f1', 'precision', 'recall', 'auroc', 'mcc']
                    
                    def get_priority(item):
                        name = item[0]
                        if name in priority_metrics:
                            return priority_metrics.index(name)
                        return len(priority_metrics)
                    
                    # 排序并添加到CSV
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
            
            # 计算进度百分比
            progress_percentage = (current / total) * 100 if total > 0 else 0
            
            # 创建现代化的进度条HTML
            if is_completed:
                # 训练完成状态
                status_html = """
                <div style="background-color: #f8f9fa; border-radius: 10px; padding: 20px; margin-bottom: 15px; box-shadow: 0 2px 5px rgba(0,0,0,0.05);">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 12px;">
                        <div>
                            <span style="font-weight: 600; font-size: 16px;">Training Status: </span>
                            <span style="color: #4caf50; font-weight: 500; font-size: 16px;">Training complete!</span>
                        </div>
                        <div>
                            <span style="font-weight: 600; color: #333;">100%</span>
                        </div>
                    </div>
                    
                    <div style="margin-bottom: 15px; background-color: #e9ecef; height: 10px; border-radius: 5px; overflow: hidden;">
                        <div style="background-color: #4caf50; width: 100%; height: 100%; border-radius: 5px;"></div>
                    </div>
                </div>
                """
            else:
                # 训练或验证阶段
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
            
            # 构建最佳模型信息
            if best_epoch >= 0 and best_metric_value > 0:
                best_info = f"Best model: Epoch {best_epoch} ({best_metric_name}: {best_metric_value:.4f})"
            else:
                best_info = "No best model found yet"
            
            # 获取并更新图表
            loss_fig = monitor.get_loss_plot()
            metrics_fig = monitor.get_metrics_plot()
            
            # 返回更新的组件
            return status_html, best_info, test_html_update, loss_fig, metrics_fig, download_btn_update

        def handle_train(*args) -> Generator:
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
            initial_status_html = """
            <div style="background-color: #f8f9fa; border-radius: 10px; padding: 20px; margin-bottom: 15px; box-shadow: 0 2px 5px rgba(0,0,0,0.05);">
                <div style="display: flex; justify-content: space-between; margin-bottom: 12px;">
                    <div>
                        <span style="font-weight: 600; font-size: 16px;">Training Status: </span>
                        <span style="color: #1976d2; font-weight: 500; font-size: 16px;">Initializing training environment...</span>
                    </div>
                </div>
                <div style="font-size: 14px; color: #555; margin-top: 10px;">
                    <p>• Parsing configuration parameters</p>
                    <p>• Preparing training environment</p>
                    <p>• This may take a few moments...</p>
                </div>
            </div>
            """
            
            # First yield to update UI with "initializing" state
            yield initial_stats, initial_status_html, "Best Model: None", gr.update(value="", visible=False), None, None, gr.update(visible=False)
            
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
                preparing_status_html = """
                <div style="background-color: #f8f9fa; border-radius: 10px; padding: 20px; margin-bottom: 15px; box-shadow: 0 2px 5px rgba(0,0,0,0.05);">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 12px;">
                        <div>
                            <span style="font-weight: 600; font-size: 16px;">Training Status: </span>
                            <span style="color: #1976d2; font-weight: 500; font-size: 16px;">Preparing dataset and model...</span>
                        </div>
                    </div>
                    <div style="font-size: 14px; color: #555; margin-top: 10px;">
                        <p>• Loading dataset</p>
                        <p>• Initializing model architecture</p>
                        <p>• Setting up training environment</p>
                    </div>
                </div>
                """
                yield initial_stats, preparing_status_html, "Best Model: None", gr.update(value="", visible=False), None, None, gr.update(visible=False)
                
                # Save arguments to file
                save_arguments(args_dict, args_dict.get('output_dir', 'ckpt'))
                
                # Start training
                is_training = True
                process_aborted = False  # Reset abort flag
                monitor.start_training(args_dict)
                current_process = monitor.process  # Store the process reference
                
                starting_status_html = """
                <div style="background-color: #f8f9fa; border-radius: 10px; padding: 20px; margin-bottom: 15px; box-shadow: 0 2px 5px rgba(0,0,0,0.05);">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 12px;">
                        <div>
                            <span style="font-weight: 600; font-size: 16px;">Training Status: </span>
                            <span style="color: #1976d2; font-weight: 500; font-size: 16px;">Starting training process...</span>
                        </div>
                    </div>
                    <div style="font-size: 14px; color: #555; margin-top: 10px;">
                        <p>• Training process launched</p>
                        <p>• Waiting for first statistics to appear</p>
                        <p>• This may take a moment for large models</p>
                    </div>
                </div>
                """
                
                yield initial_stats, starting_status_html, "Best Model: None", gr.update(value="", visible=False), None, None, gr.update(visible=False)

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
                    if process_aborted or not monitor.is_training or current_process is None or (current_process and current_process.poll() is not None):
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
                                
                                error_status_html = f"""
                                <div style="padding: 10px; background-color: #ffebee; border-radius: 5px; margin-bottom: 10px;">
                                    <p style="margin: 0; color: #c62828; font-weight: bold;">Training failed with error code {return_code}:</p>
                                    <pre style="margin: 5px 0 0; white-space: pre-wrap; max-height: 300px; overflow-y: auto; background-color: #f5f5f5; padding: 10px; border-radius: 4px; font-family: monospace;">{error_output}</pre>
                                </div>
                                """
                                yield (
                                    initial_stats,
                                    error_status_html,
                                    "Training failed",
                                    gr.update(value="", visible=False),
                                    None,
                                    None,
                                    gr.update(visible=False)
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
                        
                        yield model_stats, status_html, best_info, test_html_update, loss_fig, metrics_fig, download_btn_update
                        
                    except Exception as e:
                        # Get complete output log
                        error_output = "\n".join(progress_info.get("lines", []))
                        if not error_output:
                            error_output = "No output captured from the training process"
                        
                        error_status_html = f"""
                        <div style="padding: 10px; background-color: #ffebee; border-radius: 5px; margin-bottom: 10px;">
                            <p style="margin: 0; color: #c62828; font-weight: bold;">Error during training:</p>
                            <p style="margin: 5px 0; color: #c62828;">{str(e)}</p>
                            <pre style="margin: 5px 0 0; white-space: pre-wrap; max-height: 300px; overflow-y: auto; background-color: #f5f5f5; padding: 10px; border-radius: 4px; font-family: monospace;">{error_output}</pre>
                        </div>
                        """
                        print(f"Error updating UI: {str(e)}")
                        traceback.print_exc()
                        yield initial_stats, error_status_html, "Training error", gr.update(value="", visible=False), None, None, gr.update(visible=False)
                        return
                
                # Check if aborted
                if process_aborted:
                    is_training = False
                    current_process = None
                    aborted_status_html = """
                    <div style="padding: 10px; background-color: #e8f5e9; border-radius: 5px;">
                        <p style="margin: 0; color: #2e7d32; font-weight: bold;">Training was manually terminated.</p>
                    </div>
                    """
                    yield initial_stats, aborted_status_html, "Training aborted", gr.update(value="", visible=False), None, None, gr.update(visible=False)
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
                        
                        error_status_html = f"""
                        <div style="padding: 10px; background-color: #ffebee; border-radius: 5px; margin-bottom: 10px;">
                            <p style="margin: 0; color: #c62828; font-weight: bold;">Error in final update:</p>
                            <p style="margin: 5px 0; color: #c62828;">{str(e)}</p>
                            <pre style="margin: 5px 0 0; white-space: pre-wrap; max-height: 300px; overflow-y: auto; background-color: #f5f5f5; padding: 10px; border-radius: 4px; font-family: monospace;">{error_output}</pre>
                        </div>
                        """
                        yield initial_stats, error_status_html, "Error in final update", gr.update(value="", visible=False), None, None, gr.update(visible=False)
                
            except Exception as e:
                # Initialization error, may not have output log
                error_status_html = f"""
                <div style="padding: 10px; background-color: #ffebee; border-radius: 5px; margin-bottom: 10px;">
                    <p style="margin: 0; color: #c62828; font-weight: bold;">Training initialization failed:</p>
                    <p style="margin: 5px 0; color: #c62828;">{str(e)}</p>
                </div>
                """
                yield initial_stats, error_status_html, "Training failed", gr.update(value="", visible=False), None, None, gr.update(visible=False)
            finally:
                is_training = False
                current_process = None
        
        def handle_abort():
            """Handle abortion of the training process"""
            nonlocal is_training, current_process, stop_thread, process_aborted
            
            if not is_training or current_process is None:
                return (gr.HTML("""
                <div style="padding: 10px; background-color: #f5f5f5; border-radius: 5px;">
                    <p style="margin: 0;">No training process is currently running.</p>
                </div>
                """),
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
                
                success_html = """
                <div style="padding: 10px; background-color: #e8f5e9; border-radius: 5px;">
                    <p style="margin: 0; color: #2e7d32; font-weight: bold;">Training successfully terminated!</p>
                    <p style="margin: 5px 0 0; color: #388e3c;">All training state has been reset. You can start a new training session.</p>
                </div>
                """
                
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
                
                error_html = f"""
                <div style="padding: 10px; background-color: #ffebee; border-radius: 5px;">
                    <p style="margin: 0; color: #c62828; font-weight: bold;">Failed to terminate training: {str(e)}</p>
                    <p style="margin: 5px 0 0; color: #c62828;">Training state has been reset.</p>
                </div>
                """
                
                # Return updates for all relevant components including empty model stats
                return (gr.HTML(error_html),
                      empty_model_stats,
                      "Best Model: None",
                      gr.update(value="", visible=False),
                      None,
                      None,
                      gr.update(visible=False))

        def update_wandb_visibility(checkbox):
            return {
                wandb_project: gr.update(visible=checkbox),
                wandb_entity: gr.update(visible=checkbox)
            }

        # define all input components
        input_components = [
            plm_model, #0
            is_custom_dataset, #1
            dataset_config, #2
            dataset_custom, #3
            problem_type, #4
            num_labels, #5
            metrics, #6
            training_method, #7
            pooling_method, #8
            batch_mode, #9
            batch_size, #10
            batch_token, #11
            learning_rate, #12
            num_epochs, #13
            max_seq_len, #14
            gradient_accumulation_steps, #15
            warmup_steps, #16
            scheduler_type, #17
            output_model_name, #18
            output_dir, #19
            wandb_logging, #20
            wandb_project, #21
            wandb_entity, #22
            patience, #23
            num_workers, #24
            max_grad_norm, #25
            structure_seq, #26
            lora_r, #27
            lora_alpha, #28
            lora_dropout, #29
            lora_target_modules, #30
            monitored_metrics, #31
            monitored_strategy, #32
        ]

        # bind preview and train buttons
        def handle_preview(*args):
            if command_preview.visible:
                return gr.update(visible=False)
            training_args = TrainingArgs(args, plm_models, dataset_configs)
            preview_text = preview_command(training_args.to_dict())
            return gr.update(value=preview_text, visible=True)

        def reset_train_ui():
            """Reset the UI state before training starts"""
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
            
            empty_progress_status = """
            <div style="background-color: #f8f9fa; border-radius: 10px; padding: 20px; margin-bottom: 15px; box-shadow: 0 2px 5px rgba(0,0,0,0.05);">
                <div style="display: flex; justify-content: space-between; margin-bottom: 12px;">
                    <div>
                        <span style="font-weight: 600; font-size: 16px;">Training Status: </span>
                        <span style="color: #1976d2; font-weight: 500; font-size: 16px;">Preparing to start training...</span>
                    </div>
                </div>
            </div>
            """
            
            # Return exactly 7 values matching the 7 output components
            return (
                empty_model_stats, 
                empty_progress_status,
                "Best Model: None",
                gr.update(value="", visible=False),
                None,  # loss_plot must be None, not a string
                None,  # metrics_plot must be None, not a string
                gr.update(visible=False)
            )

        preview_button.click(
            fn=handle_preview,
            inputs=input_components,
            outputs=[command_preview]
        )
        
        train_button.click(
            fn=reset_train_ui,
            outputs=[model_stats, progress_status, best_model_info, test_results_html, loss_plot, metrics_plot, download_csv_btn]
        ).then(
            fn=handle_train, 
            inputs=input_components,
            outputs=[model_stats, progress_status, best_model_info, test_results_html, loss_plot, metrics_plot, download_csv_btn]
        )

        # bind abort button
        abort_button.click(
            fn=handle_abort,
            outputs=[progress_status, model_stats, best_model_info, test_results_html, loss_plot, metrics_plot, download_csv_btn]
        )
        
        wandb_logging.change(
            fn=update_wandb_visibility,
            inputs=[wandb_logging],
            outputs=[wandb_project, wandb_entity]
        )

        def update_dataset_preview(dataset_type=None, dataset_name=None, custom_dataset=None):
            """Update dataset preview content"""
            # Determine which dataset to use based on selection
            if dataset_type == "Use Custom Dataset" and custom_dataset:
                try:
                    # Try to load custom dataset
                    dataset = load_dataset(custom_dataset)
                    stats_html = f"""
                    <div style="text-align: center; margin: 20px 0;">
                        <table style="width: 100%; border-collapse: collapse; margin: 0 auto;">
                            <tr>
                                <th style="padding: 8px; font-size: 14px; border: 1px solid #ddd; background-color: #e0e0e0; font-weight: bold; border-bottom: 1px solid #ccc; text-align: center;">Dataset</th>
                                <th style="padding: 8px; font-size: 14px; border: 1px solid #ddd; background-color: #e0e0e0; font-weight: bold; border-bottom: 1px solid #ccc; text-align: center;">Train Samples</th>
                                <th style="padding: 8px; font-size: 14px; border: 1px solid #ddd; background-color: #e0e0e0; font-weight: bold; border-bottom: 1px solid #ccc; text-align: center;">Val Samples</th>
                                <th style="padding: 8px; font-size: 14px; border: 1px solid #ddd; background-color: #e0e0e0; font-weight: bold; border-bottom: 1px solid #ccc; text-align: center;">Test Samples</th>
                            </tr>
                            <tr>
                                <td style="padding: 15px; font-size: 14px; border: 1px solid #ddd; text-align: center;">{custom_dataset}</td>
                                <td style="padding: 15px; font-size: 14px; border: 1px solid #ddd; text-align: center;">{len(dataset["train"]) if "train" in dataset else 0}</td>
                                <td style="padding: 15px; font-size: 14px; border: 1px solid #ddd; text-align: center;">{len(dataset["validation"]) if "validation" in dataset else 0}</td>
                                <td style="padding: 15px; font-size: 14px; border: 1px solid #ddd; text-align: center;">{len(dataset["test"]) if "test" in dataset else 0}</td>
                            </tr>
                        </table>
                    </div>
                    """
                    
                    # Get sample data points
                    split = "train" if "train" in dataset else list(dataset.keys())[0]
                    samples = dataset[split].select(range(min(3, len(dataset[split]))))
                    if len(samples) == 0:
                        return gr.update(value=stats_html), gr.update(value=[["No data available", "-", "-"]], headers=["Name", "Sequence", "Label"]), gr.update(open=True)
                    
                    # Get fields actually present in the dataset
                    available_fields = list(samples[0].keys())
                    
                    # Build sample data
                    sample_data = []
                    for sample in samples:
                        sample_dict = {}
                        for field in available_fields:
                            # Keep full sequence
                            sample_dict[field] = str(sample[field])
                        sample_data.append(sample_dict)
                    
                    df = pd.DataFrame(sample_data)
                    return gr.update(value=stats_html), gr.update(value=df.values.tolist(), headers=df.columns.tolist()), gr.update(open=True)
                except Exception as e:
                    error_html = f"""
                    <div>
                        <h2>Error loading dataset</h2>
                        <p style="color: #c62828;">{str(e)}</p>
                    </div>
                    """
                    return gr.update(value=error_html), gr.update(value=[["Error", str(e), "-"]], headers=["Name", "Sequence", "Label"]), gr.update(open=True)
            
            # Use predefined dataset
            elif dataset_type == "Use Pre-defined Dataset" and dataset_name:
                try:
                    config_path = dataset_configs[dataset_name]
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                    
                    # Load dataset statistics
                    dataset = load_dataset(config["dataset"])
                    stats_html = f"""
                    <div style="text-align: center; margin: 20px 0;">
                        <table style="width: 100%; border-collapse: collapse; margin: 0 auto;">
                            <tr>
                                <th style="padding: 8px; font-size: 14px; border: 1px solid #ddd; background-color: #e0e0e0; font-weight: bold; border-bottom: 1px solid #ccc; text-align: center;">Dataset</th>
                                <th style="padding: 8px; font-size: 14px; border: 1px solid #ddd; background-color: #e0e0e0; font-weight: bold; border-bottom: 1px solid #ccc; text-align: center;">Train Samples</th>
                                <th style="padding: 8px; font-size: 14px; border: 1px solid #ddd; background-color: #e0e0e0; font-weight: bold; border-bottom: 1px solid #ccc; text-align: center;">Val Samples</th>
                                <th style="padding: 8px; font-size: 14px; border: 1px solid #ddd; background-color: #e0e0e0; font-weight: bold; border-bottom: 1px solid #ccc; text-align: center;">Test Samples</th>
                            </tr>
                            <tr>
                                <td style="padding: 15px; font-size: 14px; border: 1px solid #ddd; text-align: center;">{config["dataset"]}</td>
                                <td style="padding: 15px; font-size: 14px; border: 1px solid #ddd; text-align: center;">{len(dataset["train"]) if "train" in dataset else 0}</td>
                                <td style="padding: 15px; font-size: 14px; border: 1px solid #ddd; text-align: center;">{len(dataset["validation"]) if "validation" in dataset else 0}</td>
                                <td style="padding: 15px; font-size: 14px; border: 1px solid #ddd; text-align: center;">{len(dataset["test"]) if "test" in dataset else 0}</td>
                            </tr>
                        </table>
                    </div>
                    """
                    
                    # Get sample data points and available fields
                    samples = dataset["train"].select(range(min(3, len(dataset["train"]))))
                    if len(samples) == 0:
                        return gr.update(value=stats_html), gr.update(value=[["No data available", "-", "-"]], headers=["Name", "Sequence", "Label"]), gr.update(open=True)
                    
                    # Get fields actually present in the dataset
                    available_fields = list(samples[0].keys())
                    
                    # Build sample data
                    sample_data = []
                    for sample in samples:
                        sample_dict = {}
                        for field in available_fields:
                            # Keep full sequence
                            sample_dict[field] = str(sample[field])
                        sample_data.append(sample_dict)
                    
                    df = pd.DataFrame(sample_data)
                    return gr.update(value=stats_html), gr.update(value=df.values.tolist(), headers=df.columns.tolist()), gr.update(open=True)
                except Exception as e:
                    error_html = f"""
                    <div>
                        <h2>Error loading dataset</h2>
                        <p style="color: #c62828;">{str(e)}</p>
                    </div>
                    """
                    return gr.update(value=error_html), gr.update(value=[["Error", str(e), "-"]], headers=["Name", "Sequence", "Label"]), gr.update(open=True)
            
            # If no valid dataset information provided
            return gr.update(value=""), gr.update(value=[["No dataset selected", "-", "-"]], headers=["Name", "Sequence", "Label"]), gr.update(open=True)

        # Preview button click event
        dataset_preview_button.click(
            fn=update_dataset_preview,
            inputs=[is_custom_dataset, dataset_config, dataset_custom],
            outputs=[dataset_stats_md, preview_table, preview_accordion]
        )

        # 添加自定义数据集设置的函数
        def update_dataset_settings(choice, dataset_name=None):
            if choice == "Use Pre-defined Dataset":
                # 从dataset_config加载配置
                result = {
                    dataset_config: gr.update(visible=True),
                    dataset_custom: gr.update(visible=False),
                    custom_dataset_settings: gr.update(visible=True)
                }
                
                # 如果有选择特定数据集，自动加载配置
                if dataset_name and dataset_name in dataset_configs:
                    with open(dataset_configs[dataset_name], 'r') as f:
                        config = json.load(f)
                    
                    # 处理metrics，将字符串转换为列表以适应多选组件
                    metrics_value = config.get("metrics", "accuracy,mcc,f1,precision,recall,auroc")
                    if isinstance(metrics_value, str):
                        metrics_value = metrics_value.split(",")
                    
                    # 处理monitored_metrics，单选
                    monitored_metrics_value = config.get("monitor", "accuracy")
                    monitored_strategy_value = config.get("monitor_strategy", "max")
                    result.update({
                        problem_type: gr.update(value=config.get("problem_type", "single_label_classification"), interactive=False),
                        num_labels: gr.update(value=config.get("num_labels", 2), interactive=False),
                        metrics: gr.update(value=metrics_value, interactive=False),
                        monitored_metrics: gr.update(value=monitored_metrics_value, interactive=False),
                        monitored_strategy: gr.update(value=monitored_strategy_value, interactive=False)
                    })
                return result
            else:
                # 自定义数据集设置，清零/设为默认值并可编辑
                # 为多选组件提供默认值列表
                default_metrics = ["accuracy", "mcc", "f1", "precision", "recall", "auroc"]
                default_monitored_metrics = ["accuracy"]
                default_monitored_strategy = ["max"]
                return {
                    dataset_config: gr.update(visible=False),
                    dataset_custom: gr.update(visible=True),
                    custom_dataset_settings: gr.update(visible=True),
                    problem_type: gr.update(value="single_label_classification", interactive=True),
                    num_labels: gr.update(value=2, interactive=True),
                    metrics: gr.update(value=default_metrics, interactive=True),
                    monitored_metrics: gr.update(value=default_monitored_metrics, interactive=True),
                    monitored_strategy: gr.update(value=default_monitored_strategy, interactive=True)
                }

        # 绑定数据集设置更新事件
        is_custom_dataset.change(
            fn=update_dataset_settings,
            inputs=[is_custom_dataset, dataset_config],
            outputs=[dataset_config, dataset_custom, custom_dataset_settings, problem_type, num_labels, metrics, monitored_metrics, monitored_strategy]
        )

        dataset_config.change(
            fn=lambda x: update_dataset_settings("Use Pre-defined Dataset", x),
            inputs=[dataset_config],
            outputs=[dataset_config, dataset_custom, custom_dataset_settings, problem_type, num_labels, metrics, monitored_metrics, monitored_strategy]
        )

        # Return components that need to be accessed from outside
        return {
            "output_text": progress_status,
            "loss_plot": loss_plot,
            "metrics_plot": metrics_plot,
            "train_button": train_button,
            "monitor": monitor,
            "test_results_html": test_results_html,  # 添加测试结果HTML组件
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
                "lora_r": lora_r,
                "lora_alpha": lora_alpha,
                "lora_dropout": lora_dropout,
                "lora_target_modules": lora_target_modules,
            }
        }