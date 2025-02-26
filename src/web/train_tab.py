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
            "structure_seq": self.structure_seq
        }

        # 添加数据集相关参数
        if self.dataset_selection == "Use Pre-defined Dataset":
            args_dict["dataset_config"] = self.dataset_config
        else:
            args_dict["dataset"] = self.dataset_custom
            args_dict["problem_type"] = self.problem_type
            args_dict["num_labels"] = self.num_labels
            args_dict["metrics"] = self.metrics

        # Add LoRA parameters
        if self.training_method == "plm-lora":
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
    
    plm_models = constant["plm_models"]
    dataset_configs = constant["dataset_configs"]

    with gr.Tab("Training"):
        # Model and Dataset Selection
        gr.Markdown("### Model and Dataset Configuration")
        
        with gr.Tabs():
            # Dataset Preview Tab
            with gr.Tab("Dataset Preview"):
                with gr.Row():
                    preview_dataset_type = gr.Radio(
                        choices=["Use Pre-defined Dataset", "Use Custom Dataset"],
                        label="Dataset Type",
                        value="Use Pre-defined Dataset"
                    )
                
                with gr.Row():
                    preview_dataset = gr.Dropdown(
                        choices=list(dataset_configs.keys()),
                        label="Select Dataset to Preview",
                        value=None,  # Default not selected
                        allow_custom_value=False,
                        visible=True
                    )
                
                # 添加自定义数据集预览输入
                preview_custom_dataset = gr.Textbox(
                    label="Preview Custom Dataset",
                    placeholder="Huggingface Dataset eg: user/dataset",
                    interactive=True,
                    visible=False
                )
                
                dataset_preview_button = gr.Button("Preview Dataset", variant="primary")
                
                with gr.Row():
                    dataset_stats_md = gr.Markdown("")  # Initial empty
                
                with gr.Row():
                    preview_table = gr.Dataframe(
                        headers=["Name", "Sequence", "Label"],  # More friendly default header
                        value=[["No dataset selected", "-", "-"]],  # More friendly default value
                        label="Sample Data Points",
                        wrap=True,
                        interactive=False,  # Only allow copy, not edit
                        row_count=3,
                        elem_classes=["preview-table"]  # 添加CSS类以便应用样式
                    )

                # 添加CSS样式
                gr.HTML("""
                <style>
                    /* 只对表格内容应用灰色背景，不影响标签 */
                    .preview-table table {
                        background-color: #f5f5f5 !important;
                    }
                    .preview-table .gr-block.gr-box {
                        background-color: transparent !important;
                    }
                    .preview-table .gr-input-label {
                        background-color: transparent !important;
                    }
                    /* 表格外观美化 */
                    .preview-table table {
                        margin-top: 10px;
                        border-radius: 8px;
                        overflow: hidden;
                    }
                    /* 强化表头样式 */
                    .preview-table th {
                        background-color: #e0e0e0 !important;
                        font-weight: bold !important;
                        padding: 8px !important;
                        border-bottom: 1px solid #ccc !important;
                    }
                </style>
                """, visible=True)

        # Original training interface components
        with gr.Group():
            with gr.Row():
                with gr.Column():
                    plm_model = gr.Dropdown(
                        choices=list(plm_models.keys()),
                        label="Protein Language Model",
                        value=list(plm_models.keys())[0]
                    )
                
                with gr.Column():
                    # 新增数据集选择方式
                    is_custom_dataset = gr.Radio(
                        choices=["Use Custom Dataset", "Use Pre-defined Dataset"],
                        label="Dataset Selection",
                        value="Use Pre-defined Dataset"
                    )
            
                with gr.Column():
                    dataset_config = gr.Dropdown(
                        choices=list(dataset_configs.keys()),
                        label="Dataset Configuration",
                        value=list(dataset_configs.keys())[0],
                        visible=True
                    )
                    
                    dataset_custom = gr.Textbox(
                        label="Custom Dataset Path",
                        placeholder="Huggingface Dataset eg: user/dataset",
                        visible=False
                    )
            
            # 自定义数据集的额外配置选项（单独一行）
            with gr.Row(visible=True) as custom_dataset_settings:
                with gr.Column(scale=1):
                    problem_type = gr.Dropdown(
                        choices=["single_label_classification", "multi_label_classification", "regression"],
                        label="Problem Type",
                        value="single_label_classification"
                    )
                with gr.Column(scale=1):
                    num_labels = gr.Number(
                        value=2,
                        label="Number of Labels"
                    )
                with gr.Column(scale=1):
                    metrics = gr.Textbox(
                        label="Metrics",
                        placeholder="accuracy,recall,precision,f1,mcc,auroc,f1max,spearman_corr,mse",
                        value="accuracy,mcc,f1,precision,recall,auroc"
                    )
                
            with gr.Row():
                structure_seq = gr.Textbox(
                    label="Structure Sequence", 
                    placeholder="foldseek_seq,ss8_seq", 
                    value="foldseek_seq,ss8_seq",
                    visible=False
                )

        # ! add for plm-lora
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
                        choices=["full", "freeze", "ses-adapter", "plm-lora"],
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
                        minimum=1, maximum=200, value=100, step=1,
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
                    lora_params_row: gr.update(visible=method == "plm-lora")
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
                    value="ckpt",
                    placeholder="Path to save training results"
                )
                
                output_model_name = gr.Textbox(
                    label="Output Model Name",
                    value="model.pt",
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
                    ["Adapter Model", "-", "-", "-"],
                    ["Pre-trained Model", "-", "-", "-"],
                    ["Combined Model", "-", "-", "-"]
                ],
                interactive=False
            )

        def update_model_stats(stats: Dict[str, str]) -> List[List[str]]:
            """Update model statistics in table format."""
            if not stats:
                return [
                    ["Adapter Model", "-", "-", "-"],
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
                ["Adapter Model", str(adapter_total), str(adapter_trainable), "-"],
                ["Pre-trained Model", str(pretrain_total), str(pretrain_trainable), "-"],
                ["Combined Model", str(combined_total), str(combined_trainable), str(trainable_percentage)]
            ]

        # Training Progress
        gr.Markdown("### Training Progress")
        with gr.Row():
            progress_status = gr.Textbox(
                label="Status",
                value="Waiting to start...",
                interactive=False
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

        # Training plot in a separate row for full width
        with gr.Row():
            plot_output = gr.Plot(
                label="Training Progress",
                elem_id="training-plot"
            )

        def update_progress(progress_info):
            if not progress_info:
                return (
                    "Waiting to start...",
                    "Best Model: None",
                    gr.update(value="", visible=True)
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
            
            # Build progress string
            if stage == 'Waiting':
                status = "Waiting to start training..."
            elif stage == 'Testing':  # 特别处理测试阶段
                status = "Testing Phase\n"
                
                # Add current stage information with progress bar
                progress_percentage = (current / total) * 100 if total > 0 else 0
                progress_bar = "█" * int(progress_percentage // 5) + "░" * (20 - int(progress_percentage // 5))
                
                status += f"Progress: {progress_percentage:.1f}% |{progress_bar}| {current}/{total}\n"
                
                # Add time information if available
                if elapsed_time and remaining_time:
                    status += f"Time: [{elapsed_time}<{remaining_time}, {it_per_sec:.2f}it/s]\n"
                
                # Add reference to best model
                if best_epoch > 0:
                    status += f"Using best model from epoch {best_epoch}\n"
                
                # Add test metrics if available
                test_metrics = progress_info.get('test_metrics', {})
                if test_metrics:
                    status += "\nTest Metrics:\n"
                    for metric_name, metric_value in sorted(test_metrics.items()):
                        status += f"- {metric_name}: {metric_value:.4f}\n"
            else:
                # Format current epoch and stage for training/validation
                # 使用从progress_info获取的total_epochs，而不是num_epochs组件
                epoch_total = total_epochs if total_epochs > 0 else (num_epochs.value if hasattr(num_epochs, 'value') else 100)
                status = f"Epoch: {epoch}/{epoch_total}\n"
                
                # Add current stage information
                progress_percentage = (current / total) * 100 if total > 0 else 0
                
                # Create progress bar style
                progress_bar = "█" * int(progress_percentage // 5) + "░" * (20 - int(progress_percentage // 5))
                
                # Add formatted progress bar
                status += f"Stage: {stage}\n"
                status += f"Progress: {progress_percentage:.1f}% |{progress_bar}| {current}/{total}\n"
                
                # Add time and speed information
                if elapsed_time and remaining_time:
                    status += f"Time: [{elapsed_time}<{remaining_time}, {it_per_sec:.2f}it/s]\n"
                
                # Add training related information
                if stage == 'Training' and grad_step > 0:
                    status += f"Gradient updates: {grad_step}, Loss: {loss:.4f}\n"
            
            # Build best model information
            if best_epoch > 0:
                best_info = f"Best model: Epoch {best_epoch} ({best_metric_name}: {best_metric_value:.4f})"
            else:
                best_info = "No best model found yet"
            
            return status, best_info, test_html_update

        def handle_train(*args) -> Generator:
            if monitor.is_training:
                yield None, None, None, None, None
                return
            
            # Initialize table state
            initial_stats = [
                ["Adapter Model", "-", "-", "-"],
                ["Pre-trained Model", "-", "-", "-"],
                ["Combined Model", "-", "-", "-"]
            ]
            
            try:
                training_args = TrainingArgs(args, plm_models, dataset_configs)
                args_dict = training_args.to_dict()
                
                # 保存总epoch数到monitor中，以便在progress_info中使用
                total_epochs = args_dict.get('num_epochs', 100)
                monitor.current_progress['total_epochs'] = total_epochs
                
                # Save arguments to file
                save_arguments(args_dict, args_dict.get('output_dir', 'ckpt'))
                
                # Ensure the monitor stats are reset
                monitor._reset_stats()
                
                # Start training
                monitor.start_training(args_dict)
                
                yield None, initial_stats, "Waiting to start...", "Best Model: None", gr.update(value="", visible=False)

                # Add delay to ensure enough time for parsing initial statistics
                for i in range(3):
                    time.sleep(1)
                    # Check if statistics are already available
                    stats = monitor.get_stats()
                    if stats and len(stats) > 0:
                        break
                
                update_count = 0
                while monitor.is_training:
                    try:
                        update_count += 1
                        
                        # Get latest statistics
                        stats = monitor.get_stats()
                        
                        # Get plot data and log
                        plot = monitor.get_plot()
                        progress_info = monitor.get_progress()
                        # Update progress display (no longer updating progress bar)
                        status, best_info, test_html_update = update_progress(progress_info)
                        
                        # Update model statistics table
                        model_stats_update = update_model_stats(stats)
                        
                        yield (plot, 
                              model_stats_update,
                              status,
                              best_info,
                              test_html_update)
                        
                        # Add appropriate delay to avoid too frequent updates
                        time.sleep(1)
                               
                    except Exception as e:
                        yield None, initial_stats, f"Error: {str(e)}", "Error occurred", gr.update(value="", visible=False)
                        time.sleep(5)  # Extend wait time when error occurs
                
                # Final update after training ends
                try:
                    final_stats = monitor.get_stats()
                    final_plot = monitor.get_plot()
                    final_progress = monitor.get_progress()
                    
                    final_status, final_best_info, final_test_html = update_progress(final_progress)
                    final_model_stats = update_model_stats(final_stats)
                    
                    yield (final_plot, 
                          final_model_stats,
                          final_status,
                          final_best_info,
                          final_test_html)
                except Exception as e:
                    yield None, initial_stats, f"Training completed with error: {str(e)}", "See log for details", gr.update(value="", visible=False)
            except Exception as e:
                yield None, initial_stats, f"Error occurred: {str(e)}", "Training failed", gr.update(value="", visible=False)

        def handle_abort():
            monitor.abort_training()
            
            # Reset table to initial state
            initial_stats = [
                ["Adapter Model", "-", "-", "-"],
                ["Pre-trained Model", "-", "-", "-"],
                ["Combined Model", "-", "-", "-"]
            ]
            
            return (
                "Training aborted!",  # status
                "Training aborted!",  # best model info
                initial_stats         # Reset Model Statistics table
            )

        def update_wandb_visibility(checkbox):
            return {
                wandb_project: gr.update(visible=checkbox),
                wandb_entity: gr.update(visible=checkbox)
            }

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
            lora_r,
            lora_alpha,
            lora_dropout,
            lora_target_modules,
        ]

        # bind preview and train buttons
        def handle_preview(*args):
            if command_preview.visible:
                return gr.update(visible=False)
            training_args = TrainingArgs(args, plm_models, dataset_configs)
            preview_text = preview_command(training_args.to_dict())
            return gr.update(value=preview_text, visible=True)

        preview_button.click(
            fn=handle_preview,
            inputs=input_components,
            outputs=[command_preview]
        )
        
        train_button.click(
            fn=handle_train,
            inputs=input_components,
            outputs=[plot_output, model_stats, progress_status, best_model_info, test_results_html]
        )

        # bind abort button
        abort_button.click(
            fn=handle_abort,
            outputs=[progress_status, best_model_info, model_stats]
        )
        
        wandb_logging.change(
            fn=update_wandb_visibility,
            inputs=[wandb_logging],
            outputs=[wandb_project, wandb_entity]
        )

        def update_dataset_preview(dataset_type=None, dataset_name=None, custom_dataset=None):
            """更新数据集预览内容"""
            # 根据数据集类型选择确定使用哪种数据集
            if dataset_type == "Use Custom Dataset" and custom_dataset:
                try:
                    # 尝试加载自定义数据集
                    dataset = load_dataset(custom_dataset)
                    stats_html = f"""
                    <div style="text-align: center; margin: 20px 0;">
                        <h2 style="font-size: 24px; margin-bottom: 20px;">{custom_dataset}</h2>
                        <table style="width: 100%; border-collapse: collapse; margin: 0 auto;">
                            <tr>
                                <th style="padding: 8px; font-size: 18px; border: 1px solid #ddd; background-color: #e0e0e0; font-weight: bold; border-bottom: 1px solid #ccc;">Train Samples</th>
                                <th style="padding: 8px; font-size: 18px; border: 1px solid #ddd; background-color: #e0e0e0; font-weight: bold; border-bottom: 1px solid #ccc;">Val Samples</th>
                                <th style="padding: 8px; font-size: 18px; border: 1px solid #ddd; background-color: #e0e0e0; font-weight: bold; border-bottom: 1px solid #ccc;">Test Samples</th>
                            </tr>
                            <tr>
                                <td style="padding: 15px; font-size: 18px; border: 1px solid #ddd; text-align: center;">{len(dataset["train"]) if "train" in dataset else 0}</td>
                                <td style="padding: 15px; font-size: 18px; border: 1px solid #ddd; text-align: center;">{len(dataset["validation"]) if "validation" in dataset else 0}</td>
                                <td style="padding: 15px; font-size: 18px; border: 1px solid #ddd; text-align: center;">{len(dataset["test"]) if "test" in dataset else 0}</td>
                            </tr>
                        </table>
                    </div>
                    """
                    
                    # 获取样本数据点
                    split = "train" if "train" in dataset else list(dataset.keys())[0]
                    samples = dataset[split].select(range(min(3, len(dataset[split]))))
                    if len(samples) == 0:
                        return "Dataset is empty", gr.update(value=[["No data available", "-", "-"]], headers=["Name", "Sequence", "Label"])
                    
                    # 获取数据集中实际存在的字段
                    available_fields = list(samples[0].keys())
                    
                    # 构建样本数据
                    sample_data = []
                    for sample in samples:
                        sample_dict = {}
                        for field in available_fields:
                            # 保留完整序列
                            sample_dict[field] = str(sample[field])
                        sample_data.append(sample_dict)
                    
                    df = pd.DataFrame(sample_data)
                    return stats_html, gr.update(value=df.values.tolist(), headers=df.columns.tolist())
                except Exception as e:
                    return f"Error loading custom dataset: {str(e)}", gr.update(value=[["Error", str(e), "-"]], headers=["Name", "Sequence", "Label"])
            
            # 使用预定义数据集或者未指定数据集类型
            elif dataset_type == "Use Pre-defined Dataset" and dataset_name:
                try:
                    # 处理预设数据集
                    config_path = dataset_configs[dataset_name]
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                    
                    # 加载数据集统计信息
                    dataset = load_dataset(config["dataset"])
                    stats_html = f"""
                    <div style="text-align: center; margin: 20px 0;">
                        <h2 style="font-size: 24px; margin-bottom: 20px;">{config["dataset"]}</h2>
                        <table style="width: 100%; border-collapse: collapse; margin: 0 auto;">
                            <tr>
                                <th style="padding: 8px; font-size: 18px; border: 1px solid #ddd; background-color: #e0e0e0; font-weight: bold; border-bottom: 1px solid #ccc;">Train Samples</th>
                                <th style="padding: 8px; font-size: 18px; border: 1px solid #ddd; background-color: #e0e0e0; font-weight: bold; border-bottom: 1px solid #ccc;">Val Samples</th>
                                <th style="padding: 8px; font-size: 18px; border: 1px solid #ddd; background-color: #e0e0e0; font-weight: bold; border-bottom: 1px solid #ccc;">Test Samples</th>
                            </tr>
                            <tr>
                                <td style="padding: 15px; font-size: 18px; border: 1px solid #ddd; text-align: center;">{len(dataset["train"])}</td>
                                <td style="padding: 15px; font-size: 18px; border: 1px solid #ddd; text-align: center;">{len(dataset["validation"])}</td>
                                <td style="padding: 15px; font-size: 18px; border: 1px solid #ddd; text-align: center;">{len(dataset["test"])}</td>
                            </tr>
                        </table>
                    </div>
                    """
                    
                    # 获取样本数据点和可用字段
                    samples = dataset["train"].select(range(min(3, len(dataset["train"]))))
                    if len(samples) == 0:
                        return "Dataset is empty", gr.update(value=[["No data available", "-", "-"]], headers=["Name", "Sequence", "Label"])
                    
                    # 获取数据集中实际存在的字段
                    available_fields = list(samples[0].keys())
                    
                    # 构建样本数据
                    sample_data = []
                    for sample in samples:
                        sample_dict = {}
                        for field in available_fields:
                            # 保留完整序列
                            sample_dict[field] = str(sample[field])
                        sample_data.append(sample_dict)
                    
                    df = pd.DataFrame(sample_data)
                    return stats_html, gr.update(value=df.values.tolist(), headers=df.columns.tolist())
                except Exception as e:
                    return f"Error loading dataset: {str(e)}", gr.update(value=[["Error", str(e), "-"]], headers=["Name", "Sequence", "Label"])
            
            # 如果未提供有效的数据集信息
            return "", gr.update(value=[["No dataset selected", "-", "-"]], headers=["Name", "Sequence", "Label"])

        # Auto-update when dataset is selected
        preview_dataset.change(
            fn=lambda x: update_dataset_preview(dataset_type="Use Pre-defined Dataset", dataset_name=x),
            inputs=[preview_dataset],
            outputs=[dataset_stats_md, preview_table]
        )
        
        # 添加预览按钮点击事件
        dataset_preview_button.click(
            fn=update_dataset_preview,
            inputs=[preview_dataset_type, preview_dataset, preview_custom_dataset],
            outputs=[dataset_stats_md, preview_table]
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
                    result.update({
                        problem_type: gr.update(value=config.get("problem_type", "single_label_classification"), interactive=False),
                        num_labels: gr.update(value=config.get("num_labels", 2), interactive=False),
                        metrics: gr.update(value=config.get("metrics", "accuracy,mcc,f1,precision,recall,auroc"), interactive=False),
                    })
                return result
            else:
                # 自定义数据集设置，清零/设为默认值并可编辑
                return {
                    dataset_config: gr.update(visible=False),
                    dataset_custom: gr.update(visible=True),
                    custom_dataset_settings: gr.update(visible=True),
                    problem_type: gr.update(value="single_label_classification", interactive=True),
                    num_labels: gr.update(value=2, interactive=True),
                    metrics: gr.update(value="accuracy,mcc,f1,precision,recall,auroc", interactive=True)
                }

        # 绑定数据集设置更新事件
        is_custom_dataset.change(
            fn=update_dataset_settings,
            inputs=[is_custom_dataset, dataset_config],
            outputs=[dataset_config, dataset_custom, custom_dataset_settings, problem_type, num_labels, metrics]
        )

        dataset_config.change(
            fn=lambda x: update_dataset_settings("Use Pre-defined Dataset", x),
            inputs=[dataset_config],
            outputs=[dataset_config, dataset_custom, custom_dataset_settings, problem_type, num_labels, metrics]
        )

        # 添加数据集类型切换的逻辑
        def update_preview_inputs(choice):
            """更新预览数据集的输入控件可见性"""
            return {
                preview_dataset: gr.update(visible=choice == "Use Pre-defined Dataset"),
                preview_custom_dataset: gr.update(visible=choice == "Use Custom Dataset")
            }
        
        # 绑定数据集类型切换事件
        preview_dataset_type.change(
            fn=update_preview_inputs,
            inputs=[preview_dataset_type],
            outputs=[preview_dataset, preview_custom_dataset]
        )

        # Return components that need to be accessed from outside
        return {
            "output_text": progress_status,
            "plot_output": plot_output,
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