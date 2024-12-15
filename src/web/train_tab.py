import os
import json
import gradio as gr
from .utils import preview_command, save_arguments, build_command_list
from dataclasses import dataclass
from typing import Any, Dict, Union, Optional

@dataclass
class TrainingArgs:
    def __init__(self, args: list, plm_models: dict, dataset_configs: dict):
        # 基础参数
        self.plm_model = plm_models[args[0]]
        self.dataset_config = dataset_configs[args[1]]
        self.training_method = args[2]
        self.pooling_method = args[3]
        
        # 批处理参数
        self.batch_mode = args[4]
        if self.batch_mode == "Batch Size Mode":
            self.batch_size = args[5]
        else:
            self.batch_token = args[6]
        
        # 训练参数
        self.learning_rate = args[7]
        self.num_epochs = args[8]
        self.max_seq_len = args[9]
        self.gradient_accumulation_steps = args[10]
        self.warmup_steps = args[11]
        self.scheduler = args[12]
        self.loss_function = args[13]
        
        # 输出参数
        self.output_model_name = args[14]
        self.output_dir = args[15]
        
        # Wandb参数
        self.wandb_enabled = args[16]
        if self.wandb_enabled:
            self.wandb_project = args[17]
            self.wandb_entity = args[18]
        
        # 其他参数
        self.patience = args[19]
        self.num_workers = args[20]
        self.max_grad_norm = args[21]

    def to_dict(self) -> Dict[str, Any]:
        args_dict = {
            "plm_model": self.plm_model,
            "dataset_config": self.dataset_config,
            "training_method": self.training_method,
            "pooling_method": self.pooling_method,
            "learning_rate": self.learning_rate,
            "num_epochs": self.num_epochs,
            "max_seq_len": self.max_seq_len,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "warmup_steps": self.warmup_steps,
            "scheduler": self.scheduler,
            "loss_function": self.loss_function,
            "output_model_name": self.output_model_name,
            "output_dir": self.output_dir,
            "patience": self.patience,
            "num_workers": self.num_workers,
            "max_grad_norm": self.max_grad_norm
        }

        # 添加批处理参数
        if self.batch_mode == "Batch Size Mode":
            args_dict["batch_size"] = self.batch_size
        else:
            args_dict["batch_token"] = self.batch_token

        # 添加wandb参数
        if self.wandb_enabled:
            args_dict["wandb"] = True
            if self.wandb_project:
                args_dict["wandb_project"] = self.wandb_project
            if self.wandb_entity:
                args_dict["wandb_entity"] = self.wandb_entity

        return args_dict

def create_train_tab(monitor, constant):
    plm_models = constant["plm_models"]
    dataset_configs = constant["dataset_configs"]

    with gr.Tab("Training"):
        # Model and Dataset Selection
        gr.Markdown("### Model and Dataset Configuration")
        with gr.Group():
            with gr.Row():
                with gr.Column():
                    plm_model = gr.Dropdown(
                        choices=list(plm_models.keys()),
                        label="Protein Language Model",
                        value=list(plm_models.keys())[0]
                    )
                
                with gr.Column():
                    dataset_config = gr.Dropdown(
                        choices=list(dataset_configs.keys()),
                        label="Dataset Configuration",
                        value=list(dataset_configs.keys())[0]
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
                        choices=["full", "freeze", "lora", "ses-adapter"],
                        label="Training Method",
                        value="freeze"
                    )
                with gr.Column(scale=1, min_width=150):
                    loss_function = gr.Dropdown(
                        choices=["cross_entropy", "focal_loss"],
                        label="Loss Function",
                        value="cross_entropy"
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
            refresh_button = gr.Button("Refresh",  elem_id="refresh-btn")
            abort_button = gr.Button("Abort", variant="stop")
            train_button = gr.Button("Start", variant="primary")
        
        
        with gr.Row():
            command_preview = gr.Code(
                label="Command Preview",
                language="shell",
                interactive=False,
                visible=False
            )
        
        # Training Status and Plot, same height
        with gr.Row(equal_height=True):
            # left: training status
            with gr.Column(scale=3):
                output_text = gr.Textbox(
                    label="Training Status",
                    interactive=False,
                    elem_id="training-status-box",
                    show_copy_button=True,
                )

            # right: training plot  
            with gr.Column(scale=1):
                plot_output = gr.Plot(
                    label="Training Progress",
                    elem_id="training-plot",
                )


        # CSS styles for scrollbar and layout
        gr.HTML("""
            <style>
                /* 设置文本框容器和文本区域的高度 */
                #training-status-box {
                    height: 300px !important;
                }
                
                #training-status-box textarea {
                    height: 90% !important;  /* 填充整个容器高度 */
                    overflow-y: auto;
                    scrollbar-width: thin;
                    scrollbar-color: #888888 #f0f0f0;
                    min-height: 250px;
                }
                
                #training-status-box textarea::-webkit-scrollbar {
                    width: 8px;
                }
                
                #training-status-box textarea::-webkit-scrollbar-track {
                    background: #f0f0f0;
                    border-radius: 4px;
                }
                
                #training-status-box textarea::-webkit-scrollbar-thumb {
                    background: #888888;
                    border-radius: 4px;
                }
                
                #training-status-box textarea::-webkit-scrollbar-thumb:hover {
                    background: #555555;
                }

                #training-plot {
                    height: 300px;  /* 与文本框相同高度 */
                }
            </style>
        """)

        # define all processing functions
        def handle_preview(*args):
            if command_preview.visible:
                return gr.update(visible=False)
            
            training_args = TrainingArgs(args, plm_models, dataset_configs)
            preview_text = preview_command(training_args.to_dict())
            return gr.update(value=preview_text, visible=True)

        def handle_train(*args):
            if monitor.is_training:
                return "Training is already in progress!"
            
            training_args = TrainingArgs(args, plm_models, dataset_configs)
            monitor.start_training(training_args.to_dict())
            return "Training started! Please wait for updates..."

        
        def handle_refresh():
            if monitor.is_training:
                messages = monitor.get_messages()
                plot = monitor.get_plot()
                return messages, plot
            else:
                return "Click Start to begin training!", None
        
        
        def handle_abort():
            monitor.abort_training()
            return "Training aborted!"

        def update_wandb_visibility(checkbox):
            return {
                wandb_project: gr.update(visible=checkbox),
                wandb_entity: gr.update(visible=checkbox)
            }

        # define all input components
        input_components = [
            plm_model,
            dataset_config, 
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
            loss_function,
            output_model_name,
            output_dir,
            wandb_logging,
            wandb_project,
            wandb_entity,
            patience,
            num_workers,
            max_grad_norm
        ]

        # bind preview and train buttons
        preview_button.click(
            fn=handle_preview,
            inputs=input_components,
            outputs=[command_preview]
        )
        
        refresh_button.click(
            fn=handle_refresh,
            outputs=[output_text, plot_output]
        )
        
        train_button.click(
            fn=handle_train, 
            inputs=input_components,
            outputs=[output_text]
        )

        # bind abort button
        abort_button.click(
            fn=handle_abort,
            outputs=[output_text]
        )
        
        wandb_logging.change(
            fn=update_wandb_visibility,
            inputs=[wandb_logging],
            outputs=[wandb_project, wandb_entity]
        )


        # Return components that need to be accessed from outside
        return {
            "output_text": output_text,
            "plot_output": plot_output,
            "train_button": train_button,
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
                "loss_function": loss_function,
                "output_model_name": output_model_name,
                "output_dir": output_dir,
                "wandb_logging": wandb_logging,
                "wandb_project": wandb_project,
                "wandb_entity": wandb_entity,
                "patience": patience,
                "num_workers": num_workers,
                "max_grad_norm": max_grad_norm
            }
        }