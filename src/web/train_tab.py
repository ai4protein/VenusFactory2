import os
import json
import gradio as gr
from .utils import preview_command, save_arguments, build_command_list

def create_train_tab(monitor, constant):
    plm_models = constant["plm_models"]
    dataset_configs = constant["dataset_configs"]

    with gr.Tab("Training"):
        # Model and Dataset Selection
        gr.Markdown("### Model and Dataset Configuration")
        with gr.Group():
            with gr.Row():
                with gr.Column():
                    model_name = gr.Dropdown(
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
                    num_worker = gr.Slider(
                        minimum=0, maximum=16, value=4, step=1,
                        label="Number of Workers"
                    )
                
        # Output and Logging Settings
        gr.Markdown("### Output and Logging Settings")
        with gr.Row():
            with gr.Column():
                save_dir = gr.Textbox(
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
            save_args_button = gr.Button("Save Arguments")
            load_args_button = gr.Button("Load Arguments")
            train_button = gr.Button("Start", variant="primary")
        
        # Save arguments components
        with gr.Group(visible=False) as save_group:
            with gr.Row(equal_height=True):
                save_path = gr.Textbox(
                    label="Save Path",
                    placeholder="Enter path to save arguments (e.g., configs/my_args.json)",
                    lines=1,
                    scale=3
                )
                save_confirm = gr.Button("Save", variant="primary", scale=1)
            
            save_status = gr.Textbox(
                label="Status",
                interactive=False,
                visible=False
            )
        
        command_preview = gr.Textbox(
            label="Preview Command",
            lines=3,
            visible=False,
            interactive=False
        )
        
        output_text = gr.Textbox(label="Training Status", lines=10)
        progress_bar = gr.HTML(visible=True)

        # Training function
        def train_model(
            model_name,
            dataset_config,
            training_method,
            batch_mode,
            batch_value,
            learning_rate,
            num_epochs,
            max_seq_len,
            gradient_accumulation_steps,
            warmup_steps,
            scheduler_type,
            loss_function,
            output_model_name,
            save_dir,
            wandb_logging,
            wandb_project,
            wandb_entity,
            patience,
            num_worker,
            max_grad_norm
        ):
            if monitor.is_training:
                return "Training is already in progress!"

            args_dict = {
                "plm_model": plm_models[model_name],
                "dataset_config": dataset_configs[dataset_config],
                "train_method": training_method,
                "learning_rate": learning_rate,
                "train_epoch": num_epochs,
                "gradient_accumulation_step": gradient_accumulation_steps,
                "warmup_step": warmup_steps,
                "scheduler": scheduler_type,
                "loss_fn": loss_function,
                "output_model_name": output_model_name,
                "output_dir": save_dir,
                "patience": patience,
                "num_worker": num_worker,
                "max_grad_norm": max_grad_norm
            }

            if batch_mode == "size":
                args_dict["batch_size"] = batch_value
            elif batch_mode == "token":
                args_dict["batch_token"] = batch_value

            if max_seq_len is not None and max_seq_len > 0:
                args_dict["max_seq_len"] = max_seq_len
            
            if wandb_logging:
                args_dict["wandb"] = True
                if wandb_project:
                    args_dict["wandb_project"] = wandb_project
                if wandb_entity:
                    args_dict["wandb_entity"] = wandb_entity

            monitor.start_training(args_dict)
            return "Training started! Please wait for updates..."

        # Event handlers
        def handle_preview():
            # 如果已经显示，则隐藏
            if command_preview.visible:
                return gr.update(visible=False)
            # 否则显示预览
            args = get_current_args()
            cmd_list = build_command_list(args)
            preview_text = preview_command(args, cmd_list)
            return gr.update(value=preview_text, visible=True)


        def handle_save():
            args = get_current_args()
            save_result = save_arguments(save_path.value, args)
            # 保存后隐藏保存组件和状态
            return [
                save_result,
                gr.update(visible=False)  # 隐藏save_group
            ]
        
        
        def update_wandb_visibility(checkbox):
            return {
                wandb_project: gr.update(visible=checkbox),
                wandb_entity: gr.update(visible=checkbox)
            }

        def get_current_args():
            return_dict = {
                "model_name": model_name.value,
                "dataset_config": dataset_config.value,
                "training_method": training_method.value,
                "learning_rate": learning_rate.value,
                "num_epochs": num_epochs.value,
                "max_seq_len": max_seq_len.value,
                "gradient_accumulation_steps": gradient_accumulation_steps.value,
                "warmup_steps": warmup_steps.value,
                "scheduler_type": scheduler_type.value,
                "loss_function": loss_function.value,
                "output_model_name": output_model_name.value,
                "save_dir": save_dir.value,
                "wandb_logging": wandb_logging.value,
                "wandb_project": wandb_project.value,
                "wandb_entity": wandb_entity.value,
                "patience": patience.value,
                "num_worker": num_worker.value,
                "max_grad_norm": max_grad_norm.value
            }
            if batch_mode.value == "Batch Size Mode":
                return_dict["batch_size"] = batch_size.value
            else:
                return_dict["batch_token"] = batch_token.value
            return return_dict

        # Bind events
        preview_button.click(fn=handle_preview, outputs=[command_preview])
        save_args_button.click(fn=lambda: gr.update(visible=True), outputs=save_group)
        save_confirm.click(fn=handle_save, outputs=[save_status])
        wandb_logging.change(fn=update_wandb_visibility, inputs=[wandb_logging], outputs=[wandb_project, wandb_entity])

        # Return components that need to be accessed from outside
        return {
            "output_text": output_text,
            "progress_bar": progress_bar,
            "train_button": train_button,
            "train_fn": train_model,
            "components": {
                "model_name": model_name,
                "dataset_config": dataset_config,
                "training_method": training_method,
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
                "save_dir": save_dir,
                "wandb_logging": wandb_logging,
                "wandb_project": wandb_project,
                "wandb_entity": wandb_entity,
                "patience": patience,
                "num_worker": num_worker,
                "max_grad_norm": max_grad_norm
            }
        }