import os
import sys
import json
import subprocess
import torch
import queue
import gradio as gr
from datetime import datetime
from threading import Thread

monitor = None
constant = json.load(open("src/constant.json"))
plm_models = constant["plm_models"]
dataset_configs = constant["dataset_configs"]

class TrainingMonitor:
    def __init__(self):
        self.message_queue = queue.Queue()
        self.is_training = False
        self.current_epoch = 0
        self.total_epochs = 0
        self.current_step = 0
        self.total_steps = 0

    def start_training(self, args_dict):
        self.is_training = True
        self.current_epoch = 0
        self.total_epochs = int(args_dict.get("train_epoch", 0))
        thread = Thread(target=self._run_training, args=(args_dict,))
        thread.start()
        return thread

    def _run_training(self, args_dict):
        cmd = [sys.executable, "src/train.py"]
        for k, v in args_dict.items():
            if v is True:
                cmd.append(f"--{k}")
            elif v is not False and v is not None:
                cmd.append(f"--{k}")
                cmd.append(str(v))
        print(" ".join(cmd))
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True
            )

            for line in process.stdout:
                self.message_queue.put(line.strip())
                if "EPOCH" in line and "TRAIN loss:" in line:
                    try:
                        self.current_epoch = int(line.split("EPOCH")[1].split()[0])
                    except:
                        pass

            process.wait()
            if process.returncode == 0:
                self.message_queue.put("Training completed successfully!")
            else:
                self.message_queue.put(f"Training failed with return code {process.returncode}")
        except Exception as e:
            self.message_queue.put(f"Error occurred: {str(e)}")
        finally:
            self.is_training = False

    def get_messages(self):
        messages = []
        while not self.message_queue.empty():
            messages.append(self.message_queue.get())
        return "\n".join(messages)

    def get_progress(self):
        if not self.is_training:
            return ""
        progress = f"Training Progress: Epoch {self.current_epoch}/{self.total_epochs}"
        percentage = (self.current_epoch / self.total_epochs) * 100 if self.total_epochs > 0 else 0
        progress_bar = f"""
        <div style="width:100%; height:20px; background-color:#f0f0f0; border-radius:5px;">
            <div style="width:{percentage}%; height:100%; background-color:#4CAF50; border-radius:5px;">
            </div>
        </div>
        <p style="text-align:center;">{progress}</p>
        """
        return progress_bar

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
    global monitor
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

    if max_seq_len is not None:
        args_dict["max_seq_len"] = max_seq_len
    
    if wandb_logging:
        args_dict["wandb"] = True
        if wandb_project:
            args_dict["wandb_project"] = wandb_project
        if wandb_entity:
            args_dict["wandb_entity"] = wandb_entity

    monitor.start_training(args_dict)
    return "Training started! Please wait for updates..."

def create_ui():
    global monitor
    monitor = TrainingMonitor()

    def preview_command(
        model_name,
        dataset_config,
        training_method,
        batch_mode,
        size_value,
        token_value,
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
        # build command dictionary
        args_dict = {
            "plm_model": plm_models[model_name],
            "dataset_config": dataset_configs[dataset_config],
            "train_method": training_method,
            "learning_rate": learning_rate,
            "train_epoch": num_epochs,
            "max_seq_len": max_seq_len,
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

        # add batch related parameters
        mode = "size" if batch_mode == "Batch Size Mode" else "token"
        value = size_value if mode == "size" else token_value
        if mode == "size":
            args_dict["batch_size"] = value
        else:
            args_dict["batch_token"] = value

        # add wandb related parameters
        if wandb_logging:
            args_dict["wandb"] = True
            if wandb_project:
                args_dict["wandb_project"] = wandb_project
            if wandb_entity:
                args_dict["wandb_entity"] = wandb_entity

        # build command
        cmd = [sys.executable, "src/train.py"]
        for k, v in args_dict.items():
            if v is True:
                cmd.append(f"--{k}")
            elif v is not False and v is not None:
                cmd.append(f"--{k}")
                cmd.append(str(v))

        # build final command
        final_cmd = ""
        for i, part in enumerate(cmd):
            if i > 0:  # Skip first element (python path)
                if part.startswith("--"):
                    final_cmd += "\n\t" + part
                else:
                    # add '\' at end of part if not the last one
                    final_cmd += " " + part + (" \\" if i != len(cmd) - 1 else "")
            else:
                final_cmd += part

        return gr.update(value=final_cmd, visible=True)
    
    def save_arguments(
        save_path,
        model_name,
        dataset_config,
        training_method,
        batch_mode,
        size_value,
        token_value,
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
        if not save_path:
            return gr.update(value="Please enter a save path!", visible=True)
            
        # build arguments dictionary
        args_dict = {
            "model_name": model_name,
            "dataset_config": dataset_config,
            "training_method": training_method,
            "batch_mode": batch_mode,
            "batch_value": size_value if batch_mode == "Batch Size Mode" else token_value,
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

        try:
            # ensure directory exists
            save_path = os.path.join('configs', save_path)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            # save arguments to json file
            with open(save_path, 'w') as f:
                json.dump(args_dict, f, indent=4)
            return gr.update(value=f"Arguments saved to {save_path}", visible=True)
        except Exception as e:
            return gr.update(value=f"Error saving arguments: {str(e)}", visible=True)

    
    def update_output():
        if monitor.is_training:
            messages = monitor.get_messages()
            progress = monitor.get_progress()
            return messages, gr.update(value=progress, visible=True)
        return "", gr.update(value="", visible=False)

    with gr.Blocks() as demo:
        gr.Markdown("# ProFactory Training Interface")
        
        with gr.Tabs():
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
                gr.Markdown("### Training Parameters (-1 for unlimited)")
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
                                label="Max Sequence Length"
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
                                label="Max Gradient Norm"
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
                    refresh_button = gr.Button("Refresh", variant="secondary")
                
                # 创建保存参数的组件组
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

                # 绑定预览按钮事件
                preview_button.click(
                    fn=preview_command,
                    inputs=[
                        model_name,
                        dataset_config,
                        training_method,
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
                        save_dir,
                        wandb_logging,
                        wandb_project,
                        wandb_entity,
                        patience,
                        num_worker,
                        max_grad_norm
                    ],
                    outputs=[command_preview]
                )

                # 点击Save Arguments按钮显示保存组件
                save_args_button.click(
                    fn=lambda: gr.update(visible=True),
                    inputs=None,
                    outputs=save_group
                )

                # 点击Save确认按钮保存参数
                save_confirm.click(
                    fn=save_arguments,
                    inputs=[
                        save_path,
                        model_name,
                        dataset_config,
                        training_method,
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
                        save_dir,
                        wandb_logging,
                        wandb_project,
                        wandb_entity,
                        patience,
                        num_worker,
                        max_grad_norm
                    ],
                    outputs=[save_status]
                ).then(
                    fn=lambda: gr.update(visible=True),
                    inputs=None,
                    outputs=save_status
                )

                def update_wandb_visibility(checkbox):
                    return {
                        wandb_project: gr.update(visible=checkbox),
                        wandb_entity: gr.update(visible=checkbox)
                    }
                
                wandb_logging.change(
                    update_wandb_visibility,
                    inputs=[wandb_logging],
                    outputs=[wandb_project, wandb_entity]
                )

            with gr.Tab("Evaluation"):
                gr.Markdown("## Model Evaluation")
                with gr.Row():
                    with gr.Column():
                        eval_model_path = gr.Textbox(
                            label="Model Path",
                            placeholder="Path to the trained model"
                        )
                        eval_dataset = gr.Dropdown(
                            choices=list(dataset_configs.keys()),
                            label="Evaluation Dataset"
                        )
                    with gr.Column():
                        eval_batch_size = gr.Slider(
                            minimum=1,
                            maximum=128,
                            value=32,
                            step=1,
                            label="Evaluation Batch Size"
                        )
                
                eval_button = gr.Button("Start Evaluation")
                eval_output = gr.Textbox(label="Evaluation Results", lines=10)

        def train_with_batch_mode(
            model_name,
            dataset_config,
            training_method,
            batch_mode,
            size_value,
            token_value,
            *args
        ):
            mode = "size" if batch_mode == "Batch Size Mode" else "token"
            value = size_value if mode == "size" else token_value
            
            return train_model(
                model_name,
                dataset_config,
                training_method,
                mode,
                value,
                *args
            )
        
        train_button.click(
            fn=train_with_batch_mode,
            inputs=[
                model_name,
                dataset_config,
                training_method,
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
                save_dir,
                wandb_logging,
                wandb_project,
                wandb_entity,
                patience,
                num_worker,
                max_grad_norm
            ],
            outputs=[output_text]
        )
        
        refresh_button.click(
            fn=update_output,
            inputs=None,
            outputs=[output_text, progress_bar]
        )

        demo.load(
            fn=update_output,
            inputs=None,
            outputs=[output_text, progress_bar]
        )

    return demo

if __name__ == "__main__":
    demo = create_ui()
    demo.launch(server_name="0.0.0.0", share=True)