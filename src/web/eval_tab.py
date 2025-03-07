import gradio as gr
import json
import os
import subprocess
import sys
from functools import partial

def create_inference_tab(constant):
    plm_models = constant["plm_models"]
    dataset_configs = constant["dataset_configs"]
    is_evaluating = False
    plm_models = constant["plm_models"]
    def format_metrics(metrics):
        """Format metrics dictionary into a readable string."""
        return "\n".join([f"{k}: {v:.4f}" for k, v in metrics.items()])

    def evaluate_model(eval_method, plm_model, model_path, dataset, batch_size, eval_structure_seq, pooling_method, progress=gr.Progress()):
        nonlocal is_evaluating
        
        if is_evaluating:
            return "Evaluation is already in progress. Please wait..."
        
        is_evaluating = True
        yield "Starting evaluation..."
        progress(0.1)
        
        try:
            # Validate inputs
            if not model_path or not os.path.exists(os.path.dirname(model_path)):
                is_evaluating = False
                yield "Error: Invalid model path"
                return
                
            if dataset not in dataset_configs:
                is_evaluating = False
                yield "Error: Invalid dataset selection"
                return

            config_path = dataset_configs[dataset]
            with open(config_path, 'r') as f:
                dataset_config = json.load(f)

            test_result_name = f"test_results_{os.path.basename(model_path)}_{dataset}"
            test_result_dir = os.path.join(os.path.dirname(model_path), test_result_name)

            # Prepare command
            cmd = [sys.executable, "src/eval.py"]
            args_dict = {
                "eval_method": eval_method,
                "model_path": model_path,
                "test_file": dataset_config["dataset"],
                "problem_type": dataset_config["problem_type"],
                "num_labels": dataset_config["num_labels"],
                "metrics": dataset_config["metrics"],
                "batch_size": batch_size,
                "plm_model": plm_models[plm_model],
                "test_result_dir": test_result_dir,
                "structure_seq": eval_structure_seq,
                "dataset": dataset,
                "pooling_method": pooling_method
            }
            
            for k, v in args_dict.items():
                if v is True:
                    cmd.append(f"--{k}")
                elif v is not False and v is not None:
                    cmd.append(f"--{k}")
                    cmd.append(str(v))

            yield "Running evaluation..."
            progress(0.3)
            # 获取当前脚本所在目录
            current_dir = os.path.dirname(os.path.abspath(__file__))
            progress(0.4)
            # 获取 VenusFactory 根目录，假设它在当前目录的上一级
            pro_factory_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=pro_factory_dir
            )
            progress(0.7)
            yield "Processing results..."

            if result.returncode == 0:
                metrics = {}
                for line in result.stdout.split('\n'):
                    if any(x in line for x in ["accuracy:", "f1:", "precision:", "recall:", "spearman_corr:", "loss:"]):
                        try:
                            name, value = line.strip().split(':')
                            metrics[name.strip()] = float(value.strip())
                        except:
                            continue
                
                progress(1.0)
                is_evaluating = False
                yield f"Evaluation completed successfully!\n\nMetrics:\n{format_metrics(metrics)}"
            else:
                is_evaluating = False
                yield f"Evaluation failed:\n{result.stderr}\n{result.stdout}"

        except Exception as e:
            is_evaluating = False
            yield f"Error occurred during evaluation:\n{str(e)}"

    with gr.Tab("Evaluation"):
        gr.Markdown("## Model Evaluation")
        with gr.Row():
            with gr.Column():
                eval_method = gr.Dropdown(
                        choices=["full", "freeze", "lora", "ses-adapter", "plm-lora", "plm-qlora"],
                        label="evaluation Method",
                        value="freeze"
                    )
                eval_model_path = gr.Textbox(
                    label="Model Path",
                    placeholder="Path to the trained model"
                )
                eval_dataset = gr.Dropdown(
                    choices=list(dataset_configs.keys()),
                    label="Evaluation Dataset"
                )
            with gr.Column():
                with gr.Row():
                    eval_batch_size = gr.Slider(
                        minimum=1,
                        maximum=128,
                        value=32,
                        step=1,
                        label="Evaluation Batch Size"
                    )
                with gr.Row():
                    # 默认foldseek_seq,ss8_seq
                    eval_structure_seq = gr.Textbox(label="Structure Sequence", placeholder="foldseek_seq,ss8_seq")
        with gr.Row():
            with gr.Column():
                eval_plm_model = gr.Dropdown(
                    choices=list(plm_models.keys()),
                    label="Protein Language Model",
                )
            with gr.Column():
                eval_pooling_method = gr.Dropdown(
                    choices=["mean", "attention1d", "light_attention"],
                    label="Pooling Method",
                )
            

        eval_button = gr.Button("Start Evaluation")
        eval_output = gr.Textbox(label="Evaluation Results", lines=10)

        # Bind evaluation event
        eval_button.click(
            fn=evaluate_model,
            inputs=[eval_method, eval_plm_model, eval_model_path, eval_dataset, eval_batch_size, eval_structure_seq, eval_pooling_method],
            outputs=eval_output,
            queue=True  # Enable queuing for generators
        )

        return {
            "eval_button": eval_button,
            "eval_output": eval_output
        }