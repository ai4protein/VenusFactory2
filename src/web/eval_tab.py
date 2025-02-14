import gradio as gr
import json
import os
import subprocess
import sys
import signal
import threading
import queue
import time
import pandas as pd

def create_inference_tab(constant):
    plm_models = constant["plm_models"]
    dataset_configs = constant["dataset_configs"]
    is_evaluating = False
    current_process = None
    output_queue = queue.Queue()
    stop_thread = False
    plm_models = constant["plm_models"]

    def format_metrics(metrics_file):
        """Format metrics from csv file into a readable string."""
        try:

            df = pd.read_csv(metrics_file)
            metrics_dict = df.iloc[0].to_dict()
            formatted_lines = [f"{key}: {value}" for key, value in metrics_dict.items()]
            return "\n".join(formatted_lines)
                
        except Exception as e:
            return f"Error formatting metrics: {str(e)}"

    def process_output(process, queue):
        nonlocal stop_thread
        while True:
            if stop_thread:
                break
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                queue.put(output.strip())
        process.stdout.close()

    def evaluate_model(plm_model, model_path, training_method, is_custom_dataset, dataset_defined, dateset_custom, problem_type, num_labels, metrics, batch_mode, batch_size, batch_token, eval_structure_seq, pooling_method):
        nonlocal is_evaluating, current_process, stop_thread
        
        if is_evaluating:
            return "Evaluation is already in progress. Please wait..."
        
        is_evaluating = True
        stop_thread = False
        yield "Starting evaluation..."
        
        try:
            # Validate inputs
            if not model_path or not os.path.exists(os.path.dirname(model_path)):
                is_evaluating = False
                yield "Error: Invalid model path"
                return
            
            if is_custom_dataset == "Use Custom Dataset":
                dataset = dateset_custom
                test_file = dateset_custom
            else:
                dataset = dataset_defined
                if dataset not in dataset_configs:
                    is_evaluating = False
                    yield "Error: Invalid dataset selection"
                    return
                config_path = dataset_configs[dataset]
                with open(config_path, 'r') as f:
                    dataset_config = json.load(f)
                test_file = dataset_config["dataset"]

            # split the dataset name from the 'user/dataset'
            dataset = dataset.split('/')[-1]
            test_result_name = f"test_results_{os.path.basename(model_path)}_{dataset}"
            test_result_dir = os.path.join(os.path.dirname(model_path), test_result_name)

            # Prepare command
            cmd = [sys.executable, "src/eval.py"]
            args_dict = {
                "model_path": model_path,
                "test_file": test_file,
                "problem_type": problem_type,
                "num_labels": num_labels,
                "metrics": metrics,
                "plm_model": plm_models[plm_model],
                "test_result_dir": test_result_dir,
                "dataset": dataset,
                "pooling_method": pooling_method
            }
            if batch_mode == "Batch Size Mode":
                args_dict["batch_size"] = batch_size
            else:
                args_dict["batch_token"] = batch_token

            if training_method == "ses-adapter":
                args_dict["structure_seq"] = eval_structure_seq
            elif training_method == "plm-lora":
                args_dict["lora_rank"] = lora_r
                args_dict["lora_alpha"] = lora_alpha
                args_dict["lora_dropout"] = lora_dropout
                args_dict["lora_target_modules"] = lora_target_modules
                args_dict["structure_seq"] = None
            else:
                args_dict["structure_seq"] = None
            
            for k, v in args_dict.items():
                if v is True:
                    cmd.append(f"--{k}")
                elif v is not False and v is not None:
                    cmd.append(f"--{k}")
                    cmd.append(str(v))
            
            # Start evaluation process
            current_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                preexec_fn=os.setsid
            )
            
            output_thread = threading.Thread(target=process_output, args=(current_process, output_queue))
            output_thread.daemon = True
            output_thread.start()
            
            while current_process.poll() is None:
                try:
                    lines_processed = 0
                    while lines_processed < 10:
                        try:
                            line = output_queue.get_nowait()
                            print(line, flush=True)
                            yield f"Progress: {line}"
                            lines_processed += 1
                        except queue.Empty:
                            break
                    if lines_processed == 0:
                        time.sleep(0.1)
                except Exception as e:
                    yield f"Error reading output: {str(e)}"
            
            if current_process.returncode == 0:
                # Load and format results
                result_file = os.path.join(test_result_dir, "test_metrics.csv")
                if os.path.exists(result_file):
                    with open(result_file, 'r') as f:
                        metrics = json.load(f)
                    formatted_results = format_metrics(metrics)
                    yield f"Evaluation completed successfully!\n{formatted_results}"
            else:
                stderr_output = current_process.stderr.read()
                yield f"Evaluation failed:\n{stderr_output}"

        except Exception as e:
            yield f"Error occurred during evaluation:\n{str(e)}"
        finally:
            if current_process:
                stop_thread = True
                is_evaluating = False
                current_process = None

            try:
                result_file = os.path.join(test_result_dir, "metrics.json")
                retry_count = 0
                while not os.path.exists(result_file) and retry_count < 5:
                    time.sleep(1)
                    retry_count += 1
                
                if os.path.exists(result_file):
                    formatted_results = format_metrics(result_file)
                    yield f"Evaluation completed successfully!\n{formatted_results}"
                else:
                    yield f"Evaluation completed successfully!\nNo metrics file found."
            except Exception as e:
                yield f"Error processing results: {str(e)}"

    def handle_abort():
        nonlocal is_evaluating, current_process, stop_thread
        if current_process is not None:
            try:
                stop_thread = True
                os.killpg(os.getpgid(current_process.pid), signal.SIGTERM)
                current_process.wait(timeout=5)
                current_process = None
                is_evaluating = False
                return "Evaluation aborted successfully!"
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(current_process.pid), signal.SIGKILL)
                    return "Evaluation forcefully terminated!"
                except Exception as e:
                    return f"Failed to forcefully terminate evaluation: {str(e)}"
            except Exception as e:
                return f"Failed to abort evaluation: {str(e)}"
        return "No evaluation process to abort."

    with gr.Tab("Inference"):

        with gr.Row():
            with gr.Column():
                gr.Markdown("## Evaluate & Predict")

        with gr.Tab("Evaluation"):


            gr.Markdown("### Model and Dataset Configuration")

            with gr.Group():

                with gr.Row():
                    eval_model_path = gr.Textbox(
                        label="Model Path",
                        placeholder="Path to the trained model"
                    )
                    eval_plm_model = gr.Dropdown(
                        choices=list(plm_models.keys()),
                        label="Protein Language Model"
                    )

                with gr.Row():
                        training_method = gr.Dropdown(
                            choices=["full", "freeze", "ses-adapter", "plm-lora"],
                            label="Training Method",
                            value="freeze"
                        )
                        eval_pooling_method = gr.Dropdown(
                            choices=["mean", "attention1d", "light_attention"],
                            label="Pooling Method",
                            value="mean"
                        )

                with gr.Row():
                    is_custom_dataset = gr.Radio(
                        choices=["Use Custom Dataset", "Use Pre-defined Dataset"],
                        label="Dataset Selection",
                        value="Use Pre-defined Dataset"
                    )
                    eval_dataset_defined = gr.Dropdown(
                        choices=list(dataset_configs.keys()),
                        label="Evaluation Dataset",
                        visible=True
                    )
                    eval_dataset_custom = gr.Textbox(
                        label="Custom Dataset Path",
                        placeholder="Huggingface Dataset eg: user/dataset",
                        visible=False
                    )
                
                ### These are settings for custom dataset. ###
                with gr.Row(visible=True) as custom_dataset_row:
                    problem_type = gr.Dropdown(
                        choices=["single_label_classification", "multi_label_classification", "regression"],
                        label="Problem Type",
                        value="single_label_classification",
                        interactive=False
                    )
                    num_labels = gr.Number(
                        value=2,
                        label="Number of Labels",
                        interactive=False
                    )
                    metrics = gr.Textbox(
                        label="Metrics",
                        placeholder="accuracy,recall,precision,f1,mcc,auroc,f1max,spearman_corr,mse",
                        value="accuracy,mcc,f1,precision,recall,auroc",
                        interactive=False
                    )


                def update_dataset_settings(choice, dataset_name=None):
                    if choice == "Use Pre-defined Dataset":
                        # 从dataset_config加载配置
                        if dataset_name and dataset_name in dataset_configs:
                            with open(dataset_configs[dataset_name], 'r') as f:
                                config = json.load(f)
                            return [
                                gr.update(visible=True),  # eval_dataset_defined
                                gr.update(visible=False), # eval_dataset_custom
                                gr.update(value=config.get("problem_type", ""), interactive=False),
                                gr.update(value=config.get("num_labels", 1), interactive=False),
                                gr.update(value=config.get("metrics", ""), interactive=False)
                            ]
                    else:
                        # 自定义数据集设置
                        return [
                            gr.update(visible=False),  # eval_dataset_defined
                            gr.update(visible=True),   # eval_dataset_custom
                            gr.update(value="", interactive=True),
                            gr.update(value=2, interactive=True),
                            gr.update(value="", interactive=True)
                        ]
                
                is_custom_dataset.change(
                    fn=update_dataset_settings,
                    inputs=[is_custom_dataset, eval_dataset_defined],
                    outputs=[eval_dataset_defined, eval_dataset_custom, 
                            problem_type, num_labels, metrics]
                )

                eval_dataset_defined.change(
                    fn=lambda x: update_dataset_settings("Use Pre-defined Dataset", x),
                    inputs=[eval_dataset_defined],
                    outputs=[eval_dataset_defined, eval_dataset_custom, 
                            problem_type, num_labels, metrics]
                )



                ### These are settings for different training methods. ###

                # for ses-adapter
                with gr.Row(visible=False) as structure_seq_row:
                    eval_structure_seq = gr.Textbox(label="Structure Sequence", placeholder="foldseek_seq,ss8_seq", value="foldseek_seq,ss8_seq")

                # for plm-lora
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
                        
            def update_training_method(method):
                return {
                    structure_seq_row: gr.update(visible=method == "ses-adapter"),
                    lora_params_row: gr.update(visible=method == "plm-lora")
                }

            training_method.change(
                fn=update_training_method,
                inputs=[training_method],
                outputs=[structure_seq_row, lora_params_row]
            )


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

            with gr.Row():
                eval_button = gr.Button("Start Evaluation", variant="primary")
                abort_button = gr.Button("Abort", variant="stop")
            eval_output = gr.Textbox(label="Evaluation Results", lines=10)
            
            # Connect buttons to functions
            eval_button.click(
                fn=evaluate_model,
                inputs=[
                    eval_plm_model,
                    eval_model_path,
                    training_method,
                    is_custom_dataset,
                    eval_dataset_defined,
                    eval_dataset_custom,
                    problem_type,
                    num_labels,
                    metrics,
                    batch_mode,
                    batch_size,
                    batch_token,
                    eval_structure_seq,
                    eval_pooling_method
                ],
                outputs=eval_output
            )
            abort_button.click(
                fn=handle_abort,
                inputs=[],
                outputs=eval_output
            )
            return {
                "eval_button": eval_button,
                "eval_output": eval_output
            }
            
        with gr.Tab("Prediction"):
            pass