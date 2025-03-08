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
import numpy as np
import matplotlib.pyplot as plt
import re
from datasets import load_dataset

def create_eval_tab(constant):
    plm_models = constant["plm_models"]
    dataset_configs = constant["dataset_configs"]
    is_evaluating = False
    current_process = None
    output_queue = queue.Queue()
    stop_thread = False
    plm_models = constant["plm_models"]


    def evaluate_model(eval_method, plm_model, model_path, dataset, batch_size, eval_structure_seq, pooling_method, progress=gr.Progress()):
        nonlocal is_evaluating
        
        if is_evaluating:
            return "Evaluation is already in progress. Please wait...", gr.update(visible=False)
        
        is_evaluating = True
        stop_thread = False
        
        # Initialize progress info and start time
        start_time = time.time()
        progress_info = {
            "stage": "Preparing",
            "progress": 0,
            "total_samples": 0,
            "current": 0,
            "total": 0,
            "elapsed_time": "00:00:00",
            "lines": []
        }
        
        yield generate_progress_bar(progress_info), gr.update(visible=False)
        
        try:
            # Validate inputs
            if not model_path or not os.path.exists(os.path.dirname(model_path)):
                is_evaluating = False
                yield """
                <div style="padding: 10px; background-color: #ffebee; border-radius: 5px; margin-bottom: 10px;">
                    <p style="margin: 0; color: #c62828; font-weight: bold;">Error: Invalid model path</p>
                </div>
                """, gr.update(visible=False)
                return
            
            if is_custom_dataset == "Use Custom Dataset":
                dataset = dateset_custom
                test_file = dateset_custom
            else:
                dataset = dataset_defined
                if dataset not in dataset_configs:
                    is_evaluating = False
                    yield """
                    <div style="padding: 10px; background-color: #ffebee; border-radius: 5px; margin-bottom: 10px;">
                        <p style="margin: 0; color: #c62828; font-weight: bold;">Error: Invalid dataset selection</p>
                    </div>
                    """, gr.update(visible=False)
                    return
                config_path = dataset_configs[dataset]
                with open(config_path, 'r') as f:
                    dataset_config = json.load(f)
                test_file = dataset_config["dataset"]

            # Get dataset name
            dataset_display_name = dataset.split('/')[-1]
            test_result_name = f"test_results_{os.path.basename(model_path)}_{dataset_display_name}"
            test_result_dir = os.path.join(os.path.dirname(model_path), test_result_name)

            # Prepare command
            cmd = [sys.executable, "src/eval.py"]
            args_dict = {
                "eval_method": eval_method,
                "model_path": model_path,
                "test_file": test_file,
                "problem_type": problem_type,
                "num_labels": num_labels,
                "metrics": metrics,
                "plm_model": plm_models[plm_model],
                "test_result_dir": test_result_dir,
                "dataset": dataset_display_name,
                "pooling_method": pooling_method,
                "training_method": training_method
            }
            if batch_mode == "Batch Size Mode":
                args_dict["batch_size"] = batch_size
            else:
                args_dict["batch_token"] = batch_token

            if training_method == "ses-adapter":
                args_dict["structure_seq"] = eval_structure_seq
                # Add flags for using foldseek and ss8
                if "foldseek_seq" in eval_structure_seq:
                    args_dict["use_foldseek"] = True
                if "ss8_seq" in eval_structure_seq:
                    args_dict["use_ss8"] = True
            elif training_method == "plm-lora":
                args_dict["lora_rank"] = lora_r
                args_dict["lora_alpha"] = lora_alpha
                args_dict["lora_dropout"] = lora_dropout
                args_dict["lora_target_modules"] = lora_target_modules
                args_dict["structure_seq"] = ""
            else:
                args_dict["structure_seq"] = ""
            
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
            
            sample_pattern = r"Total samples: (\d+)"
            progress_pattern = r"(\d+)/(\d+)"
            
            last_update_time = time.time()
            
            while current_process.poll() is None:
                try:
                    new_lines = []
                    lines_processed = 0
                    while lines_processed < 10:
                        try:
                            line = output_queue.get_nowait()
                            new_lines.append(line)
                            progress_info["lines"].append(line)
                            
                            # Parse total samples
                            if "Total samples" in line:
                                match = re.search(sample_pattern, line)
                                if match:
                                    progress_info["total_samples"] = int(match.group(1))
                                    progress_info["stage"] = "Evaluating"
                            
                            # Parse progress
                            if "it/s" in line and "/" in line:
                                match = re.search(progress_pattern, line)
                                if match:
                                    progress_info["current"] = int(match.group(1))
                                    progress_info["total"] = int(match.group(2))
                                    progress_info["progress"] = (progress_info["current"] / progress_info["total"]) * 100
                            
                            if "Evaluation completed" in line:
                                progress_info["stage"] = "Completed"
                                progress_info["progress"] = 100
                            
                            lines_processed += 1
                        except queue.Empty:
                            break
                    
                    # 无论是否有新行，都更新时间信息
                    elapsed = time.time() - start_time
                    hours, remainder = divmod(int(elapsed), 3600)
                    minutes, seconds = divmod(remainder, 60)
                    progress_info["elapsed_time"] = f"{hours:02}:{minutes:02}:{seconds:02}"
                    
                    # 即使没有新行，也定期更新进度条（每0.5秒）
                    current_time = time.time()
                    if lines_processed > 0 or (current_time - last_update_time) >= 0.5:
                        # Generate progress bar HTML
                        progress_html = generate_progress_bar(progress_info)
                        # Only show progress bar, removing scrolling message output
                        yield f"{progress_html}", gr.update(visible=False)
                        last_update_time = current_time
                    
                    time.sleep(0.1)  # 减少循环间隔，使更新更频繁
                except Exception as e:
                    yield f"""
                    <div style="padding: 10px; background-color: #ffebee; border-radius: 5px; margin-bottom: 10px;">
                        <p style="margin: 0; color: #c62828;">Error reading output: {str(e)}</p>
                    </div>
                    """, gr.update(visible=False)
            
            if current_process.returncode == 0:
                # Load and format results
                result_file = os.path.join(test_result_dir, "test_metrics.csv")
                if os.path.exists(result_file):
                    metrics_html = format_metrics(result_file)
                    
                    # Calculate total evaluation time
                    total_time = time.time() - start_time
                    hours, remainder = divmod(int(total_time), 3600)
                    minutes, seconds = divmod(remainder, 60)
                    time_str = f"{hours:02}:{minutes:02}:{seconds:02}"
                    
                    summary_html = f"""
                    <div style="padding: 15px; background-color: #e8f5e9; border-radius: 5px; margin-bottom: 15px;">
                        <h3 style="margin-top: 0; color: #2e7d32;">Evaluation completed successfully!</h3>
                        <p><b>Total evaluation time:</b> {time_str}</p>
                        <p><b>Evaluation dataset:</b> {dataset_display_name}</p>
                        <p><b>Total samples:</b> {progress_info.get('total_samples', 0)}</p>
                    </div>
                    <div style="margin-top: 20px; font-weight: bold; font-size: 18px; text-align: center;">Evaluation Results</div>
                    {metrics_html}
                    """
                    
                    # 设置下载按钮可见并指向结果文件
                    yield summary_html, gr.update(value=result_file, visible=True)
                else:
                    yield """
                    <div style="padding: 10px; background-color: #fff8e1; border-radius: 5px; margin-bottom: 10px;">
                        <p style="margin: 0; color: #f57f17; font-weight: bold;">Evaluation completed, but metrics file not found.</p>
                    </div>
                    """, gr.update(visible=False)
            else:
                stderr_output = current_process.stderr.read() if current_process.stderr else "No error information available"
                yield f"""
                <div style="padding: 10px; background-color: #ffebee; border-radius: 5px; margin-bottom: 10px;">
                    <p style="margin: 0; color: #c62828; font-weight: bold;">Evaluation failed:</p>
                    <pre style="margin: 5px 0 0; white-space: pre-wrap;">{stderr_output}</pre>
                </div>
                """, gr.update(visible=False)

        except Exception as e:
            yield f"""
            <div style="padding: 10px; background-color: #ffebee; border-radius: 5px; margin-bottom: 10px;">
                <p style="margin: 0; color: #c62828; font-weight: bold;">Error during evaluation process:</p>
                <pre style="margin: 5px 0 0; white-space: pre-wrap;">{str(e)}</pre>
            </div>
            """, gr.update(visible=False)
        finally:
            if current_process:
                stop_thread = True
                is_evaluating = False
                current_process = None

    def generate_progress_bar(progress_info):
        """Generate HTML for evaluation progress bar"""
        stage = progress_info.get("stage", "Preparing")
        progress = progress_info.get("progress", 0)
        current = progress_info.get("current", 0)
        total = progress_info.get("total", 0)
        total_samples = progress_info.get("total_samples", 0)
        
        # 确保进度在0-100之间
        progress = max(0, min(100, progress))
        
        # 准备详细信息
        details = []
        if total_samples > 0:
            details.append(f"Total samples: {total_samples}")
        if current > 0 and total > 0:
            details.append(f"Current progress: {current}/{total}")
        
        # 计算评估时间（如果有）
        elapsed_time = progress_info.get("elapsed_time", "")
        if elapsed_time:
            details.append(f"Elapsed time: {elapsed_time}")
        
        details_text = ", ".join(details)
        
        # 创建更现代化的进度条
        html = f"""
        <div style="background-color: #f8f9fa; border-radius: 10px; padding: 20px; margin-bottom: 15px; box-shadow: 0 2px 5px rgba(0,0,0,0.05);">
            <div style="display: flex; justify-content: space-between; margin-bottom: 12px;">
                <div>
                    <span style="font-weight: 600; font-size: 16px;">Evaluation Status: </span>
                    <span style="color: #1976d2; font-weight: 500; font-size: 16px;">{stage}</span>
                </div>
                <div>
                    <span style="font-weight: 600; color: #333;">{progress:.1f}%</span>
                </div>
            </div>
            
            <div style="margin-bottom: 15px; background-color: #e9ecef; height: 10px; border-radius: 5px; overflow: hidden;">
                <div style="background-color: #4285f4; width: {progress}%; height: 100%; border-radius: 5px; transition: width 0.3s ease;"></div>
            </div>
            
            <div style="display: flex; flex-wrap: wrap; gap: 10px; font-size: 14px; color: #555;">
                {f'<div style="background-color: #e3f2fd; padding: 5px 10px; border-radius: 4px;"><span style="font-weight: 500;">Total samples:</span> {total_samples}</div>' if total_samples > 0 else ''}
                {f'<div style="background-color: #e8f5e9; padding: 5px 10px; border-radius: 4px;"><span style="font-weight: 500;">Progress:</span> {current}/{total}</div>' if current > 0 and total > 0 else ''}
                {f'<div style="background-color: #fff8e1; padding: 5px 10px; border-radius: 4px;"><span style="font-weight: 500;">Time:</span> {elapsed_time}</div>' if elapsed_time else ''}
            </div>
        </div>
        """
        return html

    def handle_abort():
        nonlocal is_evaluating, current_process, stop_thread
        if current_process is not None:
            try:
                stop_thread = True
                os.killpg(os.getpgid(current_process.pid), signal.SIGTERM)
                current_process.wait(timeout=5)
                current_process = None
                is_evaluating = False
                return """
                <div style="padding: 10px; background-color: #e8f5e9; border-radius: 5px;">
                    <p style="margin: 0; color: #2e7d32; font-weight: bold;">Evaluation successfully terminated!</p>
                </div>
                """, gr.update(visible=False)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(current_process.pid), signal.SIGKILL)
                    return """
                    <div style="padding: 10px; background-color: #fff8e1; border-radius: 5px;">
                        <p style="margin: 0; color: #f57f17; font-weight: bold;">Evaluation forcefully terminated!</p>
                    </div>
                    """, gr.update(visible=False)
                except Exception as e:
                    return f"""
                    <div style="padding: 10px; background-color: #ffebee; border-radius: 5px;">
                        <p style="margin: 0; color: #c62828; font-weight: bold;">Failed to terminate evaluation: {str(e)}</p>
                    </div>
                    """, gr.update(visible=False)
            except Exception as e:
                return f"""
                <div style="padding: 10px; background-color: #ffebee; border-radius: 5px;">
                    <p style="margin: 0; color: #c62828; font-weight: bold;">Failed to terminate evaluation: {str(e)}</p>
                </div>
                """, gr.update(visible=False)
        return """
        <div style="padding: 10px; background-color: #f5f5f5; border-radius: 5px;">
            <p style="margin: 0;">No evaluation in progress to terminate.</p>
        </div>
        """, gr.update(visible=False)

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
            
            # Add dataset preview functionality
            with gr.Row():
                preview_button = gr.Button("Preview Dataset", variant="primary")
            
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
            </style>
            """, visible=True)
            
            ### These are settings for custom dataset. ###
            with gr.Row(visible=True) as custom_dataset_row:
                with gr.Column(scale=1):
                    problem_type = gr.Dropdown(
                        choices=["single_label_classification", "multi_label_classification", "regression"],
                        label="Problem Type",
                        value="single_label_classification",
                        interactive=False
                    )
                with gr.Column(scale=1):
                    num_labels = gr.Number(
                        value=2,
                        label="Number of Labels",
                        interactive=False
                    )
                with gr.Column(scale=1):
                    metrics = gr.Textbox(
                        label="Metrics",
                        placeholder="accuracy,recall,precision,f1,mcc,auroc,f1max,spearman_corr,mse",
                        value="accuracy,mcc,f1,precision,recall,auroc",
                        interactive=False
                    )
            
            # Add dataset preview function
            def update_dataset_preview(dataset_type=None, defined_dataset=None, custom_dataset=None):
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
                elif dataset_type == "Use Pre-defined Dataset" and defined_dataset:
                    try:
                        config_path = dataset_configs[defined_dataset]
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
            preview_button.click(
                fn=update_dataset_preview,
                inputs=[is_custom_dataset, eval_dataset_defined, eval_dataset_custom],
                outputs=[dataset_stats_md, preview_table, preview_accordion]
            )

            def update_dataset_settings(choice, dataset_name=None):
                if choice == "Use Pre-defined Dataset":
                    # Load configuration from dataset_config
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
                    # Custom dataset settings
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
        
        # 使用HTML组件替代简单的Textbox，以支持更丰富的显示效果
        eval_output = gr.HTML(
            value="<div style='padding: 15px; background-color: #f5f5f5; border-radius: 5px;'><p style='margin: 0;'>Click the 「Start Evaluation」 button to begin model evaluation</p></div>",
            label="Evaluation Status & Results"
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
        
        # Connect buttons to functions
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