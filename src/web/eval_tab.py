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

def create_inference_tab(constant):
    plm_models = constant["plm_models"]
    dataset_configs = constant["dataset_configs"]
    is_evaluating = False
    current_process = None
    output_queue = queue.Queue()
    stop_thread = False
    plm_models = constant["plm_models"]

    def format_metrics(metrics_file):
        """将指标转换成表格式的HTML以便于显示"""
        try:
            df = pd.read_csv(metrics_file)
            metrics_dict = df.iloc[0].to_dict()
            
            # 创建HTML表格
            html = """
            <div style="margin: 20px 0;">
                <table style="width: 100%; border-collapse: collapse; margin: 0 auto; background-color: white;">
                    <tr>
                        <th style="padding: 8px; font-size: 16px; border: 1px solid #ddd; background-color: #e0e0e0; font-weight: bold; border-bottom: 1px solid #ccc;">Metric</th>
                        <th style="padding: 8px; font-size: 16px; border: 1px solid #ddd; background-color: #e0e0e0; font-weight: bold; border-bottom: 1px solid #ccc;">Value</th>
                    </tr>
            """
            
            # 添加每个指标
            for key, value in metrics_dict.items():
                if isinstance(value, (int, float)):
                    value_str = f"{value:.4f}" if isinstance(value, float) else str(value)
                else:
                    value_str = str(value)
                
                html += f"""
                    <tr>
                        <td style="padding: 8px; font-size: 14px; border: 1px solid #ddd;">{key}</td>
                        <td style="padding: 8px; font-size: 14px; border: 1px solid #ddd;">{value_str}</td>
                    </tr>
                """
            
            html += """
                </table>
            </div>
            """
            
            return html
                
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
        
        yield generate_progress_bar(progress_info)
        
        try:
            # Validate inputs
            if not model_path or not os.path.exists(os.path.dirname(model_path)):
                is_evaluating = False
                yield """
                <div style="padding: 10px; background-color: #ffebee; border-radius: 5px; margin-bottom: 10px;">
                    <p style="margin: 0; color: #c62828; font-weight: bold;">Error: Invalid model path</p>
                </div>
                """
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
                    """
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
                    
                    # Update time information
                    elapsed = time.time() - start_time
                    hours, remainder = divmod(int(elapsed), 3600)
                    minutes, seconds = divmod(remainder, 60)
                    progress_info["elapsed_time"] = f"{hours:02}:{minutes:02}:{seconds:02}"
                    
                    if lines_processed > 0:
                        # Generate progress bar HTML
                        progress_html = generate_progress_bar(progress_info)
                        # Only show progress bar, removing scrolling message output
                        yield f"{progress_html}"
                    
                    time.sleep(0.2)
                except Exception as e:
                    yield f"""
                    <div style="padding: 10px; background-color: #ffebee; border-radius: 5px; margin-bottom: 10px;">
                        <p style="margin: 0; color: #c62828;">Error reading output: {str(e)}</p>
                    </div>
                    """
            
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
                    <div style="margin-top: 20px; font-weight: bold; font-size: 18px;">Evaluation Results:</div>
                    {metrics_html}
                    """
                    
                    yield summary_html
                else:
                    yield """
                    <div style="padding: 10px; background-color: #fff8e1; border-radius: 5px; margin-bottom: 10px;">
                        <p style="margin: 0; color: #f57f17; font-weight: bold;">Evaluation completed, but metrics file not found.</p>
                    </div>
                    """
            else:
                stderr_output = current_process.stderr.read() if current_process.stderr else "No error information available"
                yield f"""
                <div style="padding: 10px; background-color: #ffebee; border-radius: 5px; margin-bottom: 10px;">
                    <p style="margin: 0; color: #c62828; font-weight: bold;">Evaluation failed:</p>
                    <pre style="margin: 5px 0 0; white-space: pre-wrap;">{stderr_output}</pre>
                </div>
                """

        except Exception as e:
            yield f"""
            <div style="padding: 10px; background-color: #ffebee; border-radius: 5px; margin-bottom: 10px;">
                <p style="margin: 0; color: #c62828; font-weight: bold;">Error during evaluation process:</p>
                <pre style="margin: 5px 0 0; white-space: pre-wrap;">{str(e)}</pre>
            </div>
            """
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
        
        # Ensure progress is between 0-100
        progress = max(0, min(100, progress))
        
        # Create progress bar style
        progress_bar = "█" * int(progress // 5) + "░" * (20 - int(progress // 5))
        
        # Prepare details
        details = []
        if total_samples > 0:
            details.append(f"Total samples: {total_samples}")
        if current > 0 and total > 0:
            details.append(f"Current progress: {current}/{total}")
        
        # Calculate evaluation time (if available)
        elapsed_time = progress_info.get("elapsed_time", "")
        if elapsed_time:
            details.append(f"Elapsed time: {elapsed_time}")
        
        details_text = ", ".join(details)
        
        html = f"""
        <div style="background-color: #f5f5f5; border-radius: 8px; padding: 15px; margin-bottom: 10px;">
            <div style="margin-bottom: 10px;">
                <span style="font-weight: bold; font-size: 16px;">Evaluation status: </span>
                <span style="color: #1976d2; font-size: 16px;">{stage}</span>
            </div>
            
            <div style="margin-bottom: 5px;">
                <span style="font-weight: bold;">Details: </span>
                <span>{details_text}</span>
            </div>
            
            <div style="margin-bottom: 5px;">
                <span style="font-weight: bold;">Progress: </span>
                <span>{progress:.1f}%</span>
            </div>
            
            <div style="font-family: monospace; font-size: 16px; margin-bottom: 10px;">
                |{progress_bar}| 
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
                """
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(current_process.pid), signal.SIGKILL)
                    return """
                    <div style="padding: 10px; background-color: #fff8e1; border-radius: 5px;">
                        <p style="margin: 0; color: #f57f17; font-weight: bold;">Evaluation forcefully terminated!</p>
                    </div>
                    """
                except Exception as e:
                    return f"""
                    <div style="padding: 10px; background-color: #ffebee; border-radius: 5px;">
                        <p style="margin: 0; color: #c62828; font-weight: bold;">Failed to terminate evaluation: {str(e)}</p>
                    </div>
                    """
            except Exception as e:
                return f"""
                <div style="padding: 10px; background-color: #ffebee; border-radius: 5px;">
                    <p style="margin: 0; color: #c62828; font-weight: bold;">Failed to terminate evaluation: {str(e)}</p>
                </div>
                """
        return """
        <div style="padding: 10px; background-color: #f5f5f5; border-radius: 5px;">
            <p style="margin: 0;">No evaluation in progress to terminate.</p>
        </div>
        """

    with gr.Tab("Inference"):

        with gr.Row():
            with gr.Column():
                gr.Markdown("## Evaluate & Predict")

        # 评估选项卡
        with gr.Tab("Evaluation"):
            gr.Markdown("### Model and Dataset Configuration")

            # 原评估界面组件
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
                
                # 添加数据集预览功能
                with gr.Row():
                    preview_button = gr.Button("Preview Dataset", variant="primary")
                
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
                    /* 表格内容应用白色背景，与train_tab保持一致 */
                    .preview-table table {
                        background-color: white !important;
                    }
                    .preview-table .gr-block.gr-box {
                        background-color: white !important;
                    }
                    .preview-table .gr-input-label {
                        background-color: white !important;
                    }
                    /* 确保预览区域背景为白色 */
                    .preview-table {
                        background-color: white !important;
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
                
                # 添加数据集预览函数
                def update_dataset_preview(dataset_type=None, defined_dataset=None, custom_dataset=None):
                    """更新数据集预览内容"""
                    # 根据选择确定使用哪种数据集
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
                    
                    # 使用预定义数据集
                    elif dataset_type == "Use Pre-defined Dataset" and defined_dataset:
                        try:
                            config_path = dataset_configs[defined_dataset]
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
                                        <td style="padding: 15px; font-size: 18px; border: 1px solid #ddd; text-align: center;">{len(dataset["train"]) if "train" in dataset else 0}</td>
                                        <td style="padding: 15px; font-size: 18px; border: 1px solid #ddd; text-align: center;">{len(dataset["validation"]) if "validation" in dataset else 0}</td>
                                        <td style="padding: 15px; font-size: 18px; border: 1px solid #ddd; text-align: center;">{len(dataset["test"]) if "test" in dataset else 0}</td>
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
                
                # 预览按钮点击事件
                preview_button.click(
                    fn=update_dataset_preview,
                    inputs=[is_custom_dataset, eval_dataset_defined, eval_dataset_custom],
                    outputs=[dataset_stats_md, preview_table]
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
            
            # 使用HTML组件替代简单的Textbox，以支持更丰富的显示效果
            eval_output = gr.HTML(
                value="<div style='padding: 15px; background-color: #f5f5f5; border-radius: 5px;'><p style='margin: 0;'>点击「Start Evaluation」按钮开始评估模型</p></div>",
                label="Evaluation Status & Results"
            )
            
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