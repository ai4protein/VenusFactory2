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
import tempfile
import csv
from pathlib import Path
import traceback
import re

def create_predict_tab(constant):
    plm_models = constant["plm_models"]
    is_predicting = False
    current_process = None
    output_queue = queue.Queue()
    stop_thread = False

    def process_output(process, queue):
        """Process output from subprocess and put it in queue"""
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

    def generate_status_html(status_info):
        """Generate HTML for single sequence prediction status"""
        stage = status_info.get("current_step", "Preparing")
        status = status_info.get("status", "running")
        
        # Determine status color and icon
        if status == "running":
            status_color = "#4285f4"  # Blue
            icon = "⏳"
            animation = """
            @keyframes pulse {
                0% { transform: scale(1); }
                50% { transform: scale(1.05); }
                100% { transform: scale(1); }
            }
            """
            animation_style = "animation: pulse 1.5s infinite ease-in-out;"
        elif status == "completed":
            status_color = "#2ecc71"  # Green
            icon = "✅"
            animation = ""
            animation_style = ""
        else:  # failed
            status_color = "#e74c3c"  # Red
            icon = "❌"
            animation = ""
            animation_style = ""
        
        # 创建简洁美观的居中提示
        return f"""
        <div style="text-align: center; background-color: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin: 20px 0;">
            <div style="display: inline-block; background-color: {status_color}; color: white; border-radius: 50%; width: 60px; height: 60px; line-height: 60px; font-size: 24px; margin-bottom: 15px; {animation_style}">
                {icon}
            </div>
            <h2 style="color: #2c3e50; margin: 10px 0; font-size: 20px;">{stage}</h2>
            <p style="color: #7f8c8d; font-size: 16px;">{status.capitalize()}</p>
            <style>
                {animation}
            </style>
        </div>
        """

    def predict_sequence(plm_model, model_path, eval_method, aa_seq, foldseek_seq, ss8_seq, eval_structure_seq, pooling_method, problem_type, num_labels):
        """Predict a single protein sequence"""
        nonlocal is_predicting, current_process, stop_thread
        
        if is_predicting:
            return gr.HTML("""
            <div class="warning-container">
                <div class="warning-icon">⚠️</div>
                <div class="warning-message">A prediction is already running. Please wait or abort it.</div>
            </div>
            <style>
                .warning-container {
                    background-color: #fffbea;
                    border-left: 5px solid #ecc94b;
                    padding: 15px;
                    border-radius: 5px;
                    margin: 10px 0;
                }
                .warning-icon {
                    font-size: 20px;
                    margin-bottom: 8px;
                }
                .warning-message {
                    font-weight: 500;
                }
            </style>
            """)
        
        is_predicting = True
        stop_thread = False
        
        # 创建一个状态信息对象，类似于批量预测中的progress_info
        status_info = {
            "status": "running",
            "current_step": "Starting prediction"
        }
        
        # 生成初始状态HTML
        yield generate_status_html(status_info)
        
        try:
            # Validate inputs
            if not model_path:
                is_predicting = False
                return gr.HTML("""
                <div class="error-container">
                    <div class="error-icon">❌</div>
                    <div class="error-message">Please provide a model path</div>
                </div>
                <style>
                    .error-container {
                        background-color: #fff5f5;
                        border-left: 5px solid #f56565;
                        padding: 15px;
                        border-radius: 5px;
                        margin: 10px 0;
                    }
                    .error-icon {
                        font-size: 20px;
                        margin-bottom: 8px;
                    }
                    .error-message {
                        font-weight: 500;
                    }
                </style>
                """)
                
            if not os.path.exists(os.path.dirname(model_path)):
                is_predicting = False
                return gr.HTML("""
                <div class="error-container">
                    <div class="error-icon">❌</div>
                    <div class="error-message">Invalid model path - directory does not exist</div>
                </div>
                <style>
                    .error-container {
                        background-color: #fff5f5;
                        border-left: 5px solid #f56565;
                        padding: 15px;
                        border-radius: 5px;
                        margin: 10px 0;
                    }
                    .error-icon {
                        font-size: 20px;
                        margin-bottom: 8px;
                    }
                    .error-message {
                        font-weight: 500;
                    }
                </style>
                """)
                
            if not aa_seq:
                is_predicting = False
                return gr.HTML("""
                <div class="error-container">
                    <div class="error-icon">❌</div>
                    <div class="error-message">Amino acid sequence is required</div>
                </div>
                <style>
                    .error-container {
                        background-color: #fff5f5;
                        border-left: 5px solid #f56565;
                        padding: 15px;
                        border-radius: 5px;
                        margin: 10px 0;
                    }
                    .error-icon {
                        font-size: 20px;
                        margin-bottom: 8px;
                    }
                    .error-message {
                        font-weight: 500;
                    }
                </style>
                """)
            
            # 更新状态
            status_info["current_step"] = "Preparing model and parameters"
            yield generate_status_html(status_info)
            
            # Prepare command
            cmd = [sys.executable, "src/predict.py"]
            args_dict = {
                "model_path": model_path,
                "plm_model": plm_models[plm_model],
                "aa_seq": aa_seq,
                "foldseek_seq": foldseek_seq if foldseek_seq else "",
                "ss8_seq": ss8_seq if ss8_seq else "",
                "pooling_method": pooling_method,
                "problem_type": problem_type,
                "num_labels": num_labels,
                "eval_method": eval_method
            }
            
            if eval_method == "ses-adapter":
                # 使用多选下拉框选择的结构序列
                args_dict["structure_seq"] = ",".join(eval_structure_seq) if eval_structure_seq else None
                
                # 根据选择的结构序列设置标志
                if eval_structure_seq:
                    if "foldseek_seq" in eval_structure_seq:
                        args_dict["use_foldseek"] = True
                    if "ss8_seq" in eval_structure_seq:
                        args_dict["use_ss8"] = True
            else:
                args_dict["structure_seq"] = None
                args_dict["use_foldseek"] = False
                args_dict["use_ss8"] = False
            
            # Build command line
            final_cmd = [sys.executable, "src/predict.py"]
            for k, v in args_dict.items():
                if v is True:
                    final_cmd.append(f"--{k}")
                elif v is not False and v is not None:
                    final_cmd.append(f"--{k}")
                    final_cmd.append(str(v))
            
            # 更新状态
            status_info["current_step"] = "Starting prediction process"
            yield generate_status_html(status_info)
            
            # Start prediction process
            try:
                current_process = subprocess.Popen(
                    final_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    universal_newlines=True,
                    preexec_fn=os.setsid if hasattr(os, "setsid") else None
                )
            except Exception as e:
                is_predicting = False
                return gr.HTML(f"""
                <div class="error-container">
                    <div class="error-icon">❌</div>
                    <div class="error-message">Error starting prediction process: {str(e)}</div>
                </div>
                <style>"""+"""
                    .error-container {
                        background-color: #fff5f5;
                        border-left: 5px solid #f56565;
                        padding: 15px;
                        border-radius: 5px;
                        margin: 10px 0;
                    }
                    .error-icon {
                        font-size: 20px;
                        margin-bottom: 8px;
                    }
                    .error-message {
                        font-weight: 500;
                    }
                </style>
                """)
            
            output_thread = threading.Thread(target=process_output, args=(current_process, output_queue))
            output_thread.daemon = True
            output_thread.start()
            
            # Collect output
            result_output = ""
            prediction_data = None
            json_str = ""
            in_json_block = False
            json_lines = []
            
            # 更新状态
            status_info["current_step"] = "Processing sequence"
            yield generate_status_html(status_info)
            
            while current_process.poll() is None:
                try:
                    while not output_queue.empty():
                        line = output_queue.get_nowait()
                        result_output += line + "\n"
                        
                        # 更新状态信息，添加更多有意义的状态提示
                        if "Loading model" in line:
                            status_info["current_step"] = "Loading model and tokenizer"
                        elif "Processing sequence" in line:
                            status_info["current_step"] = "Processing protein sequence"
                        elif "Tokenizing" in line:
                            status_info["current_step"] = "Tokenizing sequence"
                        elif "Forward pass" in line:
                            status_info["current_step"] = "Running model inference"
                        elif "Making prediction" in line:
                            status_info["current_step"] = "Calculating final prediction"
                        elif "Prediction Results" in line:
                            status_info["current_step"] = "Finalizing results"
                        
                        # 更新状态显示
                        yield generate_status_html(status_info)
                        
                        # Detect start of JSON results block
                        if "---------- Prediction Results ----------" in line:
                            in_json_block = True
                            json_lines = []
                            continue
                        
                        # If in JSON block, collect JSON lines
                        if in_json_block and line.strip():
                            json_lines.append(line.strip())
                            
                            # Try to parse the complete JSON when we have multiple lines
                            if line.strip() == "}":  # Potential end of JSON object
                                try:
                                    complete_json = " ".join(json_lines)
                                    # Clean up the JSON string by removing line breaks and extra spaces
                                    complete_json = re.sub(r'\s+', ' ', complete_json).strip()
                                    prediction_data = json.loads(complete_json)
                                    print(f"Successfully parsed complete JSON: {prediction_data}")
                                except json.JSONDecodeError as e:
                                    print(f"Failed to parse complete JSON: {e}")
                    
                    time.sleep(0.1)
                except Exception as e:
                    yield gr.HTML(f"""
                    <div class="warning-container">
                        <div class="warning-icon">⚠️</div>
                        <div class="warning-message">Warning reading output: {str(e)}</div>
                    </div>
                    <style>"""+"""
                        .warning-container {
                            background-color: #fffbea;
                            border-left: 5px solid #ecc94b;
                            padding: 15px;
                            border-radius: 5px;
                            margin: 10px 0;
                        }
                        .warning-icon {
                            font-size: 20px;
                            margin-bottom: 8px;
                        }
                        .warning-message {
                            font-weight: 500;
                        }
                    </style>
                    """)
            
            # Process has completed
            if current_process.returncode == 0:
                # 更新状态
                status_info["status"] = "completed"
                status_info["current_step"] = "Prediction completed successfully"
                yield generate_status_html(status_info)
                
                # If no prediction data found, try to parse from complete output
                if not prediction_data:
                    try:
                        # Find the JSON block in the output
                        results_marker = "---------- Prediction Results ----------"
                        if results_marker in result_output:
                            json_part = result_output.split(results_marker)[1].strip()
                            
                            # Try to extract the JSON object
                            json_match = re.search(r'(\{.*?\})', json_part.replace('\n', ' '), re.DOTALL)
                            if json_match:
                                try:
                                    json_str = json_match.group(1)
                                    # Clean up the JSON string
                                    json_str = re.sub(r'\s+', ' ', json_str).strip()
                                    prediction_data = json.loads(json_str)
                                    print(f"Parsed prediction data from regex: {prediction_data}")
                                except json.JSONDecodeError as e:
                                    print(f"JSON parse error from regex: {e}")
                    except Exception as e:
                        print(f"Error parsing JSON from complete output: {e}")
                
                if prediction_data:
                    # Create styled HTML table based on problem type
                    if problem_type == "regression":
                        html_result = f"""
                        <div class="results-container">
                            <h2>Regression Prediction Results</h2>
                            <table class='styled-table'>
                                <thead>
                                    <tr><th>Output</th><th>Value</th></tr>
                                </thead>
                                <tbody>
                                    <tr><td>Predicted Value</td><td>{prediction_data['prediction']:.4f}</td></tr>
                                </tbody>
                            </table>
                        </div>
                        """
                    elif problem_type == "single_label_classification":
                        # Create probability table
                        prob_rows = ""
                        if isinstance(prediction_data['probabilities'], list):
                            prob_rows = "".join([
                                f"<tr><td>Class {i}</td><td>{prob:.4f}</td></tr>"
                                for i, prob in enumerate(prediction_data['probabilities'])
                            ])
                        else:
                            # Handle case where probabilities is not a list
                            prob_rows = f"<tr><td>Class 0</td><td>{prediction_data['probabilities']:.4f}</td></tr>"
                            
                        html_result = f"""
                        <div class="results-container">
                            <h2>Single-Label Classification Results</h2>
                            <table class='styled-table'>
                                <thead>
                                    <tr><th>Output</th><th>Value</th></tr>
                                </thead>
                                <tbody>
                                    <tr><td>Predicted Class</td><td>{prediction_data['predicted_class']}</td></tr>
                                </tbody>
                            </table>
                            <h3 style='margin-top: 25px; margin-bottom: 15px;'>Class Probabilities</h3>
                            <table class='styled-table'>
                                <thead>
                                    <tr><th>Class</th><th>Probability</th></tr>
                                </thead>
                                <tbody>
                                    {prob_rows}
                                </tbody>
                            </table>
                        </div>
                        """
                    else:  # multi_label_classification
                        # Create prediction table
                        pred_rows = ""
                        if (isinstance(prediction_data['predictions'], list) and 
                            isinstance(prediction_data['probabilities'], list)):
                            pred_rows = "".join([
                                f"<tr><td>Label {i}</td><td>{pred}</td><td>{prob:.4f}</td></tr>"
                                for i, (pred, prob) in enumerate(zip(prediction_data['predictions'], prediction_data['probabilities']))
                            ])
                        else:
                            # Handle case where predictions or probabilities is not a list
                            pred = prediction_data['predictions'] if 'predictions' in prediction_data else "N/A"
                            prob = prediction_data['probabilities'] if 'probabilities' in prediction_data else 0.0
                            pred_rows = f"<tr><td>Label 0</td><td>{pred}</td><td>{prob:.4f}</td></tr>"
                            
                        html_result = f"""
                        <div class="results-container">
                            <h2>Multi-Label Classification Results</h2>
                            <table class='styled-table'>
                                <thead>
                                    <tr><th>Label</th><th>Prediction</th><th>Probability</th></tr>
                                </thead>
                                <tbody>
                                    {pred_rows}
                                </tbody>
                            </table>
                        </div>
                        """
                    
                    # Add CSS styling
                    html_result += """
                    <style>
                        .results-container {
                            background-color: white;
                            border-radius: 8px;
                            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
                            padding: 20px;
                            margin-bottom: 20px;
                        }
                        
                        .results-container h2 {
                            color: #2c3e50;
                            text-align: center;
                            margin-bottom: 20px;
                            font-size: 20px;
                        }
                        
                        .results-container h3 {
                            color: #2c3e50;
                            text-align: center;
                            margin-bottom: 15px;
                            font-size: 18px;
                        }
                        
                        .styled-table {
                            border-collapse: collapse;
                            margin: 25px auto;
                            font-size: 14px;
                            font-family: sans-serif;
                            min-width: 400px;
                            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
                            border-radius: 6px;
                            overflow: hidden;
                        }
                        .styled-table thead tr {
                            background-color: #e0e0e0;
                            color: #2c3e50;
                            text-align: center;
                        }
                        .styled-table th {
                            padding: 8px;
                            font-size: 14px;
                            border: 1px solid #ddd;
                            font-weight: bold;
                            border-bottom: 1px solid #ccc;
                        }
                        .styled-table td {
                            padding: 15px;
                            font-size: 14px;
                            border: 1px solid #ddd;
                            text-align: center;
                        }
                        .styled-table tbody tr {
                            border-bottom: 1px solid #ddd;
                        }
                        .styled-table tbody tr:nth-of-type(even) {
                            background-color: #f9f9f9;
                        }
                        .styled-table tbody tr:hover {
                            background-color: #f0f0f0;
                        }
                    </style>
                    """
                    yield gr.HTML(html_result)
                else:
                    # If no prediction data found, display raw output
                    yield gr.HTML(f"""
                    <div style='text-align:center; background-color: white; padding: 30px; border-radius: 8px;'>
                        <h2 style='margin-bottom: 20px;'>Prediction Completed</h2>
                        <p>No prediction results found in output.</p>
                        <div style='text-align:left; max-height: 400px; overflow-y: auto; background-color: white; padding: 10px; border: 1px solid #dddddd; border-radius: 5px;'>
                            <pre>{result_output}</pre>
                        </div>
                    </div>
                    """)
            else:
                # 更新状态
                status_info["status"] = "failed"
                status_info["current_step"] = "Prediction failed"
                yield generate_status_html(status_info)
                
                stderr_output = ""
                if hasattr(current_process, 'stderr') and current_process.stderr:
                    stderr_output = current_process.stderr.read()
                yield gr.HTML(f"""
                <div style='text-align:center; background-color: white; padding: 30px; border-radius: 8px;'>
                    <h2 style='margin-bottom: 20px;'>Prediction Failed</h2>
                    <p>Error code: {current_process.returncode}</p>
                    <div style='text-align:left; max-height: 400px; overflow-y: auto; background-color: white; padding: 10px; border: 1px solid #dddddd; border-radius: 5px;'>
                        <pre>{stderr_output}\n{result_output}</pre>
                    </div>
                </div>
                """)
        except Exception as e:
            # 更新状态
            status_info["status"] = "failed"
            status_info["current_step"] = "Error occurred"
            yield generate_status_html(status_info)
            
            yield gr.HTML(f"""
            <div style='text-align:center; background-color: white; padding: 30px; border-radius: 8px;'>
                <h2 style='margin-bottom: 20px;'>Error</h2>
                <p>{str(e)}</p>
                <div style='text-align:left; max-height: 400px; overflow-y: auto; background-color: white; padding: 10px; border: 1px solid #dddddd; border-radius: 5px;'>
                    <pre>{traceback.format_exc()}</pre>
                </div>
            </div>
            """)
        finally:
            if current_process:
                stop_thread = True
                is_predicting = False
                current_process = None

    def predict_batch(plm_model, model_path, eval_method, input_file, eval_structure_seq, pooling_method, problem_type, num_labels, batch_size):
        """Batch predict multiple protein sequences"""
        nonlocal is_predicting, current_process, stop_thread
        
        if is_predicting:
            return gr.HTML("""
            <div class="error-container">
                <div class="error-icon">⚠️</div>
                <div class="error-message">Prediction is already in progress. Please wait...</div>
            </div>
            """), None
        
        is_predicting = True
        stop_thread = False
        
        # Initialize progress tracking
        progress_info = {
            "total": 0,
            "completed": 0,
            "current_step": "Initializing",
            "status": "running"
        }
        
        yield generate_progress_html(progress_info), None
        
        try:
            # Validate inputs
            if not model_path:
                is_predicting = False
                yield gr.HTML("""
                <div class="error-container">
                    <div class="error-icon">❌</div>
                    <div class="error-message">Error: Model path is required</div>
                </div>
                """), None
                return
                
            if not os.path.exists(os.path.dirname(model_path)):
                is_predicting = False
                yield gr.HTML("""
                <div class="error-container">
                    <div class="error-icon">❌</div>
                    <div class="error-message">Error: Invalid model path - directory does not exist</div>
                </div>
                """), None
                return
            
            if not input_file:
                is_predicting = False
                yield gr.HTML("""
                <div class="error-container">
                    <div class="error-icon">❌</div>
                    <div class="error-message">Error: Input file is required</div>
                </div>
                """), None
                return
            
            # Update progress
            progress_info["current_step"] = "Preparing input file"
            yield generate_progress_html(progress_info), None
            
            # Create temporary file to save uploaded file
            temp_dir = tempfile.mkdtemp()
            input_path = os.path.join(temp_dir, "input.csv")
            output_path = os.path.join(temp_dir, "predictions.csv")
            
            # Save uploaded file
            try:
                with open(input_path, "wb") as f:
                    # 修复文件上传错误，正确处理gradio上传的文件
                    if hasattr(input_file, "name"):
                        # 如果是NamedString对象，读取文件内容
                        with open(input_file.name, "rb") as uploaded:
                            f.write(uploaded.read())
                    else:
                        # 如果是bytes对象，直接写入
                        f.write(input_file)
                
                # Verify file was saved correctly
                if not os.path.exists(input_path):
                    is_predicting = False
                    yield gr.HTML("""
                    <div class="error-container">
                        <div class="error-icon">❌</div>
                        <div class="error-message">Error: Failed to save input file</div>
                    </div>
                    """), None
                    progress_info["status"] = "failed"
                    progress_info["current_step"] = "Failed to save input file"
                    return
                
                # Count sequences in input file
                try:
                    df = pd.read_csv(input_path)
                    progress_info["total"] = len(df)
                    progress_info["current_step"] = f"Found {len(df)} sequences to process"
                    yield generate_progress_html(progress_info), None
                except Exception as e:
                    is_predicting = False
                    yield gr.HTML(f"""
                    <div class="error-container">
                        <div class="error-icon">❌</div>
                        <div class="error-message">Error reading CSV file: {str(e)}</div>
                    </div>
                    """), None
                    progress_info["status"] = "failed"
                    progress_info["current_step"] = "Error reading CSV file"
                    return
                
            except Exception as e:
                is_predicting = False
                yield gr.HTML(f"""
                <div class="error-container">
                    <div class="error-icon">❌</div>
                    <div class="error-message">Error saving input file: {str(e)}</div>
                </div>
                """), None
                progress_info["status"] = "failed"
                progress_info["current_step"] = "Failed to save input file"
                return
            
            # Update progress
            progress_info["current_step"] = "Preparing model and parameters"
            yield generate_progress_html(progress_info), None
            
            # Prepare command
            args_dict = {
                "model_path": model_path,
                "plm_model": plm_models[plm_model],
                "input_file": input_path,
                "output_file": output_path,
                "pooling_method": pooling_method,
                "problem_type": problem_type,
                "num_labels": num_labels,
                "eval_method": eval_method,
                "batch_size": batch_size
            }
            
            if eval_method == "ses-adapter":
                args_dict["structure_seq"] = ",".join(eval_structure_seq) if eval_structure_seq else None
            else:
                args_dict["structure_seq"] = None
            
            # Build command line
            final_cmd = [sys.executable, "src/predict_batch.py"]
            for k, v in args_dict.items():
                if v is True:
                    final_cmd.append(f"--{k}")
                elif v is not False and v is not None:
                    final_cmd.append(f"--{k}")
                    final_cmd.append(str(v))
            
            # Update progress
            progress_info["current_step"] = "Starting batch prediction process"
            yield generate_progress_html(progress_info), None
            
            # Start prediction process
            try:
                current_process = subprocess.Popen(
                    final_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    universal_newlines=True,
                    preexec_fn=os.setsid
                )
            except Exception as e:
                is_predicting = False
                yield gr.HTML(f"""
                <div class="error-container">
                    <div class="error-icon">❌</div>
                    <div class="error-message">Error starting prediction process: {str(e)}</div>
                </div>
                """), None
                return
            
            output_thread = threading.Thread(target=process_output, args=(current_process, output_queue))
            output_thread.daemon = True
            output_thread.start()
            
            # Collect output
            result_output = ""
            log_output = []
            
            while current_process.poll() is None:
                try:
                    while not output_queue.empty():
                        line = output_queue.get_nowait()
                        result_output += line + "\n"
                        log_output.append(line)
                        
                        # Update progress based on output
                        if "Predicting:" in line:
                            try:
                                # Extract progress from tqdm output
                                match = re.search(r'(\d+)/(\d+)', line)
                                if match:
                                    current, total = map(int, match.groups())
                                    progress_info["completed"] = current
                                    progress_info["total"] = total
                                    progress_info["current_step"] = f"Processing sequence {current}/{total}"
                            except:
                                pass
                        elif "Loading Model and Tokenizer" in line:
                            progress_info["current_step"] = "Loading model and tokenizer"
                        elif "Processing sequences" in line:
                            progress_info["current_step"] = "Processing sequences"
                        elif "Saving results" in line:
                            progress_info["current_step"] = "Saving results"
                        
                        # Update progress display
                        yield generate_progress_html(progress_info), None
                        
                    time.sleep(0.1)
                except Exception as e:
                    yield gr.HTML(f"""
                    <div class="warning-container">
                        <div class="warning-icon">⚠️</div>
                        <div class="warning-message">Warning reading output: {str(e)}</div>
                    </div>
                    """), None
            
            # Process has completed
            if current_process.returncode == 0:
                progress_info["status"] = "completed"
                progress_info["current_step"] = "Prediction completed successfully"
                progress_info["completed"] = progress_info["total"]
                yield generate_progress_html(progress_info), None
                
                # Read prediction results
                if os.path.exists(output_path):
                    try:
                        df = pd.read_csv(output_path)
                        
                        # Create summary statistics based on problem type
                        summary_html = ""
                        if problem_type == "regression":
                            summary_html = f"""
                            <div class="summary-stats">
                                <div class="stat-item">
                                    <div class="stat-value">{len(df)}</div>
                                    <div class="stat-label">Predictions</div>
                                </div>
                                <div class="stat-item">
                                    <div class="stat-value">{df['prediction'].mean():.4f}</div>
                                    <div class="stat-label">Mean Value</div>
                                </div>
                                <div class="stat-item">
                                    <div class="stat-value">{df['prediction'].min():.4f}</div>
                                    <div class="stat-label">Min Value</div>
                                </div>
                                <div class="stat-item">
                                    <div class="stat-value">{df['prediction'].max():.4f}</div>
                                    <div class="stat-label">Max Value</div>
                                </div>
                            </div>
                            """
                        elif problem_type == "single_label_classification":
                            if 'predicted_class' in df.columns:
                                class_counts = df['predicted_class'].value_counts()
                                class_stats = "".join([
                                    f"""
                                    <div class="stat-item">
                                        <div class="stat-value">{count}</div>
                                        <div class="stat-label">Class {class_label}</div>
                                    </div>
                                    """
                                    for class_label, count in class_counts.items()
                                ])
                                
                                summary_html = f"""
                                <div class="summary-stats">
                                    <div class="stat-item">
                                        <div class="stat-value">{len(df)}</div>
                                        <div class="stat-label">Predictions</div>
                                    </div>
                                    {class_stats}
                                </div>
                                """
                        elif problem_type == "multi_label_classification":
                            label_cols = [col for col in df.columns if col.startswith('label_') and not col.endswith('_prob')]
                            if label_cols:
                                label_stats = "".join([
                                    f"""
                                    <div class="stat-item">
                                        <div class="stat-value">{df[col].sum()}</div>
                                        <div class="stat-label">{col}</div>
                                    </div>
                                    """
                                    for col in label_cols
                                ])
                                
                                summary_html = f"""
                                <div class="summary-stats">
                                    <div class="stat-item">
                                        <div class="stat-value">{len(df)}</div>
                                        <div class="stat-label">Predictions</div>
                                    </div>
                                    {label_stats}
                                </div>
                                """
                        
                        # Create HTML table with styling - 使用与dataset preview一致的表格样式
                        html_table = f"""
                        <div class="results-container">
                            <h2>Batch Prediction Results</h2>
                            {summary_html}
                            <div class="table-wrapper">
                                <table class="dataset-preview-table">
                                    <thead>
                                        <tr>
                                            {' '.join([f'<th>{col}</th>' for col in df.columns])}
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {generate_table_rows(df)}
                                    </tbody>
                                </table>
                            </div>
                            <div class="download-hint">
                                <p>You can download the complete results using the download button below.</p>
                            </div>
                        </div>
                        """
                        
                        # 修改CSS样式，确保与eval_tab完全一致
                        final_html = f"""
                        {html_table}
                        <style>
                            .results-container {{
                                background-color: white;
                                border-radius: 8px;
                                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
                                padding: 20px;
                                margin-bottom: 20px;
                            }}
                            
                            .results-container h2 {{
                                color: #2c3e50;
                                text-align: center;
                                margin-bottom: 20px;
                                font-size: 20px;
                            }}
                            
                            .results-container h3 {{
                                color: #2c3e50;
                                text-align: center;
                                margin-bottom: 15px;
                                font-size: 18px;
                            }}
                            
                            .summary-stats {{
                                display: flex;
                                flex-wrap: wrap;
                                justify-content: center;
                                gap: 15px;
                                margin-bottom: 25px;
                            }}
                            
                            .stat-item {{
                                background-color: #f8f9fa;
                                border-radius: 6px;
                                padding: 12px;
                                min-width: 100px;
                                text-align: center;
                                box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
                            }}
                            
                            .stat-value {{
                                font-size: 20px;
                                font-weight: bold;
                                color: #3498db;
                                margin-bottom: 4px;
                            }}
                            
                            .stat-label {{
                                font-size: 12px;
                                color: #7f8c8d;
                            }}
                            
                            .table-wrapper {{
                                max-height: 400px;
                                overflow-y: auto;
                                margin-bottom: 15px;
                                border: 1px solid #ddd;
                                border-radius: 6px;
                            }}
                            
                            /* 完全匹配eval_tab中的dataset-preview-table样式 */
                            .dataset-preview-table {{
                                width: 100%;
                                border-collapse: collapse;
                                font-size: 14px;
                                background-color: white;
                            }}
                            
                            .dataset-preview-table th {{
                                padding: 8px;
                                font-size: 14px;
                                border: 1px solid #ddd;
                                background-color: #e0e0e0;
                                font-weight: bold;
                                border-bottom: 1px solid #ccc;
                                text-align: center;
                            }}
                            
                            .dataset-preview-table td {{
                                padding: 15px;
                                font-size: 14px;
                                border: 1px solid #ddd;
                                text-align: center;
                            }}
                            
                            .dataset-preview-table tr:nth-child(even) {{
                                background-color: #f9f9f9;
                            }}
                            
                            .dataset-preview-table tr:hover {{
                                background-color: #f0f0f0;
                            }}
                            
                            .download-hint {{
                                text-align: center;
                                color: #7f8c8d;
                                font-style: italic;
                                margin-top: 10px;
                                font-size: 12px;
                            }}
                        </style>
                        """
                        
                        # Return results and download link
                        yield gr.HTML(final_html), gr.DownloadButton(value=output_path, label="Download Predictions", visible=True)
                    except Exception as e:
                        yield gr.HTML(f"""
                        <div class="error-container">
                            <div class="error-icon">❌</div>
                            <div class="error-message">Could not read prediction results: {str(e)}</div>
                            <div class="error-details">
                                <pre>{traceback.format_exc()}</pre>
                            </div>
                        </div>
                        <style>
                            .error-container {{
                                background-color: #fff5f5;
                                border-left: 5px solid #e53e3e;
                                padding: 20px;
                                border-radius: 5px;
                                margin: 20px 0;
                            }}
                            .error-icon {{
                                font-size: 24px;
                                margin-bottom: 10px;
                            }}
                            .error-message {{
                                font-weight: bold;
                                margin-bottom: 10px;
                            }}
                            .error-details {{
                                background-color: #f8f9fa;
                                padding: 10px;
                                border-radius: 5px;
                                max-height: 200px;
                                overflow-y: auto;
                            }}
                        </style>
                        """), None
                        
                else:
                    yield gr.HTML(f"""
                    <div class="warning-container">
                        <div class="warning-icon">⚠️</div>
                        <div class="warning-message">Prediction completed but no output file was generated.</div>
                        <div class="warning-details">
                            <pre>{result_output}</pre>
                        </div>
                    </div>
                    <style>
                        .warning-container {{
                            background-color: #fffbea;
                            border-left: 5px solid #ecc94b;
                            padding: 20px;
                            border-radius: 5px;
                            margin: 20px 0;
                        }}
                        .warning-icon {{
                            font-size: 24px;
                            margin-bottom: 10px;
                        }}
                        .warning-message {{
                            font-weight: bold;
                            margin-bottom: 10px;
                        }}
                        .warning-details {{
                            background-color: #f8f9fa;
                            padding: 10px;
                            border-radius: 5px;
                            max-height: 200px;
                            overflow-y: auto;
                        }}
                    </style>
                    """), None
            else:
                progress_info["status"] = "failed"
                progress_info["current_step"] = "Prediction failed"
                yield generate_progress_html(progress_info), None
                
                stderr_output = ""
                if hasattr(current_process, 'stderr') and current_process.stderr:
                    stderr_output = current_process.stderr.read()
                    
                yield gr.HTML(f"""
                <div class="error-container">
                    <div class="error-icon">❌</div>
                    <div class="error-message">Prediction Failed (Error code: {current_process.returncode})</div>
                    <div class="error-details">
                        <pre>{stderr_output}\n{result_output}</pre>
                    </div>
                </div>
                <style>
                    .error-container {{
                        background-color: #fff5f5;
                        border-left: 5px solid #e53e3e;
                        padding: 20px;
                        border-radius: 5px;
                        margin: 20px 0;
                    }}
                    .error-icon {{
                        font-size: 24px;
                        margin-bottom: 10px;
                    }}
                    .error-message {{
                        font-weight: bold;
                        margin-bottom: 10px;
                    }}
                    .error-details {{
                        background-color: #f8f9fa;
                        padding: 10px;
                        border-radius: 5px;
                        max-height: 300px;
                        overflow-y: auto;
                    }}
                </style>
                """), None

        except Exception as e:
            progress_info["status"] = "failed"
            progress_info["current_step"] = "Error occurred"
            yield generate_progress_html(progress_info), None
            
            yield gr.HTML(f"""
            <div class="error-container">
                <div class="error-icon">❌</div>
                <div class="error-message">Error: {str(e)}</div>
                <div class="error-details">
                    <pre>{traceback.format_exc()}</pre>
                </div>
            </div>
            <style>
                .error-container {{
                    background-color: #fff5f5;
                    border-left: 5px solid #e53e3e;
                    padding: 20px;
                    border-radius: 5px;
                    margin: 20px 0;
                }}
                .error-icon {{
                    font-size: 24px;
                    margin-bottom: 10px;
                }}
                .error-message {{
                    font-weight: bold;
                    margin-bottom: 10px;
                }}
                .error-details {{
                    background-color: #f8f9fa;
                    padding: 10px;
                    border-radius: 5px;
                    max-height: 300px;
                    overflow-y: auto;
                }}
            </style>
            """), None
        finally:
            if current_process:
                stop_thread = True
                is_predicting = False
                current_process = None

    def generate_progress_html(progress_info):
        """Generate HTML progress bar similar to eval_tab"""
        current = progress_info.get("completed", 0)
        total = max(progress_info.get("total", 1), 1)  # Avoid division by zero
        percentage = min(100, int((current / total) * 100))
        stage = progress_info.get("current_step", "Preparing")
        
        # 确保进度在0-100之间
        percentage = max(0, min(100, percentage))
        
        # 准备详细信息
        details = []
        if total > 0:
            details.append(f"Total sequences: {total}")
        if current > 0 and total > 0:
            details.append(f"Current progress: {current}/{total}")
        
        details_text = ", ".join(details)
        
        # 创建更现代化的进度条 - 完全匹配eval_tab的样式
        return f"""
        <div style="background-color: #f8f9fa; border-radius: 10px; padding: 20px; margin-bottom: 15px; box-shadow: 0 2px 5px rgba(0,0,0,0.05);">
            <div style="display: flex; justify-content: space-between; margin-bottom: 12px;">
                <div>
                    <span style="font-weight: 600; font-size: 16px;">Prediction Status: </span>
                    <span style="color: #1976d2; font-weight: 500; font-size: 16px;">{stage}</span>
                </div>
                <div>
                    <span style="font-weight: 600; color: #333;">{percentage:.1f}%</span>
                </div>
            </div>
            
            <div style="margin-bottom: 15px; background-color: #e9ecef; height: 10px; border-radius: 5px; overflow: hidden;">
                <div style="background-color: #4285f4; width: {percentage}%; height: 100%; border-radius: 5px; transition: width 0.3s ease;"></div>
            </div>
            
            <div style="display: flex; flex-wrap: wrap; gap: 10px; font-size: 14px; color: #555;">
                {f'<div style="background-color: #e3f2fd; padding: 5px 10px; border-radius: 4px;"><span style="font-weight: 500;">Total sequences:</span> {total}</div>' if total > 0 else ''}
                {f'<div style="background-color: #e8f5e9; padding: 5px 10px; border-radius: 4px;"><span style="font-weight: 500;">Progress:</span> {current}/{total}</div>' if current > 0 and total > 0 else ''}
                {f'<div style="background-color: #fff8e1; padding: 5px 10px; border-radius: 4px;"><span style="font-weight: 500;">Status:</span> {progress_info.get("status", "").capitalize()}</div>' if "status" in progress_info else ''}
            </div>
        </div>
        """

    def generate_table_rows(df, max_rows=100):
        """生成表格行HTML，处理序列数据的特殊显示，与eval_tab的表格样式保持一致"""
        rows = []
        for i, row in df.iterrows():
            if i >= max_rows:
                break
            
            cells = []
            for col in df.columns:
                value = row[col]
                # 对序列类型的列进行特殊处理
                if col in ['aa_seq', 'foldseek_seq', 'ss8_seq'] and isinstance(value, str) and len(value) > 30:
                    # 添加title属性以便鼠标悬停时显示完整序列
                    cell = f'<td title="{value}" style="padding: 15px; font-size: 14px; border: 1px solid #ddd; font-family: monospace; text-align: center;">{value[:30]}...</td>'
                else:
                    cell = f'<td style="padding: 15px; font-size: 14px; border: 1px solid #ddd; text-align: center;">{value}</td>'
                cells.append(cell)
            
            # 添加交替行背景色
            bg_color = "#f9f9f9" if i % 2 == 1 else "white"
            rows.append(f'<tr style="background-color: {bg_color};">{" ".join(cells)}</tr>')
        
        if len(df) > max_rows:
            cols_count = len(df.columns)
            rows.append(f'<tr><td colspan="{cols_count}" style="text-align:center; font-style:italic; padding: 15px; font-size: 14px; border: 1px solid #ddd;">Showing {max_rows} of {len(df)} rows</td></tr>')
        
        return '\n'.join(rows)

    def handle_abort():
        global is_predicting, current_process
        if not is_predicting or current_process is None:
            return gr.HTML("""
            <div class="info-container">
                <div class="info-icon">ℹ️</div>
                <div class="info-message">No prediction process is currently running.</div>
            </div>
            <style>
                .info-container {
                    background-color: #ebf8ff;
                    border-left: 5px solid #4299e1;
                    padding: 15px;
                    border-radius: 5px;
                    margin: 10px 0;
                }
                .info-icon {
                    font-size: 20px;
                    margin-bottom: 8px;
                }
                .info-message {
                    font-weight: 500;
                }
            </style>
            """)
        
        try:
            # Kill the process group
            if hasattr(os, "killpg"):
                os.killpg(os.getpgid(current_process.pid), signal.SIGTERM)
            else:
                current_process.terminate()
            
            is_predicting = False
            current_process = None
            
            return gr.HTML("""
            <div class="warning-container">
                <div class="warning-icon">⚠️</div>
                <div class="warning-message">Prediction process has been aborted.</div>
            </div>
            <style>
                .warning-container {
                    background-color: #fffbea;
                    border-left: 5px solid #ecc94b;
                    padding: 15px;
                    border-radius: 5px;
                    margin: 10px 0;
                }
                .warning-icon {
                    font-size: 20px;
                    margin-bottom: 8px;
                }
                .warning-message {
                    font-weight: 500;
                }
            </style>
            """)
        except Exception as e:
            return gr.HTML(f"""
            <div class="error-container">
                <div class="error-icon">❌</div>
                <div class="error-message">Error aborting prediction: {str(e)}</div>
            </div>"""+"""
            <style>
                .error-container {
                    background-color: #fff5f5;
                    border-left: 5px solid #f56565;
                    padding: 15px;
                    border-radius: 5px;
                    margin: 10px 0;
                }
                .error-icon {
                    font-size: 20px;
                    margin-bottom: 8px;
                }
                .error-message {
                    font-weight: 500;
                }
            </style>
            """)

    with gr.Tab("Prediction"):
        with gr.Row():
            with gr.Column():
                gr.Markdown("## Protein Function Prediction")

        gr.Markdown("### Model Configuration")
        with gr.Group():

            with gr.Row():
                model_path = gr.Textbox(
                    label="Model Path",
                    placeholder="Path to the trained model"
                )
                plm_model = gr.Dropdown(
                    choices=list(plm_models.keys()),
                    label="Protein Language Model"
                )

            with gr.Row():
                eval_method = gr.Dropdown(
                    choices=["full", "freeze", "ses-adapter", "plm-lora", "plm-qlora"],
                    label="Evaluation Method",
                    value="freeze"
                )
                pooling_method = gr.Dropdown(
                    choices=["mean", "attention1d", "light_attention"],
                    label="Pooling Method",
                    value="mean"
                )

            # Settings for different training methods
            with gr.Row(visible=False) as structure_seq_row:
                structure_seq = gr.Dropdown(
                    choices=["foldseek_seq", "ss8_seq"],
                    label="Structure Sequences",
                    multiselect=True,
                    value=["foldseek_seq", "ss8_seq"],
                    info="Select the structure sequences to use for prediction"
                )

            
            with gr.Row():
                problem_type = gr.Dropdown(
                    choices=["single_label_classification", "multi_label_classification", "regression"],
                    label="Problem Type",
                    value="single_label_classification"
                )
                num_labels = gr.Number(
                    value=2,
                    label="Number of Labels",
                    precision=0,
                    minimum=1
                )

        with gr.Tabs():
            with gr.Tab("Sequence Prediction"):
                gr.Markdown("### Input Sequences")
                with gr.Row():
                    aa_seq = gr.Textbox(
                        label="Amino Acid Sequence",
                        placeholder="Enter protein sequence",
                        lines=3
                    )
                # Put the structure input rows in a row with controllable visibility    
                with gr.Row(visible=False) as structure_input_row:
                    foldseek_seq = gr.Textbox(
                        label="Foldseek Sequence",
                        placeholder="Enter foldseek sequence if available",
                        lines=3
                    )
                    ss8_seq = gr.Textbox(
                        label="SS8 Sequence",
                        placeholder="Enter secondary structure sequence if available",
                        lines=3
                    )
                
                with gr.Row():
                    predict_button = gr.Button("Predict", variant="primary")
                    abort_button = gr.Button("Abort", variant="stop")
                
                predict_output = gr.HTML(label="Prediction Results")
                
                # Connect buttons to functions
                predict_button.click(
                    fn=predict_sequence,
                    inputs=[
                        plm_model,
                        model_path,
                        eval_method,
                        aa_seq,
                        foldseek_seq,
                        ss8_seq,
                        structure_seq,
                        pooling_method,
                        problem_type,
                        num_labels
                    ],
                    outputs=predict_output
                )
                
                abort_button.click(
                    fn=handle_abort,
                    inputs=[],
                    outputs=predict_output
                )
            
            with gr.Tab("Batch Prediction"):
                gr.Markdown("### Batch Prediction")
                # Display CSV format information with improved styling
                gr.HTML("""
                <div class="csv-format-info">
                    <h4>CSV File Format Requirements</h4>
                    <p class="format-description">Please prepare your input CSV file with the following columns:</p>
                    <div class="csv-columns">
                        <div class="column-item required">
                            <div class="column-name">aa_seq (required)</div>
                            <div class="column-desc">Amino acid sequence</div>
                        </div>
                        <div class="column-item optional">
                            <div class="column-name">id (optional)</div>
                            <div class="column-desc">Unique identifier for each sequence</div>
                        </div>
                        <div class="column-item optional">
                            <div class="column-name">foldseek_seq (optional)</div>
                            <div class="column-desc">Foldseek structure sequence</div>
                        </div>
                        <div class="column-item optional">
                            <div class="column-name">ss8_seq (optional)</div>
                            <div class="column-desc">Secondary structure sequence</div>
                        </div>
                    </div>
                </div>
                <style>
                    .csv-format-info {
                        background-color: #ffffff;
                        border-radius: 8px;
                        padding: 15px;
                        margin: 0 0 15px 0;
                    }
                    .csv-format-info h4 {
                        margin: 0 0 10px 0;
                        color: #2c3e50;
                        font-size: 16px;
                    }
                    .format-description {
                        margin-bottom: 12px;
                        color: #555;
                        font-size: 14px;
                    }
                    .csv-columns {
                        display: flex;
                        flex-wrap: wrap;
                        gap: 10px;
                        margin-bottom: 10px;
                    }
                    .column-item {
                        background-color: white;
                        border-radius: 6px;
                        padding: 10px;
                        flex: 1 1 200px;
                        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                        transition: transform 0.2s ease;
                    }
                    .column-item:hover {
                        transform: translateY(-2px);
                    }
                    .column-item.required {
                        border-left: 3px solid #3498db;
                    }
                    .column-item.optional {
                        border-left: 3px solid #27ae60;
                    }
                    .column-name {
                        font-family: monospace;
                        font-weight: bold;
                        margin-bottom: 5px;
                        color: #2c3e50;
                        font-size: 14px;
                    }
                    .column-desc {
                        font-size: 13px;
                        color: #7f8c8d;
                        line-height: 1.3;
                    }
                    .csv-example {
                        background-color: #e9ecef;
                        border-radius: 6px;
                        padding: 10px;
                        margin-top: 8px;
                    }
                </style>
                """)
                    
                with gr.Row():
                    input_file = gr.UploadButton(
                        label="Upload CSV File",
                        file_types=[".csv"],
                        file_count="single"
                    )
                
                # File preview accordion
                with gr.Accordion("File Preview", open=False) as file_preview_accordion:
                    # File info area
                    with gr.Row():
                        file_info = gr.HTML("", elem_classes=["dataset-stats"])
                    
                    # Table area
                    with gr.Row():
                        file_preview = gr.Dataframe(
                            headers=["name", "sequence"],
                            value=[["No file uploaded", "-"]],
                            wrap=True,
                            interactive=False,
                            row_count=5,
                            elem_classes=["preview-table"]
                        )
                
                # Add file preview function
                def update_file_preview(file):
                    if file is None:
                        return gr.update(value="<div class='file-info'>No file uploaded</div>"), gr.update(value=[["No file uploaded", "-"]], headers=["name", "sequence"]), gr.update(open=False)
                    try:
                        df = pd.read_csv(file.name)
                        info_html = f"""
                        <div style="text-align: center; margin: 20px 0;">
                            <table style="width: 100%; border-collapse: collapse; margin: 0 auto;">
                                <tr>
                                    <th style="padding: 8px; font-size: 14px; border: 1px solid #ddd; background-color: #e0e0e0; font-weight: bold; border-bottom: 1px solid #ccc; text-align: center;">File</th>
                                    <th style="padding: 8px; font-size: 14px; border: 1px solid #ddd; background-color: #e0e0e0; font-weight: bold; border-bottom: 1px solid #ccc; text-align: center;">Total Sequences</th>
                                    <th style="padding: 8px; font-size: 14px; border: 1px solid #ddd; background-color: #e0e0e0; font-weight: bold; border-bottom: 1px solid #ccc; text-align: center;">Columns</th>
                                </tr>
                                <tr>
                                    <td style="padding: 15px; font-size: 14px; border: 1px solid #ddd; text-align: center;">{file.name.split('/')[-1]}</td>
                                    <td style="padding: 15px; font-size: 14px; border: 1px solid #ddd; text-align: center;">{len(df)}</td>
                                    <td style="padding: 15px; font-size: 14px; border: 1px solid #ddd; text-align: center;">{', '.join(df.columns.tolist())}</td>
                                </tr>
                            </table>
                        </div>
                        """
                        return gr.update(value=info_html), gr.update(value=df.head(5).values.tolist(), headers=df.columns.tolist()), gr.update(open=True)
                    except Exception as e:
                        error_html = f"""
                        <div>
                            <h2>Error reading file</h2>
                            <p style="color: #c62828;">{str(e)}</p>
                        </div>
                        """
                        return gr.update(value=error_html), gr.update(value=[["Error", str(e)]], headers=["Error", "Message"]), gr.update(open=True)
                
                # Use upload event instead of click event
                input_file.upload(
                    fn=update_file_preview,
                    inputs=[input_file],
                    outputs=[file_info, file_preview, file_preview_accordion]
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        batch_size = gr.Slider(
                            minimum=1,
                            maximum=32,
                            value=8,
                            step=1,
                            label="Batch Size",
                            info="Number of sequences to process at once"
                        )
                
                with gr.Row():
                    batch_predict_button = gr.Button("Start Batch Prediction", variant="primary", size="lg")
                    batch_abort_button = gr.Button("Abort", variant="stop", size="lg")
                
                with gr.Row():
                    batch_predict_output = gr.HTML(label="Prediction Progress")
                
                with gr.Row():
                    result_file = gr.DownloadButton(label="Download Predictions", visible=False)
                
                # Connect buttons to functions
                batch_predict_button.click(
                    fn=predict_batch,
                    inputs=[
                        plm_model,
                        model_path,
                        eval_method,
                        input_file,
                        structure_seq,
                        pooling_method,
                        problem_type,
                        num_labels,
                        batch_size
                    ],
                    outputs=[batch_predict_output, result_file]
                )
                
                batch_abort_button.click(
                    fn=handle_abort,
                    inputs=[],
                    outputs=[batch_predict_output]
                )

    # Add this code after all UI components are defined
    def update_eval_method(method):
        return {
            structure_seq_row: gr.update(visible=method == "ses-adapter"),
            structure_input_row: gr.update(visible=method == "ses-adapter")
        }

    eval_method.change(
        fn=update_eval_method,
        inputs=[eval_method],
        outputs=[structure_seq_row, structure_input_row]
    )

    # Add a new function to control the visibility of the structure sequence input boxes
    def update_structure_inputs(structure_seq_choices):
        return {
            foldseek_seq: gr.update(visible="foldseek_seq" in structure_seq_choices),
            ss8_seq: gr.update(visible="ss8_seq" in structure_seq_choices)
        }

    # Add event handling to the UI definition section
    structure_seq.change(
        fn=update_structure_inputs,
        inputs=[structure_seq],
        outputs=[foldseek_seq, ss8_seq]
    )

    return {
        "predict_sequence": predict_sequence,
        "predict_batch": predict_batch,
        "handle_abort": handle_abort
    }