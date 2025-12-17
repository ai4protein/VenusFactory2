import gradio as gr
import json
import os
import subprocess
import sys
import threading
import queue
import time
import pandas as pd
import re
from datasets import load_dataset
from web.utils.command import preview_eval_command
from web.utils.html_ui import load_html_template, format_metrics_table
from web.utils.css_loader import get_css_style_tag

def create_eval_tab(constant):
    plm_models = constant["plm_models"]
    dataset_configs = constant["dataset_configs"]
    is_evaluating = False
    current_process = None
    output_queue = queue.Queue()
    stop_thread = False
    process_aborted = False 
    plm_models = constant["plm_models"]
    

    def format_metrics(metrics_file):
        """Convert metrics to HTML table format for display"""
        try:
            df = pd.read_csv(metrics_file)
            metrics_dict = df.iloc[0].to_dict()
            
            metrics_rows = format_metrics_table(metrics_dict)
            metrics_count = len(metrics_dict)
            
            html = f"""
            {get_css_style_tag('eval_predict_ui.css')}
            {load_html_template('metrics_table.html', 
                              metrics_count=metrics_count,
                              metrics_rows=metrics_rows,
                              completion_time=time.strftime("%Y-%m-%d %H:%M:%S"))}
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

    def evaluate_model(plm_model, model_path, eval_method, is_custom_dataset, dataset_defined, dateset_custom, problem_type, num_labels, metrics, batch_mode, batch_size, batch_token, eval_structure_seq, pooling_method, sequence_column_name, label_column_name):
        nonlocal is_evaluating, current_process, stop_thread, process_aborted
        
        if is_evaluating:
            return "Evaluation is already in progress. Please wait...", gr.update(visible=False)
        
        # First reset all state variables to ensure clean start
        is_evaluating = True
        stop_thread = False
        process_aborted = False  # Reset abort flag
        
        # Clear the output queue
        while not output_queue.empty():
            try:
                output_queue.get_nowait()
            except queue.Empty:
                break
        
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
        
        # Create initial progress bar with completely empty state
        initial_progress_html = generate_progress_bar(progress_info)
        
        yield initial_progress_html, gr.update(visible=False)
        
        try:
            # Validate inputs
            if not model_path or not os.path.exists(os.path.dirname(model_path)):
                is_evaluating = False
                yield f"""
                {get_css_style_tag('eval_predict_ui.css')}
                {load_html_template('evaluation_error.html', error_message="Error: Invalid model path")}
                """, gr.update(visible=False)
                return
            
            if is_custom_dataset == "Custom Dataset":
                dataset = dateset_custom
                test_file = dateset_custom
            else:
                dataset = dataset_defined
                if dataset not in dataset_configs:
                    is_evaluating = False
                    yield f"""
                    {get_css_style_tag('eval_predict_ui.css')}
                    {load_html_template('evaluation_error.html', error_message="Error: Invalid dataset selection")}
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
                "metrics": ",".join(metrics),
                "plm_model": plm_models[plm_model],
                "test_result_dir": test_result_dir,
                "dataset": dataset_display_name,
                "pooling_method": pooling_method,
                "sequence_column_name": sequence_column_name,
                "label_column_name": label_column_name,
            }
            if batch_mode == "Batch Size Mode":
                args_dict["batch_size"] = batch_size
            else:
                args_dict["batch_token"] = batch_token

            if eval_method == "ses-adapter":
                args_dict["structure_seq"] = ",".join(eval_structure_seq) if eval_structure_seq else None
                # Add flags for using foldseek and ss8
                if "foldseek_seq" in eval_structure_seq:
                    args_dict["use_foldseek"] = True
                if "ss8_seq" in eval_structure_seq:
                    args_dict["use_ss8"] = True
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
            
            while True:
                # Check if the process still exists and hasn't been aborted
                if process_aborted or current_process is None or current_process.poll() is not None:
                    break
                
                try:
                    new_lines = []
                    lines_processed = 0
                    while lines_processed < 10:
                        try:
                            line = output_queue.get_nowait()
                            new_lines.append(line)
                            progress_info["lines"].append(line)
                            # print(line)
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
                        # Only yield updates if there's actual new information
                        yield progress_html, gr.update(visible=False)
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
                result_file = os.path.join(test_result_dir, "evaluation_metrics.csv")
                if os.path.exists(result_file):
                    metrics_html = format_metrics(result_file)
                    
                    # Calculate total evaluation time
                    total_time = time.time() - start_time
                    hours, remainder = divmod(int(total_time), 3600)
                    minutes, seconds = divmod(remainder, 60)
                    time_str = f"{hours:02}:{minutes:02}:{seconds:02}"
                    
                    summary_html = f"""
                    {get_css_style_tag('eval_predict_ui.css')}
                    {load_html_template('evaluation_success.html', 
                                      total_time=time_str,
                                      dataset_name=dataset_display_name,
                                      total_samples=progress_info.get('total_samples', 0))}
                    <div class="results-title">Evaluation Results</div>
                    {metrics_html}
                    """
                    
                    # 设置下载按钮可见并指向结果文件
                    yield summary_html, gr.update(value=result_file, visible=True)
                else:
                    error_output = "\n".join(progress_info.get("lines", []))
                    yield f"""
                    {get_css_style_tag('eval_predict_ui.css')}
                    {load_html_template('evaluation_warning.html', warning_message=f"Evaluation completed, but metrics file not found at: {result_file}")}
                    """, gr.update(visible=False)
            else:
                error_output = "\n".join(progress_info.get("lines", []))
                if not error_output:
                    error_output = "No output captured from the evaluation process"
                
                yield f"""
                {get_css_style_tag('eval_predict_ui.css')}
                {load_html_template('evaluation_error.html', 
                                  error_message="Evaluation failed:",
                                  error_details=f"<pre class='error-details'>{error_output}</pre>")}
                """, gr.update(visible=False)

        except Exception as e:
            yield f"""
            {get_css_style_tag('eval_predict_ui.css')}
            {load_html_template('evaluation_error.html', 
                              error_message="Error during evaluation process:",
                              error_details=f"<pre class='error-details'>{str(e)}</pre>")}
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
        total_samples_detail = f'<div class="progress-detail-item progress-detail-total"><span style="font-weight: 500;">Total samples:</span> {total_samples}</div>' if total_samples > 0 else ''
        progress_detail = f'<div class="progress-detail-item progress-detail-current"><span style="font-weight: 500;">Progress:</span> {current}/{total}</div>' if current > 0 and total > 0 else ''
        time_detail = f'<div class="progress-detail-item progress-detail-time"><span style="font-weight: 500;">Time:</span> {progress_info.get("elapsed_time", "")}</div>' if progress_info.get("elapsed_time", "") else ''
        
        # 创建更现代化的进度条
        html = f"""
        {get_css_style_tag('eval_predict_ui.css')}
        {load_html_template('evaluation_progress.html',
                          stage=stage,
                          progress=progress,
                          total_samples_detail=total_samples_detail,
                          progress_detail=progress_detail,
                          time_detail=time_detail)}
        """
        return html

    def handle_eval_tab_abort():
        """Handle abortion of the evaluation process"""
        nonlocal is_evaluating, current_process, stop_thread, process_aborted
        
        if current_process is None:
                    return f"""
        {get_css_style_tag('eval_predict_ui.css')}
        {load_html_template('status_empty.html', message="No evaluation in progress to terminate.")}
        """, gr.update(visible=False)
        
        try:
            # Set the abort flag before terminating the process
            process_aborted = True
            stop_thread = True
            
            # Using terminate instead of killpg for safety
            current_process.terminate()
            
            # Wait for process to terminate (with timeout)
            try:
                current_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                current_process.kill()
            
            # Reset state completely
            current_process = None
            is_evaluating = False
            
            # Reset output queue to clear any pending messages
            while not output_queue.empty():
                try:
                    output_queue.get_nowait()
                except queue.Empty:
                    break
            
            return f"""
            {get_css_style_tag('eval_predict_ui.css')}
            {load_html_template('status_success.html', message="Evaluation successfully terminated! All evaluation state has been reset.")}
            """, gr.update(visible=False)
        except Exception as e:
            # Still need to reset states even if there's an error
            current_process = None
            is_evaluating = False
            process_aborted = False
            
            # Reset output queue
            while not output_queue.empty():
                try:
                    output_queue.get_nowait()
                except queue.Empty:
                    break
            
            return f"""
            {get_css_style_tag('eval_predict_ui.css')}
            {load_html_template('status_failed_terminate.html', error_message=f"Failed to terminate evaluation: {str(e)}")}
            """, gr.update(visible=False)
            
    gr.Markdown("## Model and Dataset Configuration")
    with gr.Accordion("Import Training Configuration", open=False) as config_import_accordion:
        gr.Markdown("### Import your training config")
        with gr.Row():
            with gr.Column(scale=4):
                config_path_input = gr.Textbox(
                    label="Configuration File Path",
                    placeholder="Enter path to your training config JSON file (e.g., ./config.json)",
                    value=""
                )
            with gr.Column(scale=1):
                import_config_button = gr.Button(
                    "Import Config",
                    variant="primary",
                    elem_classes=["import-config-btn"]
                )
    # Original evaluation interface components
    with gr.Group():
        with gr.Row():
            eval_model_path = gr.Textbox(
                label="Model Path",
                value="ckpt/demo/demo_solubility.pt",
                placeholder="Path to the trained model"
            )
            eval_plm_model = gr.Dropdown(
                choices=list(plm_models.keys()),
                label="Protein Language Model"
            )

        with gr.Row():
            eval_method = gr.Dropdown(
                choices=["full", "freeze", "ses-adapter", "plm-lora", "plm-qlora", "plm-adalora", "plm-dora", "plm-ia3"],
                label="Evaluation Method",
                value="freeze"
            )
            eval_pooling_method = gr.Dropdown(
                choices=["mean", "attention1d", "light_attention"],
                label="Pooling Method",
                value="mean"
            )
        with gr.Row():
            with gr.Column(scale=4):
                with gr.Row():
                    is_custom_dataset = gr.Radio(
                        choices=["Custom Dataset", "Pre-defined Dataset"],
                        label="Dataset Selection",
                        value="Pre-defined Dataset"
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

            with gr.Column(scale=1, min_width=120, elem_classes="preview-button-container"):
                # Add dataset preview functionality
                preview_button = gr.Button(
                    "Preview Dataset", 
                    variant="primary", 
                    size="lg",
                    elem_classes="preview-button"
                )
                
            # These are settings for custom dataset.
        with gr.Row():
            problem_type = gr.Dropdown(
                choices=["single_label_classification", "multi_label_classification", "regression", "residue_single_label_classification", "residue_regression"],
                label="Problem Type",
                value="single_label_classification",
                scale=12,
                interactive=False   
            )
            num_labels = gr.Number(
                value=2,
                label="Number of Labels",
                scale=11,
                interactive=False
            )
            metrics = gr.Dropdown(
                choices=["accuracy", "recall", "precision", "f1", "mcc", "auroc", "aupr", "f1_max", "f1_positive", "f1_negative", "spearman_corr", "mse"],
                label="Metrics",
                value=["accuracy", "mcc", "f1", "precision", "recall", "auroc"],
                scale=30,
                multiselect=True,
                interactive=False
            )
            sequence_column_name = gr.Textbox(
                label="Amino Acid Sequence Column Name",
                value="aa_seq",
                placeholder="Name of the amino acid sequence column in dataset",
                scale=20,
                interactive=False
            )
            label_column_name = gr.Textbox(
                label="Target Label Column Name",
                value="label",
                placeholder="Name of the target label column in dataset",
                scale=20,
                interactive=False
            )
                
        
    # put dataset preview in a accordion
    with gr.Row():
        with gr.Accordion("Dataset Preview", open=False) as preview_accordion:
            # dataset stats
            with gr.Row():
                dataset_stats_md = gr.HTML("", elem_classes=["dataset-stats"])
            
            # dataset preview table
            with gr.Row():
                preview_table = gr.Dataframe(
                    headers=["Name", "Sequence", "Label"],
                    value=[["No dataset selected", "-", "-"]],
                    wrap=True,
                    interactive=False,
                    row_count=3,
                    elem_classes=["preview-table"]
                )
   
    
    def update_eval_tab_dataset_preview_UI(dataset_type=None, dataset_name=None, custom_dataset=None):
        """Update dataset preview content for Gradio UI
        Args:
            dataset_type: dataset type (Custom Dataset or Pre-defined Dataset)
            dataset_name: predefined dataset name
            custom_dataset: custom dataset path
        Returns:
        """
        # Determine which dataset to use based on selection
        if dataset_type == "Custom Dataset" and custom_dataset:
            try:
                # Try to load custom dataset
                dataset = load_dataset(custom_dataset)
                stats_html = load_html_template(
                    "dataset_stats_table.html", 
                    dataset_name=custom_dataset, 
                    train_count=len(dataset["train"]) if "train" in dataset else 0, 
                    val_count=len(dataset["validation"]) if "validation" in dataset else 0, 
                    test_count=len(dataset["test"]) if "test" in dataset else 0
                    )
                
                # Get sample data points
                split = "train" if "train" in dataset else list(dataset.keys())[0]
                samples = dataset[split].select(range(min(3, len(dataset[split]))))
                if len(samples) == 0:
                    return gr.update(value=stats_html), gr.update(value=[["No data available", "-", "-"]], headers=["Name", "Sequence", "Label"]), gr.update(open=True)
                
                # Get fields actually present in the dataset
                available_fields = list(samples[0].keys())
                
                # Build sample data - ensure consistent structure
                sample_data = []
                for i, sample in enumerate(samples):
                    row_data = []
                    for field in available_fields:
                        # Truncate long sequences for display
                        value = str(sample[field])
                        if len(value) > 100:
                            value = value[:100] + "..."
                        row_data.append(value)
                    sample_data.append(row_data)
                
                return gr.update(value=stats_html), gr.update(value=sample_data, headers=available_fields), gr.update(open=True)
            except Exception as e:
                error_html = load_html_template("error_loading_dataset.html", error_message=str(e))
                return gr.update(value=error_html), gr.update(value=[["Error", str(e), "-"]], headers=["Error", "Message", "Status"]), gr.update(open=True)
        
        # Use predefined dataset
        elif dataset_type == "Pre-defined Dataset" and dataset_name:
            try:
                config_path = dataset_configs[dataset_name]
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                # Load dataset statistics
                dataset = load_dataset(config["dataset"])
                stats_html = load_html_template(
                    "dataset_stats_table.html", 
                    dataset_name=config["dataset"], 
                    train_count=len(dataset["train"]) if "train" in dataset else 0, 
                    val_count=len(dataset["validation"]) if "validation" in dataset else 0, 
                    test_count=len(dataset["test"]) if "test" in dataset else 0
                    )
                
                # Get sample data points and available fields
                samples = dataset["train"].select(range(min(3, len(dataset["train"]))))
                if len(samples) == 0:
                    return gr.update(value=stats_html), gr.update(value=[["No data available", "-", "-"]], headers=["Name", "Sequence", "Label"]), gr.update(open=True)
                
                # Get fields actually present in the dataset
                available_fields = list(samples[0].keys())
                
                # Build sample data - ensure consistent structure
                sample_data = []
                for i, sample in enumerate(samples):
                    row_data = []
                    for field in available_fields:
                        # Truncate long sequences for display
                        value = str(sample[field])
                        if len(value) > 100:
                            value = value[:100] + "..."
                        row_data.append(value)
                    sample_data.append(row_data)
                
                return gr.update(value=stats_html), gr.update(value=sample_data, headers=available_fields), gr.update(open=True)
            except Exception as e:
                error_html = load_html_template("error_loading_dataset.html", error_message=str(e))
                return gr.update(value=error_html), gr.update(value=[["Error", str(e), "-"]], headers=["Error", "Message", "Status"]), gr.update(open=True)
        
        # If no valid dataset information provided
        return gr.update(value=""), gr.update(value=[["No dataset selected", "-", "-"]], headers=["Status", "Message", "Action"]), gr.update(open=True)

    
    # Preview button click event
    preview_button.click(
        fn=update_eval_tab_dataset_preview_UI,
        inputs=[is_custom_dataset, eval_dataset_defined, eval_dataset_custom],
        outputs=[dataset_stats_md, preview_table, preview_accordion]
    )

    def update_eval_tab_dataset_settings_UI(choice, dataset_name=None):
        """Update dataset settings for Gradio UI
        Args:
            choice: dataset type (Custom Dataset or Pre-defined Dataset)
            dataset_name: predefined dataset name
        Returns:
        """
        if choice == "Pre-defined Dataset":
            # Load configuration from dataset_config
            if dataset_name and dataset_name in dataset_configs:
                with open(dataset_configs[dataset_name], 'r') as f:
                    config = json.load(f)
                # 处理metrics，将字符串转换为列表以适应多选组件
                metrics_value = config.get("metrics", "accuracy,mcc,f1,precision,recall,auroc")
                if isinstance(metrics_value, str):
                    metrics_value = metrics_value.split(",")
                return [
                    gr.update(visible=True),  # eval_dataset_defined
                    gr.update(visible=False), # eval_dataset_custom
                    gr.update(value=config.get("problem_type", ""), interactive=False),
                    gr.update(value=config.get("num_labels", 1), interactive=False),
                    gr.update(value=metrics_value, interactive=False),
                    gr.update(value=config.get("sequence_column_name", "sequence"), interactive=False),
                    gr.update(value=config.get("label_column_name", "label"), interactive=False)
                ]
        else:
            # Custom dataset settings - make fields editable but don't reset values
            # This allows config import to work correctly
            return [
                gr.update(visible=False),  # eval_dataset_defined
                gr.update(visible=True),   # eval_dataset_custom
                gr.update(interactive=True),  # problem_type
                gr.update(interactive=True),  # num_labels
                gr.update(interactive=True),  # metrics
                gr.update(interactive=True),  # sequence_column_name
                gr.update(interactive=True)   # label_column_name
            ]
    
    is_custom_dataset.change(
        fn=update_eval_tab_dataset_settings_UI,
        inputs=[is_custom_dataset, eval_dataset_defined],
        outputs=[eval_dataset_defined, eval_dataset_custom, 
                problem_type, num_labels, metrics, sequence_column_name, label_column_name]
    )

    def handle_eval_dataset_defined_change(x):
        """Handle evaluation dataset defined change event
        Args:
            x: evaluation dataset defined
        Returns:
            Updated UI components
        """
        return update_eval_tab_dataset_settings_UI("Pre-defined Dataset", x)
    
    eval_dataset_defined.change(
        fn=handle_eval_dataset_defined_change,
        inputs=[eval_dataset_defined],
        outputs=[eval_dataset_defined, eval_dataset_custom, 
                problem_type, num_labels, metrics, sequence_column_name, label_column_name]
    )

    ### These are settings for different training methods. ###

    # for ses-adapter
    with gr.Row(visible=False) as structure_seq_row:
        eval_structure_seq = gr.CheckboxGroup(
            label="Structure Sequence",
            choices=["foldseek_seq", "ss8_seq"],
            value=["foldseek_seq", "ss8_seq"]
        )
                
    def update_structure_seq_visibility_UI(method):
        """Update the ses-adapter structure sequence visibility for Gradio UI
        Args:
            method: training method (ses-adapter)
        Returns:
            structure_seq_row: structure sequence input gr.Row (visible or not)
        """
        return {
            structure_seq_row: gr.update(visible=method == "ses-adapter")
        }

    eval_method.change(
        fn=update_structure_seq_visibility_UI,
        inputs=[eval_method],
        outputs=[structure_seq_row]
    )
    
    gr.Markdown("## Batch Processing Configuration")
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

    def update_eval_tab_batch_inputs_UI(mode):
        """Update batch or token input visibility for Gradio UI
        Args:
            mode: batch mode
        Returns:
            batch_size: batch size input gr.Slider (visible or not)
            batch_token: batch token input gr.Slider (visible or not)
        """
        return {
            batch_size: gr.update(visible=mode == "Batch Size Mode"),
            batch_token: gr.update(visible=mode == "Batch Token Mode")
        }
        
    # Update visibility when mode changes
    batch_mode.change(
        fn=update_eval_tab_batch_inputs_UI,
        inputs=[batch_mode],
        outputs=[batch_size, batch_token]
    )

    with gr.Row():
        preview_button = gr.Button("Preview Command", elem_classes=["preview-command-btn"])
        eval_button = gr.Button("Start Evaluation", variant="primary", elem_classes=["train-btn"])
        abort_button = gr.Button("Abort Evaluation", variant="stop", elem_classes=["abort-btn"])
        

    with gr.Row():
        command_preview = gr.Code(
            label="Command Preview",
            language="shell",
            interactive=False,
            visible=False
        )

    def handle_eval_tab_command_preview(plm_model, model_path, eval_method, is_custom_dataset, dataset_defined, 
                        dataset_custom, problem_type, num_labels, metrics, batch_mode, 
                        batch_size, batch_token, eval_structure_seq, eval_pooling_method, sequence_column_name, label_column_name):
        """Handle the preview command button click event
        Args:
            plm_model: plm model name
            model_path: model path
            eval_method: evaluation method
            is_custom_dataset: whether to Custom Dataset (Custom Dataset or Pre-defined Dataset)
            dataset_defined: dataset name
            dataset_custom: custom dataset path
            problem_type: problem type
            num_labels: number of labels
            metrics: metrics (accuracy, recall, precision, f1, mcc, auroc, aupr, f1_max, f1_positive, f1_negative, spearman_corr, mse)
            batch_mode: batch mode (Batch Size Mode or Batch Token Mode)
            batch_size: batch size
            batch_token: batch token (tokens per batch)
            eval_structure_seq: structure sequence (foldseek_seq, ss8_seq)
            eval_pooling_method: pooling method (mean, attention1d, light_attention)
            sequence_column_name: name of the sequence column in dataset
            label_column_name: name of the label column in dataset
        Returns:
            command_preview: command preview
        """
        if command_preview.visible:
            return gr.update(visible=False)
        
        # create args dictionary
        args = {
            "plm_model": plm_models[plm_model],
            "model_path": model_path,
            "eval_method": eval_method,
            "pooling_method": eval_pooling_method
        }
        
        # process dataset related parameters
        if is_custom_dataset == "Custom Dataset":
            args["dataset"] = dataset_custom
            args["problem_type"] = problem_type
            args["num_labels"] = num_labels
            args["metrics"] = ",".join(metrics)
            args["sequence_column_name"] = sequence_column_name
            args["label_column_name"] = label_column_name
        else:
            with open(dataset_configs[dataset_defined], 'r') as f:
                config = json.load(f)
            args["dataset_config"] = dataset_configs[dataset_defined]
            args["sequence_column_name"] = sequence_column_name
            args["label_column_name"] = label_column_name
        
        # process batch processing parameters
        if batch_mode == "Batch Size Mode":
            args["batch_size"] = batch_size
        else:
            args["batch_token"] = batch_token
        
        # process structure sequence parameters
        if eval_method == "ses-adapter" and eval_structure_seq:
            args["structure_seq"] = ",".join(eval_structure_seq)
            if "foldseek_seq" in eval_structure_seq:
                args["use_foldseek"] = True
            if "ss8_seq" in eval_structure_seq:
                args["use_ss8"] = True
        
        # generate preview command
        preview_text = preview_eval_command(args)
        return gr.update(value=preview_text, visible=True)

    # bind preview button event
    preview_button.click(
        fn=handle_eval_tab_command_preview,
        inputs=[
            eval_plm_model,
            eval_model_path,
            eval_method,
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
            eval_pooling_method,
            sequence_column_name,
            label_column_name
        ],
        outputs=[command_preview]
    )

    eval_output = gr.HTML(
        value="<div style='padding: 15px; background-color: #f5f5f5; border-radius: 5px;'><p style='margin: 0;'>Click the 「Start Evaluation」 button to begin model evaluation</p></div>",
        label="Evaluation Status & Results"
    )

    with gr.Row():
        with gr.Column(scale=4):
            pass
        with gr.Column(scale=1):
            download_csv_btn = gr.DownloadButton(
                "Download CSV", 
                visible=False,
                size="lg"
            )
        with gr.Column(scale=4):
            pass
    
    # Connect buttons to functions
    eval_button.click(
        fn=evaluate_model,
        inputs=[
            eval_plm_model,
            eval_model_path,
            eval_method,
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
            eval_pooling_method,
            sequence_column_name,
            label_column_name
        ],
        outputs=[eval_output, download_csv_btn]
    )
    abort_button.click(
        fn=handle_eval_tab_abort,
        inputs=[],
        outputs=[eval_output, download_csv_btn]
    )

    # Configuration Import Handler
    def handle_config_import(config_path: str):
        """
        Loads an evaluation configuration from a JSON file and updates the UI components.
        """
        try:
            if not config_path or not config_path.strip():
                gr.Warning("Please provide a configuration file path")
                return [gr.update() for _ in range(13)]
                
            if not os.path.exists(config_path):
                gr.Warning(f"Configuration file not found: {config_path}")
                return [gr.update() for _ in range(13)]
            
            with open(config_path, "r", encoding='utf-8') as f:
                config = json.load(f)
            
            def get_config_val(key, default):
                return config.get(key, default)
            
            # Extract and validate model_path
            model_path_value = get_config_val("model_path", "ckpt/demo/demo_solubility.pt")
            
            # Extract and map plm_model (config stores full path, UI uses key)
            plm_model_path = get_config_val("plm_model", "")
            plm_model_value = None
            for key, path in plm_models.items():
                if path == plm_model_path:
                    plm_model_value = key
                    break
            if not plm_model_value and plm_models:
                plm_model_value = list(plm_models.keys())[0]
            
            # Validate eval_method
            eval_method_value = get_config_val("training_method", "freeze")
            valid_eval_methods = ["full", "freeze", "ses-adapter", "plm-lora", "plm-qlora", "plm-adalora", "plm-dora", "plm-ia3"]
            if eval_method_value not in valid_eval_methods:
                eval_method_value = "freeze"
            
            # Validate pooling_method
            pooling_method_value = get_config_val("pooling_method", "mean")
            if pooling_method_value not in ["mean", "attention1d", "light_attention"]:
                pooling_method_value = "mean"
            
            # Set to Custom Dataset and extract dataset path
            dataset_selection_value = "Custom Dataset"
            dataset_custom_value = get_config_val("dataset_custom", "")
            
            # Validate problem_type
            problem_type_value = get_config_val("problem_type", "single_label_classification")
            valid_problem_types = ["single_label_classification", "multi_label_classification", "regression", "residue_single_label_classification", "residue_regression"]
            if problem_type_value not in valid_problem_types:
                problem_type_value = "single_label_classification"
            
            # Extract num_labels
            num_labels_value = get_config_val("num_labels", 2)
            
            # Handle metrics - ensure it's a list and values are valid
            metrics_value = get_config_val("metrics", ["accuracy"])
            if isinstance(metrics_value, str):
                metrics_value = [m.strip() for m in metrics_value.split(",")]
            valid_metrics = ["accuracy", "recall", "precision", "f1", "mcc", "auroc", "aupr", "f1_max", "f1_positive", "f1_negative", "spearman_corr", "mse"]
            metrics_value = [m for m in metrics_value if m in valid_metrics]
            if not metrics_value:
                metrics_value = ["accuracy"]
            
            # Validate batch_mode
            batch_mode_value = get_config_val("batch_mode", "Batch Size Mode")
            if batch_mode_value not in ["Batch Size Mode", "Batch Token Mode"]:
                batch_mode_value = "Batch Size Mode"
            
            # Extract batch_size and batch_token
            batch_size_value = get_config_val("batch_size", 16)
            batch_token_value = get_config_val("batch_token", 10000)
            
            # Extract sequence and label column names
            sequence_column_value = get_config_val("sequence_column_name", "aa_seq")
            label_column_value = get_config_val("label_column_name", "label")
            
            gr.Info(f"✅ Configuration successfully imported from {config_path}")
            
            return [
                gr.update(value=model_path_value),
                gr.update(value=plm_model_value),
                gr.update(value=eval_method_value),
                gr.update(value=pooling_method_value),
                gr.update(value=dataset_selection_value),
                gr.update(visible=False),  # eval_dataset_defined
                gr.update(visible=True, value=dataset_custom_value),  # eval_dataset_custom
                gr.update(value=problem_type_value, interactive=True),
                gr.update(value=num_labels_value, interactive=True),
                gr.update(value=metrics_value, interactive=True),
                gr.update(value=batch_mode_value),
                gr.update(visible=(batch_mode_value == "Batch Size Mode"), value=batch_size_value),
                gr.update(visible=(batch_mode_value == "Batch Token Mode"), value=batch_token_value),
                gr.update(value=sequence_column_value, interactive=True),
                gr.update(value=label_column_value, interactive=True),
            ]
        
        except (json.JSONDecodeError, KeyError) as e:
            gr.Warning(f"❌ Error parsing configuration file: {str(e)}")
            return [gr.update() for _ in range(15)]
        except Exception as e:
            gr.Warning(f"❌ An unexpected error occurred during import: {str(e)}")
            return [gr.update() for _ in range(15)]
    
    # Bind import config button
    import_config_button.click(
        fn=handle_config_import,
        inputs=[config_path_input],
        outputs=[
            eval_model_path,
            eval_plm_model,
            eval_method,
            eval_pooling_method,
            is_custom_dataset,
            eval_dataset_defined,
            eval_dataset_custom,
            problem_type,
            num_labels,
            metrics,
            batch_mode,
            batch_size,
            batch_token,
            sequence_column_name,
            label_column_name
        ]
    )

    return {
        "eval_button": eval_button,
        "eval_output": eval_output
    }