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
import traceback
import re
from pathlib import Path
from web.utils.command import preview_predict_command
from web.utils.html_ui import load_html_template, generate_prediction_status_html, generate_prediction_results_html, generate_batch_prediction_results_html, generate_table_rows
from web.utils.css_loader import get_css_style_tag
from web.utils.common_utils import get_save_path
from datetime import datetime


def _scan_folders_under(root: str, max_depth: int = 5) -> list:
    """Scan root for subdirectories. Returns list of (display_label, full_path)."""
    if not root or not os.path.isdir(root):
        return []
    result = []
    root_path = Path(root)
    for path in sorted(root_path.rglob("*")):
        if not path.is_dir():
            continue
        try:
            rel = path.relative_to(root_path)
        except ValueError:
            continue
        if not rel.parts or len(rel.parts) > max_depth:
            continue
        path_str = str(path)
        result.append((path_str, path_str))
    return result


def _scan_models_in_folder(folder_path: str) -> list:
    """Scan folder for .pt model files. Returns list of (display_label, full_path)."""
    if not folder_path or not os.path.isdir(folder_path):
        return []
    result = []
    folder = Path(folder_path)
    for pt_file in sorted(folder.rglob("*.pt")):
        name = pt_file.stem
        if any(s in name for s in ("_lora", "_qlora", "_dora", "_adalora", "_ia3")):
            continue
        result.append((str(pt_file.relative_to(folder)), str(pt_file)))
    return result


def _load_model_config(model_path: str) -> dict:
    """Load config from model's .json file. Returns dict or empty dict."""
    if not model_path or not str(model_path).endswith(".pt"):
        return {}
    config_path = os.path.join(os.path.dirname(model_path), Path(model_path).stem + ".json")
    try:
        with open(config_path, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def _parse_fasta_sequences(content: str) -> list:
    """Parse multi-FASTA from text. Returns list of (id, sequence) tuples."""
    if not content or not content.strip():
        return []
    lines = content.splitlines()
    records = []
    current_id = ""
    current_seq = []
    for line in lines:
        line = line.rstrip()
        if not line:
            continue
        if line.startswith(">"):
            if current_seq:
                seq = "".join(current_seq).replace(" ", "")
                if seq:
                    records.append((current_id or f"seq{len(records)+1}", seq))
            current_id = line[1:].strip().split()[0] if line[1:].strip() else f"seq{len(records)+1}"
            current_seq = []
        else:
            if line.startswith("#"):
                continue
            current_seq.append(line)
    if current_seq:
        seq = "".join(current_seq).replace(" ", "")
        if seq:
            records.append((current_id or f"seq{len(records)+1}", seq))
    if not records:
        # No FASTA header: treat whole content as one sequence
        seq = "".join(l.strip() for l in lines if l.strip() and not l.strip().startswith("#")).replace(" ", "")
        if seq:
            records = [("seq1", seq)]
    return records


def _parse_fasta_from_file(file_obj) -> list:
    """Read and parse multi-FASTA from an uploaded file. Returns list of (id, sequence) tuples."""
    if file_obj is None:
        return []
    path = file_obj.name if hasattr(file_obj, "name") else str(file_obj)
    if not path or not os.path.isfile(path):
        return []
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
    except Exception:
        return []
    return _parse_fasta_sequences(content)


def create_single_prediction_csv(prediction_data, problem_type, aa_seq):
    """Create CSV file for single prediction results"""
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv', prefix='single_prediction_') as temp_file:
            if problem_type == "residue_single_label_classification":
                # For residue classification, create detailed CSV with position-level predictions
                aa_seq_list = prediction_data.get('aa_seq', list(aa_seq))
                predicted_classes = prediction_data.get('predicted_classes', [])
                probabilities = prediction_data.get('probabilities', [])
                
                # Write header
                temp_file.write("Position,Amino_Acid,Predicted_Class")
                if probabilities and len(probabilities) > 0 and len(probabilities[0]) > 0:
                    for i in range(len(probabilities[0])):
                        temp_file.write(f",Class_{i}_Probability")
                temp_file.write("\n")
                
                # Write data rows
                for pos, (aa, pred_class, probs) in enumerate(zip(aa_seq_list, predicted_classes, probabilities)):
                    temp_file.write(f"{pos + 1},{aa},{pred_class}")
                    if probs:
                        for prob in probs:
                            temp_file.write(f",{prob:.6f}")
                    temp_file.write("\n")
                    
            elif problem_type == "residue_regression":
                # For residue regression, create CSV with position-level predictions
                aa_seq_list = prediction_data.get('aa_seq', list(aa_seq))
                predictions = prediction_data.get('predictions', [])
                
                # Write header
                temp_file.write("Position,Amino_Acid,Predicted_Value\n")
                
                # Write data rows
                for pos, (aa, pred_value) in enumerate(zip(aa_seq_list, predictions)):
                    temp_file.write(f"{pos + 1},{aa},{pred_value:.6f}\n")
                    
            elif problem_type == "single_label_classification":
                # For single-label classification (API may return predicted_class or predicted_classes)
                predicted_class = prediction_data.get('predicted_class')
                if predicted_class is None and 'predicted_classes' in prediction_data:
                    pcs = prediction_data['predicted_classes']
                    predicted_class = pcs[0] if isinstance(pcs, list) and pcs else 0
                if predicted_class is None:
                    predicted_class = 0
                probs_raw = prediction_data.get('probabilities', [])
                probabilities = probs_raw[0] if (isinstance(probs_raw, list) and probs_raw and isinstance(probs_raw[0], list)) else probs_raw
                if not isinstance(probabilities, list):
                    probabilities = [probabilities]
                
                # Write header
                temp_file.write("Predicted_Class")
                for i in range(len(probabilities)):
                    temp_file.write(f",Class_{i}_Probability")
                temp_file.write(",Amino_Acid_Sequence\n")
                
                # Write data row
                temp_file.write(f"{predicted_class}")
                for prob in probabilities:
                    temp_file.write(f",{prob:.6f}")
                temp_file.write(f",{aa_seq}\n")
                
            elif problem_type == "multi_label_classification":
                # For multi-label classification
                predictions = prediction_data.get('predictions', [])
                probabilities = prediction_data.get('probabilities', [])
                
                # Write header
                temp_file.write("Predicted_Labels")
                for i in range(len(probabilities)):
                    temp_file.write(f",Label_{i}_Probability")
                temp_file.write(",Amino_Acid_Sequence\n")
                
                # Write data row
                temp_file.write(f"{predictions}")
                for prob in probabilities:
                    temp_file.write(f",{prob:.6f}")
                temp_file.write(f",{aa_seq}\n")
                
            elif problem_type == "regression":
                # For regression
                prediction = prediction_data.get('prediction', 0)
                
                # Write header
                temp_file.write("Predicted_Value,Amino_Acid_Sequence\n")
                
                # Write data row
                temp_file.write(f"{prediction:.6f},{aa_seq}\n")
        
        return temp_file.name
    except Exception as e:
        print(f"Error creating CSV file: {e}")
        return None

def create_predict_tab(constant):
    plm_models = constant["plm_models"]
    is_predicting = False
    current_process = None
    output_queue = queue.Queue()
    stop_thread = False
    process_aborted = False  # Flag indicating if the process was manually terminated
    
    def track_usage(module):
        try:
            import requests
            requests.post("http://localhost:8000/api/stats/track", 
                         json={"module": module, "timestamp": datetime.now().isoformat()})
        except Exception as e:
            print(f"Failed to track usage: {e}")

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
        
        return generate_prediction_status_html(stage, status)

    def predict_sequence(plm_model, model_path, aa_seq, eval_method, eval_structure_seq, pooling_method, problem_type, num_labels):
        """Predict for a single protein sequence"""
        nonlocal is_predicting, current_process, stop_thread, process_aborted
        
        # Check if we're already predicting
        if is_predicting:
            return gr.HTML(f"""
            {get_css_style_tag('prediction_ui.css')}
            {load_html_template('prediction_warning.html', warning_message="A prediction is already running. Please wait or abort it.")}
            """), gr.update(visible=False)
        
        track_usage("mutation_prediction")
        
        # If the process was aborted but not reset properly, ensure we're in a clean state
        if process_aborted:
            process_aborted = False
            
        # Set the prediction flag
        is_predicting = True
        stop_thread = False  # Ensure this is reset
        
        # Create a status info object, similar to batch prediction
        status_info = {
            "status": "running",
            "current_step": "Starting prediction"
        }
        
        # Show initial status
        yield generate_status_html(status_info), gr.update(visible=False)
        
        try:
            # Validate inputs
            if not model_path:
                is_predicting = False
                return gr.HTML(f"""
                {get_css_style_tag('prediction_ui.css')}
                {load_html_template('prediction_error.html', error_message="Please provide a model path")}
                """), gr.update(visible=False)
                
            if not os.path.exists(os.path.dirname(model_path)):
                is_predicting = False
                return gr.HTML(f"""
                {get_css_style_tag('prediction_ui.css')}
                {load_html_template('prediction_error.html', error_message="Invalid model path - directory does not exist")}
                """), gr.update(visible=False)
                
            if not aa_seq:
                is_predicting = False
                return gr.HTML(f"""
                {get_css_style_tag('prediction_ui.css')}
                {load_html_template('prediction_error.html', error_message="Amino acid sequence is required")}
                """), gr.update(visible=False)
            
            # Update status
            status_info["current_step"] = "Preparing model and parameters"
            yield generate_status_html(status_info), gr.update(visible=False)
            
            # Prepare command
            args_dict = {
                "model_path": model_path,
                "plm_model": plm_models[plm_model],
                "aa_seq": aa_seq,
                "pooling_method": pooling_method,
                "problem_type": problem_type,
                "num_labels": num_labels,
                "eval_method": eval_method,
            }
            
            if eval_method == "ses-adapter":
                # Handle structure sequence selection from multi-select dropdown
                args_dict["structure_seq"] = ",".join(eval_structure_seq) if eval_structure_seq else None
                
                # Set flags based on selected structure sequences
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
            
            # Update status
            status_info["current_step"] = "Starting prediction process"
            yield generate_status_html(status_info), gr.update(visible=False)
            
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
                {get_css_style_tag('prediction_ui.css')}
                {load_html_template('prediction_error.html', error_message=f"Error starting prediction process: {str(e)}")}
                """), gr.update(visible=False)
            
            output_thread = threading.Thread(target=process_output, args=(current_process, output_queue))
            output_thread.daemon = True
            output_thread.start()
            
            # Collect output
            result_output = ""
            prediction_data = None
            json_str = ""
            in_json_block = False
            json_lines = []
            
            # Update status
            status_info["current_step"] = "Processing sequence"
            yield generate_status_html(status_info), gr.update(visible=False)
            
            while current_process.poll() is None:
                # Check if the process was aborted
                if process_aborted or stop_thread:
                    break
                
                try:
                    while not output_queue.empty():
                        line = output_queue.get_nowait()
                        result_output += line + "\n"
                        
                        # Update status with more meaningful messages
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
                        
                        # Update status display
                        yield generate_status_html(status_info), gr.update(visible=False)
                        
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
                    {get_css_style_tag('prediction_ui.css')}
                    {load_html_template('prediction_warning.html', warning_message=f"Warning reading output: {str(e)}")}
                    """), gr.update(visible=False)
            
            # Check if the process was aborted
            if process_aborted:
                # Show aborted message
                yield gr.HTML(f"""
                {get_css_style_tag('prediction_ui.css')}
                {load_html_template('prediction_warning.html', warning_message="Prediction was aborted by user")}
                """), gr.update(visible=False)
                is_predicting = False
                return
            
            # Process has completed
            if current_process and current_process.returncode == 0:
                # Update status
                status_info["status"] = "completed"
                status_info["current_step"] = "Prediction completed successfully"
                yield generate_status_html(status_info), gr.update(visible=False)
                
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
                    # Generate prediction results HTML using template
                    html_result = f"""
                    {get_css_style_tag('prediction_ui.css')}
                    {generate_prediction_results_html(problem_type, prediction_data)}
                    """
                    
                    # Create CSV file for download
                    csv_file = create_single_prediction_csv(prediction_data, problem_type, aa_seq)
                    
                    yield gr.HTML(html_result), gr.update(value=csv_file, visible=True)
                else:
                    # If no prediction data found, display raw output
                    yield gr.HTML(f"""
                    {get_css_style_tag('prediction_ui.css')}
                    {load_html_template('prediction_completed_no_results.html', result_output=result_output)}
                    """), gr.update(visible=False)
            else:
                # Update status
                status_info["status"] = "failed"
                status_info["current_step"] = "Prediction failed"
                yield generate_status_html(status_info), gr.update(visible=False)
                
                stderr_output = ""
                if current_process and hasattr(current_process, 'stderr') and current_process.stderr:
                    stderr_output = current_process.stderr.read()
                combined_error = f"{stderr_output}\n{result_output}"
                yield gr.HTML(f"""
                {get_css_style_tag('prediction_ui.css')}
                {load_html_template('prediction_failed.html', 
                                  error_code=current_process.returncode if current_process else 'Unknown',
                                  error_output=combined_error)}
                """), gr.update(visible=False)
        except Exception as e:
            # Update status
            status_info["status"] = "failed"
            status_info["current_step"] = "Error occurred"
            yield generate_status_html(status_info), gr.update(visible=False)
            
            yield gr.HTML(f"""
            {get_css_style_tag('prediction_ui.css')}
            {load_html_template('prediction_error_with_traceback.html', 
                              error_message=str(e),
                              traceback=traceback.format_exc())}
            """), gr.update(visible=False)
        finally:
            # Reset state
            is_predicting = False
            
            # Properly clean up the process
            if current_process and current_process.poll() is None:
                try:
                    # Use process group ID to kill all related processes if possible
                    if hasattr(os, "killpg") and hasattr(os, "getpgid"):
                        os.killpg(os.getpgid(current_process.pid), signal.SIGTERM)
                    else:
                        # On Windows or if killpg is not available
                        current_process.terminate()
                        
                    # Wait briefly for termination
                    try:
                        current_process.wait(timeout=1)
                    except subprocess.TimeoutExpired:
                        # Force kill if necessary
                        if hasattr(os, "killpg") and hasattr(os, "getpgid"):
                            os.killpg(os.getpgid(current_process.pid), signal.SIGKILL)
                        else:
                            current_process.kill()
                except Exception as e:
                    # Ignore errors during process cleanup
                    print(f"Error cleaning up process: {e}")
                
            # Reset process reference
            current_process = None
            stop_thread = False

    def predict_batch(plm_model, model_path, eval_method, input_file, eval_structure_seq, pooling_method, problem_type, num_labels, batch_size, pdb_dir_val=None):
        """Batch predict multiple protein sequences"""
        nonlocal is_predicting, current_process, stop_thread, process_aborted
        
        # Check if we're already predicting (this check is performed first)
        if is_predicting:
            return gr.HTML(f"""
            {get_css_style_tag('prediction_ui.css')}
            {load_html_template('prediction_warning.html', warning_message="A prediction is already running. Please wait or abort it.")}
            """)

        track_usage("mutation_prediction")
        
        # If the process was aborted but not reset properly, ensure we're in a clean state
        if process_aborted:
            process_aborted = False
        
        # Reset all state completely
        is_predicting = True
        stop_thread = False
        
        # Clear the output queue
        while not output_queue.empty():
            try:
                output_queue.get_nowait()
            except queue.Empty:
                break
        
        # Initialize progress tracking (match Evaluation tab style)
        start_time = time.time()
        progress_info = {
            "total": 0,
            "completed": 0,
            "current_step": "Initializing",
            "stage": "Preparing",
            "progress": 0,
            "current": 0,
            "total_samples": 0,
            "elapsed_time": "00:00:00",
            "status": "running",
            "lines": []
        }
        
        # Generate initial progress bar (same style as Evaluation)
        initial_progress_html = _generate_prediction_progress_bar(progress_info)
        
        # Always ensure the download button is hidden when starting a new prediction
        yield gr.HTML(initial_progress_html), gr.update(visible=False)
        
        try:
            # Check abort state before continuing
            if process_aborted:
                is_predicting = False
                return gr.HTML(f"""
                {get_css_style_tag('prediction_ui.css')}
                {load_html_template('status_success.html', message="Process was aborted.")}
                """), gr.update(visible=False)
            
            # Validate inputs
            if not model_path:
                is_predicting = False
                yield gr.HTML(f"""
                {get_css_style_tag('prediction_ui.css')}
                {load_html_template('prediction_error.html', error_message="Error: Model path is required")}
                """), gr.update(visible=False)
                return
                
            if not os.path.exists(os.path.dirname(model_path)):
                is_predicting = False
                yield gr.HTML(f"""
                {get_css_style_tag('prediction_ui.css')}
                {load_html_template('prediction_error.html', error_message="Error: Invalid model path - directory does not exist")}
                """), gr.update(visible=False)
                return
            
            if not input_file:
                is_predicting = False
                yield gr.HTML(f"""
                {get_css_style_tag('prediction_ui.css')}
                {load_html_template('prediction_error.html', error_message="Error: Input file is required")}
                """), gr.update(visible=False)
                return
            
            # Update progress
            progress_info["current_step"] = "Preparing input file"
            yield _generate_prediction_progress_bar(progress_info), gr.update(visible=False)
            
            # Use get_save_path (env TEMP_OUTPUTS_DIR) for output, create unique run subdir
            base_dir = get_save_path("Prediction", "Results")
            run_dir = tempfile.mkdtemp(dir=str(base_dir))
            input_path = os.path.join(run_dir, "input.csv")
            output_dir = run_dir
            output_file = "predictions.csv"
            output_path = os.path.join(output_dir, output_file)
            
            # Save uploaded file
            try:
                with open(input_path, "wb") as f:
                    # Fix file upload error, correctly handle files uploaded through gradio
                    if hasattr(input_file, "name"):
                        # If it's a NamedString object, read the file content
                        with open(input_file.name, "rb") as uploaded:
                            f.write(uploaded.read())
                    else:
                        # If it's a bytes object, write directly
                        f.write(input_file)
                
                # Verify file was saved correctly
                if not os.path.exists(input_path):
                    is_predicting = False
                    yield gr.HTML(f"""
                    {get_css_style_tag('prediction_ui.css')}
                    {load_html_template('prediction_error.html', error_message="Error: Failed to save input file")}
                    """), gr.update(visible=False)
                    progress_info["status"] = "failed"
                    progress_info["current_step"] = "Failed to save input file"
                    return
                
                # Count sequences in input file
                try:
                    df = pd.read_csv(input_path)
                    n_seqs = len(df)
                    progress_info["total"] = n_seqs
                    progress_info["total_samples"] = n_seqs
                    progress_info["current_step"] = f"Found {n_seqs} sequences to process"
                    progress_info["stage"] = "Preparing"
                    yield _generate_prediction_progress_bar(progress_info), gr.update(visible=False)
                except Exception as e:
                    is_predicting = False
                    yield gr.HTML(f"""
                    {get_css_style_tag('prediction_ui.css')}
                    {load_html_template('prediction_error_with_details.html', 
                                      error_message="Error reading CSV file:",
                                      error_details=load_html_template('error_pre.html', content=str(e)))}
                    """), gr.update(visible=False)
                    progress_info["status"] = "failed"
                    progress_info["current_step"] = "Error reading CSV file"
                    return
                
            except Exception as e:
                is_predicting = False
                yield gr.HTML(f"""
                {get_css_style_tag('prediction_ui.css')}
                                    {load_html_template('prediction_error_with_details.html', 
                                      error_message="Error saving input file:",
                                      error_details=load_html_template('error_pre.html', content=str(e)))}
                """), gr.update(visible=False)
                progress_info["status"] = "failed"
                progress_info["current_step"] = "Failed to save input file"
                return
            
            # Update progress
            progress_info["current_step"] = "Preparing model and parameters"
            yield _generate_prediction_progress_bar(progress_info), gr.update(visible=False)
            
            # Prepare command
            args_dict = {
                "model_path": model_path,
                "plm_model": plm_models[plm_model],
                "input_file": input_path,
                "output_dir": output_dir,  # Update to output directory
                "output_file": output_file,  # Output filename
                "pooling_method": pooling_method,
                "problem_type": problem_type,
                "num_labels": num_labels,
                "eval_method": eval_method,
                "batch_size": batch_size,
            }
            
            if eval_method == "ses-adapter":
                args_dict["structure_seq"] = ",".join(eval_structure_seq) if eval_structure_seq else None
                if eval_structure_seq:
                    if "foldseek_seq" in eval_structure_seq:
                        args_dict["use_foldseek"] = True
                    if "ss8_seq" in eval_structure_seq:
                        args_dict["use_ss8"] = True
            else:
                args_dict["structure_seq"] = None
            if pdb_dir_val and str(pdb_dir_val).strip():
                args_dict["pdb_dir"] = str(pdb_dir_val).strip()
            
            # Build command line
            final_cmd = [sys.executable, "src/predict.py"]
            for k, v in args_dict.items():
                if v is True:
                    final_cmd.append(f"--{k}")
                elif v is not False and v is not None:
                    final_cmd.append(f"--{k}")
                    final_cmd.append(str(v))
            
            # Update progress
            progress_info["current_step"] = "Starting batch prediction process"
            yield _generate_prediction_progress_bar(progress_info), gr.update(visible=False)
            
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
                yield gr.HTML(f"""
                {get_css_style_tag('prediction_ui.css')}
                {load_html_template('prediction_error.html', error_message=f"Error starting prediction process: {str(e)}")}
                """), gr.update(visible=False)
                return
            
            output_thread = threading.Thread(target=process_output, args=(current_process, output_queue))
            output_thread.daemon = True
            output_thread.start()
            
            # Start monitoring loop
            last_update_time = time.time()
            result_output = ""
            
            # Modified processing loop with abort check
            while True:
                # Check if process was aborted or completed
                if process_aborted or current_process is None or current_process.poll() is not None:
                    break
                
                # Check for new output
                try:
                    # Get new lines
                    new_lines = []
                    for _ in range(10):  # Process up to 10 lines at once
                        try:
                            line = output_queue.get_nowait()
                            new_lines.append(line)
                            result_output += line + "\n"
                            progress_info["lines"].append(line)
                            
                            # Update progress based on output
                            if "Predicting:" in line or "it/s" in line:
                                try:
                                    match = re.search(r'(\d+)/(\d+)', line)
                                    if match:
                                        cur, tot = map(int, match.groups())
                                        progress_info["completed"] = cur
                                        progress_info["current"] = cur
                                        progress_info["total"] = tot
                                        progress_info["total_samples"] = tot
                                        progress_info["current_step"] = f"Processing sequence {cur}/{tot}"
                                        progress_info["stage"] = "Predicting"
                                except Exception:
                                    pass
                            elif "Loading Model and Tokenizer" in line:
                                progress_info["current_step"] = "Loading model and tokenizer"
                            elif "Processing sequences" in line:
                                progress_info["current_step"] = "Processing sequences"
                            elif "Saving results" in line:
                                progress_info["current_step"] = "Saving results"
                        except queue.Empty:
                            break
                    
                    # Check if the process has been aborted before updating UI
                    if process_aborted:
                        break
                    elapsed = time.time() - start_time
                    hours, remainder = divmod(int(elapsed), 3600)
                    minutes, seconds = divmod(remainder, 60)
                    progress_info["elapsed_time"] = f"{hours:02}:{minutes:02}:{seconds:02}"
                    current_time = time.time()
                    if new_lines or (current_time - last_update_time >= 0.5):
                        yield _generate_prediction_progress_bar(progress_info), gr.update(visible=False)
                        last_update_time = current_time
                    
                    # Small sleep to avoid busy waiting
                    if not new_lines:
                        time.sleep(0.1)
                    
                except Exception as e:
                    # Check if the process has been aborted before showing error
                    if process_aborted:
                        break
                        
                    yield gr.HTML(f"""
                    {get_css_style_tag('prediction_ui.css')}
                    {load_html_template('prediction_warning.html', warning_message=f"Warning reading output: {str(e)}")}
                    """), gr.update(visible=False)
            
            # Check if aborted instead of completed
            if process_aborted:
                is_predicting = False
                yield gr.HTML(f"""
                {get_css_style_tag('prediction_ui.css')}
                {load_html_template('prediction_warning.html', warning_message="Prediction was manually terminated. All prediction state has been reset.")}
                """), gr.update(visible=False)
                return
            
            # Process has completed
            if os.path.exists(output_path):
                if current_process and current_process.returncode == 0:
                    progress_info["status"] = "completed"
                    # Generate final success HTML
                    success_html = f"""
                    {get_css_style_tag('prediction_ui.css')}
                    {load_html_template('batch_prediction_success.html', 
                                      output_path=output_path,
                                      total_sequences=progress_info.get('total', 0))}
                    """
                    
                    # Read prediction results
                    try:
                        df = pd.read_csv(output_path)
                        
                        # Generate batch prediction results HTML using template
                        final_html = success_html + generate_batch_prediction_results_html(df, problem_type)
                        
                        # Return results preview and download link
                        yield gr.HTML(final_html), gr.update(value=output_path, visible=True)
                    except Exception as e:
                        # If reading results file fails, show error but still provide download link
                        error_html = f"""
                        {success_html}
                        {get_css_style_tag('prediction_ui.css')}
                        {load_html_template('prediction_warning_with_details.html', 
                                          warning_message=f"Unable to load preview results: {str(e)}",
                                          warning_details="You can still download the complete prediction results file.")}
                        """
                        yield gr.HTML(error_html), gr.update(value=output_path, visible=True)
                else:
                    # Process failed
                    return_code = current_process.returncode if current_process else 'Unknown'
                    error_p = load_html_template('error_p.html', content=f'Process return code: {return_code}')
                    error_pre = load_html_template('error_pre.html', content=result_output)
                    error_details = f"{error_p}{error_pre}"
                    error_html = f"""
                    {get_css_style_tag('prediction_ui.css')}
                    {load_html_template('prediction_error_with_details.html', 
                                      error_message="Prediction failed to complete",
                                      error_details=error_details)}
                    """
                    yield gr.HTML(error_html), gr.update(visible=False)
            else:
                progress_info["status"] = "failed"
                error_html = f"""
                {get_css_style_tag('prediction_ui.css')}
                {load_html_template('prediction_error_with_details.html', 
                                  error_message=f"Prediction completed, but output file not found at {output_path}",
                                  error_details=load_html_template('error_pre.html', content=result_output))}
                """
                yield gr.HTML(error_html), gr.update(visible=False)
        except Exception as e:
            # Capture the full error with traceback
            error_traceback = traceback.format_exc()
            
            # Display error with traceback in UI
            error_html = f"""
            {get_css_style_tag('prediction_ui.css')}
            {load_html_template('prediction_error_with_details.html', 
                              error_message=f"Error during batch prediction: {str(e)}",
                              error_details=load_html_template('error_pre_with_bg.html', content=error_traceback))}
            """
            yield gr.HTML(error_html), gr.update(visible=False)
        finally:
            # Always reset prediction state
            is_predicting = False
            if current_process:
                current_process = None
            process_aborted = False  # Reset abort flag

    def _generate_prediction_progress_bar(progress_info):
        """Generate HTML progress bar matching Evaluation tab style (evaluation_progress.html)."""
        stage = progress_info.get("stage", progress_info.get("current_step", "Preparing"))
        current = progress_info.get("current", progress_info.get("completed", 0))
        total = max(progress_info.get("total", progress_info.get("total_samples", 1)), 1)
        progress = (current / total) * 100 if total > 0 else 0
        progress = max(0, min(100, progress))
        total_samples = progress_info.get("total_samples", total)
        total_samples_detail = f'<div class="progress-detail-item progress-detail-total"><span style="font-weight: 500;">Total samples:</span> {total_samples}</div>' if total_samples > 0 else ''
        progress_detail = f'<div class="progress-detail-item progress-detail-current"><span style="font-weight: 500;">Progress:</span> {current}/{total}</div>' if current > 0 and total > 0 else ''
        time_detail = f'<div class="progress-detail-item progress-detail-time"><span style="font-weight: 500;">Time:</span> {progress_info.get("elapsed_time", "")}</div>' if progress_info.get("elapsed_time") else ''
        return f"""
        {get_css_style_tag('eval_predict_ui.css')}
        {load_html_template('evaluation_progress.html',
                          stage=stage,
                          progress=progress,
                          total_samples_detail=total_samples_detail,
                          progress_detail=progress_detail,
                          time_detail=time_detail)}
        """

    def generate_table_rows(df, max_rows=100):
        """Generate HTML table rows with special handling for sequence data, maintaining consistent style with eval_tab"""
        return generate_table_rows(df, max_rows)

    def handle_predict_tab_abort():
        """Handle abortion of the prediction process for both single and batch prediction"""
        nonlocal is_predicting, current_process, stop_thread, process_aborted
        
        if not is_predicting or current_process is None:
            return f"""
            {get_css_style_tag('prediction_ui.css')}
            {load_html_template('status_empty.html', message="No prediction process is currently running.")}
            """
        
        try:
            # Set the abort flag before terminating the process
            process_aborted = True
            stop_thread = True
            
            # Kill the process group
            if hasattr(os, "killpg"):
                os.killpg(os.getpgid(current_process.pid), signal.SIGTERM)
            else:
                current_process.terminate()
            
            # Wait for process to terminate (with timeout)
            try:
                current_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                if hasattr(os, "killpg"):
                    os.killpg(os.getpgid(current_process.pid), signal.SIGKILL)
                else:
                    current_process.kill()
            
            # Reset state
            is_predicting = False
            current_process = None
            
            # Clear output queue
            while not output_queue.empty():
                try:
                    output_queue.get_nowait()
                except queue.Empty:
                    break
            
            return f"""
            {get_css_style_tag('prediction_ui.css')}
            {load_html_template('status_success.html', message="Prediction successfully terminated! All prediction state has been reset.")}
            """
                
        except Exception as e:
            # Reset states even on error
            is_predicting = False
            current_process = None
            process_aborted = False
            
            # Clear queue
            while not output_queue.empty():
                try:
                    output_queue.get_nowait()
                except queue.Empty:
                    break
                    
            return f"""
            {get_css_style_tag('prediction_ui.css')}
            {load_html_template('status_failed_terminate.html', error_message=f"Failed to terminate prediction: {str(e)}")}
            """

    # Create handler functions for each tab
    def handle_abort_single():
        """Handle abort for single sequence prediction tab"""
        # Flag the process for abortion first
        nonlocal stop_thread, process_aborted, is_predicting, current_process
        
        # Only proceed if there's an active prediction
        if not is_predicting or current_process is None:
            return gr.HTML(f"""
            {get_css_style_tag('prediction_ui.css')}
            {load_html_template('status_empty.html', message="No prediction process is currently running.")}
            """), gr.update(visible=False)
            
        # Set the abort flags
        process_aborted = True
        stop_thread = True
        
        # Terminate the process
        try:
            if hasattr(os, "killpg"):
                os.killpg(os.getpgid(current_process.pid), signal.SIGTERM)
            else:
                current_process.terminate()
                
            # Wait briefly for termination
            try:
                current_process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                # Force kill if necessary
                if hasattr(os, "killpg"):
                    os.killpg(os.getpgid(current_process.pid), signal.SIGKILL)
                else:
                    current_process.kill()
        except Exception as e:
            pass  # Catch any termination errors
            
        # Reset state
        is_predicting = False
        current_process = None
        
        # Return the success message and hide download button
        return gr.HTML(f"""
        {get_css_style_tag('prediction_ui.css')}
        {load_html_template('status_success.html', message="Prediction successfully terminated! All prediction state has been reset.")}
        """), gr.update(visible=False)
        
    def handle_abort_batch():
        """Handle abort for batch prediction tab"""
        # Flag the process for abortion first
        nonlocal stop_thread, process_aborted, is_predicting, current_process
        
        # Only proceed if there's an active prediction
        if not is_predicting or current_process is None:
            return gr.HTML(f"""
            {get_css_style_tag('prediction_ui.css')}
            {load_html_template('status_empty.html', message="No prediction process is currently running.")}
            """), gr.update(visible=False)
            
        # Set the abort flags
        process_aborted = True
        stop_thread = True
        
        # Terminate the process
        try:
            if hasattr(os, "killpg"):
                os.killpg(os.getpgid(current_process.pid), signal.SIGTERM)
            else:
                current_process.terminate()
                
            # Wait briefly for termination
            try:
                current_process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                # Force kill if necessary
                if hasattr(os, "killpg"):
                    os.killpg(os.getpgid(current_process.pid), signal.SIGKILL)
                else:
                    current_process.kill()
        except Exception as e:
            pass  # Catch any termination errors
            
        # Reset state
        is_predicting = False
        current_process = None
        
        # Clear output queue
        while not output_queue.empty():
            try:
                output_queue.get_nowait()
            except queue.Empty:
                break
                
        # Return the success message and hide the download button
        return gr.HTML(f"""
        {get_css_style_tag('prediction_ui.css')}
        {load_html_template('status_success.html', message="Prediction successfully terminated! All prediction state has been reset.")}
        """), gr.update(visible=False)

    def handle_predict_tab_command_preview(input_method_val, aa_seq_val, seq_file, plm_model, model_path, eval_method, foldseek_seq, ss8_seq, eval_structure_seq, pooling_method, problem_type, num_labels, predict_pdb_dir_val=None):
        """Handle the preview command button click event. Resolve sequences from paste or upload."""
        if input_method_val == "Upload file" and seq_file is not None:
            records = _parse_fasta_from_file(seq_file)
        else:
            records = _parse_fasta_sequences((aa_seq_val or "").strip())
        if records:
            aa_seq_val = records[0][1][:80] + ("..." if len(records[0][1]) > 80 else "")
            if len(records) > 1:
                aa_seq_val = f"[{len(records)} sequences, first: {aa_seq_val}]"
        else:
            aa_seq_val = "(paste or upload FASTA)"
        args_dict = {
            "model_path": model_path,
            "plm_model": plm_models[plm_model],
            "aa_seq": aa_seq_val,
            "foldseek_seq": foldseek_seq if foldseek_seq else "",
            "ss8_seq": ss8_seq if ss8_seq else "",
            "pooling_method": pooling_method,
            "problem_type": problem_type,
            "num_labels": num_labels,
            "eval_method": eval_method,
        }
        
        if eval_method == "ses-adapter":
            args_dict["structure_seq"] = ",".join(eval_structure_seq) if eval_structure_seq else None
            if eval_structure_seq:
                if "foldseek_seq" in eval_structure_seq:
                    args_dict["use_foldseek"] = True
                if "ss8_seq" in eval_structure_seq:
                    args_dict["use_ss8"] = True
        if predict_pdb_dir_val and str(predict_pdb_dir_val).strip():
            args_dict["pdb_dir"] = str(predict_pdb_dir_val).strip()
        
        # generate preview command
        preview_text = preview_predict_command(args_dict, is_batch=False)
        return gr.update(value=preview_text, visible=True)
        
    def handle_batch_preview(plm_model, model_path, eval_method, input_file, eval_structure_seq, pooling_method, problem_type, num_labels, batch_size):
        """handle batch prediction command preview"""
        if not input_file:
            return gr.update(value="Please upload a file first", visible=True)
        
        # create temporary directory as output directory
        temp_dir = "temp_predictions"
        output_file = "predictions.csv"
        
        args_dict = {
            "model_path": model_path,
            "plm_model": plm_models[plm_model],
            "input_file": input_file.name if hasattr(input_file, "name") else "input.csv",
            "output_dir": temp_dir,  # add output directory parameter
            "output_file": output_file,  # output file name
            "pooling_method": pooling_method,
            "problem_type": problem_type,
            "num_labels": num_labels,
            "eval_method": eval_method,
            "batch_size": batch_size,
        }
        
        if eval_method == "ses-adapter":
            args_dict["structure_seq"] = ",".join(eval_structure_seq) if eval_structure_seq else None
            if eval_structure_seq:
                if "foldseek_seq" in eval_structure_seq:
                    args_dict["use_foldseek"] = True
                if "ss8_seq" in eval_structure_seq:
                    args_dict["use_ss8"] = True
        
        # generate preview command
        preview_text = preview_predict_command(args_dict, is_batch=True)
        return gr.update(value=preview_text, visible=True)

    # Layout: left = Model Configuration, Data Input, Hyperparameter Settings, buttons; right = results (like quick tools)
    _folder_choices = [(base, base) for base in ("ckpt",) if os.path.isdir(base)] + _scan_folders_under("ckpt")
    with gr.Row(equal_height=False):
        with gr.Column(scale=3):
            gr.Markdown("💡 *Inference on unlabeled/unknown data.*")
            gr.Markdown("## Model and Dataset Configuration")
            with gr.Group():
                with gr.Row():
                    model_folder = gr.Dropdown(
                        label="Model Folder Path",
                        choices=_folder_choices,
                        value=None,
                        allow_custom_value=False
                    )
                    model_dropdown = gr.Dropdown(
                        choices=[],
                        label="Model Name",
                        value=None,
                        allow_custom_value=False
                    )
            with gr.Row(visible=False) as predict_pdb_dir_row:
                predict_pdb_dir = gr.Textbox(
                    label="PDB Directory",
                    placeholder="Path to PDB files (required for ProSST/SaProt/ProtSSN; optional for ses-adapter)",
                    value="",
                    scale=3
                )
            with gr.Row(visible=False):
                plm_model = gr.Dropdown(choices=list(plm_models.keys()), value=list(plm_models.keys())[0] if plm_models else None)
            with gr.Row(visible=False) as structure_seq_row:
                structure_seq = gr.Dropdown(
                    choices=["foldseek_seq", "ss8_seq"],
                    label="Structure Sequences",
                    multiselect=True,
                    value=["foldseek_seq", "ss8_seq"]
                )
            with gr.Row(visible=False):
                problem_type = gr.Dropdown(
                    choices=["single_label_classification", "multi_label_classification", "regression", "residue_single_label_classification", "residue_regression"],
                    value="single_label_classification"
                )
                num_labels = gr.Number(value=2, precision=0, minimum=1)
            with gr.Row(visible=False):
                eval_method = gr.Dropdown(choices=["full", "freeze", "ses-adapter", "plm-lora", "plm-qlora", "plm-adalora", "plm-dora", "plm-ia3"], value="freeze")
                pooling_method = gr.Dropdown(choices=["mean", "attention1d", "light_attention"], value="mean")
            with gr.Row(visible=False):
                otg_message = gr.HTML(f"{get_css_style_tag('prediction_ui.css')}", visible=False)

            def _on_model_folder_change(folder_path):
                if not folder_path:
                    return [gr.update(choices=[], value=None)] + _on_model_selected(None)
                choices = _scan_models_in_folder(folder_path)
                if not choices:
                    return [gr.update(choices=[], value=None)] + _on_model_selected(None)
                first_path = choices[0][1]
                cfg_upds = _on_model_selected(first_path)
                return [gr.update(choices=choices, value=first_path)] + cfg_upds

            def _on_model_selected(model_path_val):
                cfg = _load_model_config(model_path_val) if model_path_val else {}
                if not cfg:
                    return [gr.update() for _ in range(6)] + [
                        gr.update(visible=False), gr.update(visible=True),
                        gr.update(visible=False), gr.update(visible=False)
                    ]
                plm_path = cfg.get("plm_model", "")
                plm_key = None
                for k, v in plm_models.items():
                    if v == plm_path:
                        plm_key = k
                        break
                if not plm_key and plm_models:
                    plm_key = list(plm_models.keys())[0]
                eval_m = cfg.get("training_method", "freeze")
                if eval_m not in ["full", "freeze", "ses-adapter", "plm-lora", "plm-qlora", "plm-adalora", "plm-dora", "plm-ia3"]:
                    eval_m = "freeze"
                struct_seq = cfg.get("structure_seq", [])
                if not isinstance(struct_seq, list):
                    struct_seq = [struct_seq] if struct_seq else []
                struct_seq = [s for s in struct_seq if s in ("foldseek_seq", "ss8_seq")] or ["foldseek_seq", "ss8_seq"]
                is_structure = bool(plm_key and (str(plm_key).startswith("ProSST") or str(plm_key).startswith("SaProt") or str(plm_key).startswith("ProtSSN")))
                is_ses = eval_m == "ses-adapter"
                show_pdb = is_structure or is_ses
                show_input = not is_structure
                vis = (gr.update(visible=show_pdb), gr.update(visible=show_input),
                       gr.update(visible=is_ses), gr.update(visible=is_ses))
                return [
                    gr.update(value=plm_key),
                    gr.update(value=eval_m),
                    gr.update(value=cfg.get("pooling_method", "mean")),
                    gr.update(value=cfg.get("problem_type", "single_label_classification")),
                    gr.update(value=int(cfg.get("num_labels", 2))),
                    gr.update(value=struct_seq),
                ] + list(vis)

            gr.Markdown("## Input Data")
            with gr.Column(visible=True) as input_data_column:
                input_method = gr.Radio(
                    choices=["Paste sequence", "Upload file", "Specify FASTA path"],
                    label="Input Method",
                    value="Paste sequence"
                )
                with gr.Column(visible=True) as paste_seq_column:
                    aa_seq = gr.Textbox(
                        label="Input Amino Acid Sequence (FASTA format)",
                        placeholder="Example: >seq1\nMVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSKGPSSGFSGKGA",
                        lines=3
                    )
                with gr.Column(visible=False) as upload_seq_column:
                    seq_file_upload = gr.File(
                        label="Upload FASTA or sequence file",
                        file_types=[".fasta", ".fa", ".txt"],
                        file_count="single"
                    )
                with gr.Column(visible=False) as path_seq_column:
                    seq_path_input = gr.Textbox(
                        label="FASTA file path",
                        placeholder="Absolute or relative path to FASTA file",
                        value=""
                    )
                with gr.Row(visible=False) as structure_input_row:
                    foldseek_seq = gr.Textbox(
                        label="Foldseek Sequence (FASTA format)",
                        placeholder=">id\nFoldseek sequence per residue...",
                        lines=2
                    )
                    ss8_seq = gr.Textbox(
                        label="SS8 Sequence (FASTA format)",
                        placeholder=">id\nHECGIBST secondary structure codes...",
                        lines=2
                    )

            model_folder.change(
                fn=_on_model_folder_change,
                inputs=[model_folder],
                outputs=[model_dropdown, plm_model, eval_method, pooling_method, problem_type, num_labels, structure_seq,
                         predict_pdb_dir_row, input_data_column, structure_seq_row, structure_input_row]
            )
            model_dropdown.change(
                fn=_on_model_selected,
                inputs=[model_dropdown],
                outputs=[plm_model, eval_method, pooling_method, problem_type, num_labels, structure_seq,
                         predict_pdb_dir_row, input_data_column, structure_seq_row, structure_input_row]
            )

            with gr.Accordion("Hyperparameter Settings", open=False):
                gr.Markdown("### Batch Processing Configuration")
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
                            value=1,
                            step=1,
                            label="Batch Size",
                            visible=True
                        )
                        batch_token = gr.Slider(
                            minimum=1000,
                            maximum=50000,
                            value=2000,
                            step=1000,
                            label="Tokens per Batch",
                            visible=False
                        )
            with gr.Row():
                preview_single_button = gr.Button("Preview Command", elem_classes=["preview-command-btn"])
                predict_button = gr.Button("Start Prediction", variant="primary", elem_classes=["train-btn"])
                abort_button = gr.Button("Abort Prediction", variant="stop", elem_classes=["abort-btn"])

        with gr.Column(scale=3):
            gr.Markdown("## Results")
            command_preview = gr.Code(
                label="Command Preview",
                language="shell",
                interactive=False,
                visible=False
            )
            predict_output = gr.HTML(
                value="<div style='padding: 15px; background-color: #f5f5f5; border-radius: 5px;'><p style='margin: 0;'>Click the 「Start Prediction」 button to run prediction</p></div>",
                label="Prediction Results",
                padding=True
            )
            single_result_file = gr.DownloadButton(label="Download Results", visible=False)

    def toggle_input_method(method):
        if method == "Upload file":
            return gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)
        if method == "Specify FASTA path":
            return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)
        return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)

    input_method.change(
        fn=toggle_input_method,
        inputs=[input_method],
        outputs=[paste_seq_column, upload_seq_column, path_seq_column]
    )

    def resolve_sequence_then_predict(input_method_val, aa_seq_val, seq_file, seq_path_val, plm_model, model_path, eval_method, structure_seq, pooling_method, problem_type, num_labels, batch_size_val, predict_pdb_dir_val=None):
        """Parse multi-FASTA from paste, file upload, or path, create CSV, run batch prediction. Must be a generator for Gradio."""
        if input_method_val == "Upload file" and seq_file is not None:
            records = _parse_fasta_from_file(seq_file)
        elif input_method_val == "Specify FASTA path" and seq_path_val and os.path.isfile(str(seq_path_val).strip()):
            records = _parse_fasta_from_file(type("PathLike", (), {"name": str(seq_path_val).strip()})())
        else:
            records = _parse_fasta_sequences((aa_seq_val or "").strip())
        # Structure model with only pdb_dir: build minimal CSV from PDB files in directory
        is_structure_plm = bool(plm_model and (str(plm_model).startswith("ProSST") or str(plm_model).startswith("SaProt") or str(plm_model).startswith("ProtSSN")))
        if not records and is_structure_plm and predict_pdb_dir_val and os.path.isdir(str(predict_pdb_dir_val).strip()):
            pdb_path = Path(str(predict_pdb_dir_val).strip())
            records = [(p.stem, "") for p in sorted(pdb_path.rglob("*.pdb"))]
        if not records:
            yield (
                f"""
                {get_css_style_tag('prediction_ui.css')}
                {load_html_template('prediction_error.html', error_message="Please paste one or more FASTA sequences, upload a FASTA file, specify a FASTA path, or (for structure models) provide a PDB directory.")}
                """,
                gr.update(visible=False)
            )
            return
        temp_dir = tempfile.mkdtemp()
        csv_path = os.path.join(temp_dir, "input.csv")
        df = pd.DataFrame([{"id": rid, "aa_seq": seq} for rid, seq in records])
        df.to_csv(csv_path, index=False)
        file_like = type("FileLike", (), {"name": csv_path})()
        yield from predict_batch(
            plm_model, model_path, eval_method, file_like, structure_seq,
            pooling_method, problem_type, num_labels, batch_size_val, predict_pdb_dir_val
        )

    def toggle_preview(button_text):
        if "Preview" in button_text:
            return gr.update(visible=True)
        return gr.update(visible=False)

    def toggle_batch_mode(mode):
        if mode == "Batch Token Mode":
            return gr.update(visible=False), gr.update(visible=True)
        return gr.update(visible=True), gr.update(visible=False)

    batch_mode.change(
        fn=toggle_batch_mode,
        inputs=[batch_mode],
        outputs=[batch_size, batch_token]
    )

    preview_single_button.click(
        fn=toggle_preview,
        inputs=[preview_single_button],
        outputs=[command_preview]
    ).then(
        fn=handle_predict_tab_command_preview,
        inputs=[
            input_method,
            aa_seq,
            seq_file_upload,
            plm_model,
            model_dropdown,
            eval_method,
            foldseek_seq,
            ss8_seq,
            structure_seq,
            pooling_method,
            problem_type,
            num_labels,
            predict_pdb_dir,
        ],
        outputs=[command_preview]
    )
    predict_button.click(
        fn=resolve_sequence_then_predict,
        inputs=[
            input_method,
            aa_seq,
            seq_file_upload,
            seq_path_input,
            plm_model,
            model_dropdown,
            eval_method,
            structure_seq,
            pooling_method,
            problem_type,
            num_labels,
            batch_size,
            predict_pdb_dir
        ],
        outputs=[predict_output, single_result_file]
    )
    abort_button.click(
        fn=handle_abort_single,
        inputs=[],
        outputs=[predict_output, single_result_file]
    )

    # Event handlers after UI components
    def _predict_plm_needs_structure(plm: str) -> bool:
        return bool(plm and (str(plm).startswith("ProSST") or str(plm).startswith("SaProt") or str(plm).startswith("ProtSSN")))

    def update_predict_visibility(method, plm, pdb_dir_val=None):
        is_structure = _predict_plm_needs_structure(plm)
        is_ses = method == "ses-adapter"
        show_pdb = is_structure or is_ses
        show_input = not is_structure
        show_structure_seq_row = is_ses
        has_pdb = bool(pdb_dir_val and str(pdb_dir_val).strip())
        show_structure_input = is_ses and not has_pdb
        return (
            gr.update(visible=show_pdb),
            gr.update(visible=show_input),
            gr.update(visible=show_structure_seq_row),
            gr.update(visible=show_structure_input),
        )

    eval_method.change(
        fn=lambda m, p: update_predict_visibility(m, p, None),
        inputs=[eval_method, plm_model],
        outputs=[predict_pdb_dir_row, input_data_column, structure_seq_row, structure_input_row]
    )
    plm_model.change(
        fn=lambda m, p: update_predict_visibility(m, p, None),
        inputs=[eval_method, plm_model],
        outputs=[predict_pdb_dir_row, input_data_column, structure_seq_row, structure_input_row]
    )
    predict_pdb_dir.change(
        fn=update_predict_visibility,
        inputs=[eval_method, plm_model, predict_pdb_dir],
        outputs=[predict_pdb_dir_row, input_data_column, structure_seq_row, structure_input_row]
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
    


    def update_components_based_on_model(plm_model):
        is_proprime = (plm_model == "ProPrime-650M-OGT")
    
        # common update params
        update_params = {
            "interactive": not is_proprime,
        }
        
        otg_message_update = gr.update(
            visible=is_proprime,
            value=f"{get_css_style_tag('prediction_ui.css')}{load_html_template('otg_message.html')}"
        )

        if is_proprime:
            return {
                model_dropdown: gr.update(**update_params),
                eval_method: gr.update(**update_params),
                pooling_method: gr.update( **update_params),
                num_labels: gr.update(value=1, **update_params),
                problem_type: gr.update(value="regression", **update_params),
                otg_message: otg_message_update
            }
        else:
            return {
                model_dropdown: gr.update(**update_params),
                eval_method: gr.update(**update_params),
                pooling_method: gr.update(**update_params),
                num_labels: gr.update(**update_params),
                problem_type: gr.update(**update_params),
                otg_message: otg_message_update
            }



    plm_model.change(
        fn=update_components_based_on_model,
        inputs=[plm_model],
        outputs=[model_dropdown, eval_method, pooling_method, num_labels, problem_type, otg_message]
    )

    return {
        "predict_sequence": predict_sequence,
        "predict_batch": predict_batch,
        "handle_abort": handle_predict_tab_abort
    }