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
from web.utils.command import preview_predict_command
from web.utils.html_ui import load_html_template, generate_prediction_status_html, generate_prediction_results_html, generate_batch_prediction_results_html, generate_table_rows
from web.utils.css_loader import get_css_style_tag
from datetime import datetime

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
                # For single-label classification
                predicted_class = prediction_data.get('predicted_class', 0)
                probabilities = prediction_data.get('probabilities', [])
                
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

    def predict_batch(plm_model, model_path, eval_method, input_file, eval_structure_seq, pooling_method, problem_type, num_labels, batch_size):
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
        
        # Initialize progress tracking with completely fresh state
        progress_info = {
            "total": 0,
            "completed": 0,
            "current_step": "Initializing",
            "status": "running",
            "lines": []  # Store lines for error handling
        }
        
        # Generate completely empty initial progress display
        initial_progress_html = f"""
        {get_css_style_tag('prediction_ui.css')}
        {load_html_template('batch_prediction_initializing.html')}
        """
        
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
            yield generate_progress_html(progress_info), gr.update(visible=False)
            
            # Create temporary file to save uploaded file
            temp_dir = tempfile.mkdtemp()
            input_path = os.path.join(temp_dir, "input.csv")
            output_dir = temp_dir  # Use the same temporary directory as output directory
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
                    progress_info["total"] = len(df)
                    progress_info["current_step"] = f"Found {len(df)} sequences to process"
                    yield generate_progress_html(progress_info), gr.update(visible=False)
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
            yield generate_progress_html(progress_info), gr.update(visible=False)
            
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
            yield generate_progress_html(progress_info), gr.update(visible=False)
            
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
                        except queue.Empty:
                            break
                    
                    # Check if the process has been aborted before updating UI
                    if process_aborted:
                        break
                        
                    # Check if we need to update the UI
                    current_time = time.time()
                    if new_lines or (current_time - last_update_time >= 0.5):
                        yield generate_progress_html(progress_info), gr.update(visible=False)
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

    def generate_progress_html(progress_info):
        """Generate HTML progress bar similar to eval_tab"""
        current = progress_info.get("completed", 0)
        total = max(progress_info.get("total", 1), 1)  # Avoid division by zero
        percentage = min(100, int((current / total) * 100))
        stage = progress_info.get("current_step", "Preparing")
        
        # ensure percentage is between 0-100
        percentage = max(0, min(100, percentage))
        
        # prepare detailed information
        total_sequences_detail = load_html_template('progress_detail_total.html', total=total) if total > 0 else ''
        progress_detail = load_html_template('progress_detail_current.html', current=current, total=total) if current > 0 and total > 0 else ''
        status_detail = load_html_template('progress_detail_status.html', status=progress_info.get("status", "").capitalize()) if "status" in progress_info else ''
        
        # create more modern progress bar - fully match eval_tab's style
        return f"""
        {get_css_style_tag('prediction_ui.css')}
        {load_html_template('prediction_progress.html',
                          stage=stage,
                          percentage=percentage,
                          total_sequences_detail=total_sequences_detail,
                          progress_detail=progress_detail,
                          status_detail=status_detail)}
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

    def handle_predict_tab_command_preview(plm_model, model_path, eval_method, aa_seq, foldseek_seq, ss8_seq, eval_structure_seq, pooling_method, problem_type, num_labels):
        """Handle the preview command button click event
        Args:
            plm_model: plm model name
            model_path: model path
            eval_method: evaluation method
            aa_seq: amino acid sequence
            foldseek_seq: foldseek sequence
            ss8_seq: ss8 sequence
            eval_structure_seq: structure sequence (foldseek_seq, ss8_seq)
            pooling_method: pooling method (mean, attention1d, light_attention)
            problem_type: problem type (single_label_classification, multi_label_classification, regression, residue_single_label_classification)
            num_labels: number of labels
        Returns:
            command_preview: command preview
        """
        # 构建参数字典
        args_dict = {
            "model_path": model_path,
            "plm_model": plm_models[plm_model],
            "aa_seq": aa_seq,
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

    # Configuration Import Section
    with gr.Accordion("Import Prediction Configuration", open=False) as config_import_accordion:
        gr.Markdown("### Import your prediction config")
        with gr.Row():
            with gr.Column(scale=4):
                config_path_input = gr.Textbox(
                    label="Configuration File Path",
                    placeholder="Enter path to your prediction config JSON file (e.g., ./predict_config.json)",
                    value=""
                )
            with gr.Column(scale=1):
                import_config_button = gr.Button(
                    "Import Config",
                    variant="primary",
                    elem_classes=["import-config-btn"]
                )

    gr.Markdown("## Model Configuration")
    with gr.Group():
        with gr.Row():
            model_path = gr.Textbox(
                label="Model Path",
                value="ckpt/demo/demo_solubility.pt",
                placeholder="Path to the trained model"
            )
            plm_model = gr.Dropdown(
                choices=list(plm_models.keys()),
                label="Protein Language Model"
            )


        with gr.Row():
            eval_method = gr.Dropdown(
                choices=["full", "freeze", "ses-adapter", "plm-lora", "plm-qlora", "plm_adalora", "plm_dora", "plm_ia3"],
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
                choices=["single_label_classification", "multi_label_classification", "regression", "residue_single_label_classification", "residue_regression"],
                label="Problem Type",
                value="single_label_classification"
            )
            num_labels = gr.Number(
                value=2,
                label="Number of Labels",
                precision=0,
                minimum=1
            )

        with gr.Row():
            otg_message = gr.HTML(
                f"""
                {get_css_style_tag('prediction_ui.css')}
                """,
                visible=False
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
                preview_single_button = gr.Button("Preview Command", elem_classes=["preview-command-btn"])
                predict_button = gr.Button("Start Prediction", variant="primary", elem_classes=["train-btn"])
                abort_button = gr.Button("Abort Prediction", variant="stop", elem_classes=["abort-btn"])
            
            # add command preview area
            command_preview = gr.Code(
                label="Command Preview",
                language="shell",
                interactive=False,
                visible=False
            )
            predict_output = gr.HTML(label="Prediction Results")
            single_result_file = gr.DownloadButton(label="Download Results", visible=False)
            
            
            
            
            predict_button.click(
                fn=predict_sequence,
                inputs=[
                    plm_model,
                    model_path,
                    aa_seq,
                    eval_method,
                    structure_seq,
                    pooling_method,
                    problem_type,
                    num_labels
                ],
                outputs=[predict_output, single_result_file]
            )
            
            abort_button.click(
                fn=handle_abort_single,
                inputs=[],
                outputs=[predict_output, single_result_file]
            )
        
        with gr.Tab("Batch Prediction"):
            gr.Markdown("### Batch Prediction")
            # display CSV format information with improved styling
            gr.HTML(f"""
            {get_css_style_tag('prediction_ui.css')}
            {load_html_template('csv_format_info.html')}
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
                    {get_css_style_tag('prediction_ui.css')}
                    {load_html_template('file_preview_table.html', 
                                      file_name=file.name.split('/')[-1],
                                      total_sequences=len(df),
                                      columns=', '.join(df.columns.tolist()))}
                    """
                    return gr.update(value=info_html), gr.update(value=df.head(5).values.tolist(), headers=df.columns.tolist()), gr.update(open=True)
                except Exception as e:
                    error_html = f"""
                    {get_css_style_tag('prediction_ui.css')}
                    {load_html_template('file_error.html', error_message=str(e))}
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
                preview_batch_button = gr.Button("Preview Command")
                batch_predict_button = gr.Button("Start Batch Prediction", variant="primary")
                batch_abort_button = gr.Button("Abort", variant="stop")
            
            # add command preview
            batch_command_preview = gr.Code(
                label="Command Preview",
                language="shell",
                interactive=False,
                visible=False
            )
            batch_predict_output = gr.HTML(label="Prediction Progress")
            result_file = gr.DownloadButton(label="Download Predictions", visible=False)

            # add command preview visibility control
            def toggle_preview(button_text):
                """toggle command preview visibility"""
                if "Preview" in button_text:
                    return gr.update(visible=True)
                return gr.update(visible=False)
            
            # connect preview button
            preview_single_button.click(
                fn=toggle_preview,
                inputs=[preview_single_button],
                outputs=[command_preview]
            ).then(
                fn=handle_predict_tab_command_preview,
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
                    num_labels,
                ],
                outputs=[command_preview]
            )
            
            # connect preview button
            preview_batch_button.click(
                fn=toggle_preview,
                inputs=[preview_batch_button],
                outputs=[batch_command_preview]
            ).then(
                fn=handle_batch_preview,
                inputs=[
                    plm_model,
                    model_path,
                    eval_method,
                    input_file,
                    structure_seq,
                    pooling_method,
                    problem_type,
                    num_labels,
                    batch_size,
                ],
                outputs=[batch_command_preview]
            )
            
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
                    batch_size,
                ],
                outputs=[batch_predict_output, result_file]
            )
            
            batch_abort_button.click(
                fn=handle_abort_batch,
                inputs=[],
                outputs=[batch_predict_output, result_file]
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
                model_path: gr.update(**update_params),
                eval_method: gr.update(**update_params),
                pooling_method: gr.update( **update_params),
                num_labels: gr.update(value=1, **update_params),
                problem_type: gr.update(value="regression", **update_params),
                otg_message: otg_message_update
            }
        else:
            return {
                model_path: gr.update(**update_params),
                eval_method: gr.update(**update_params),
                pooling_method: gr.update(**update_params),
                num_labels: gr.update(**update_params),
                problem_type: gr.update(**update_params),
                otg_message: otg_message_update
            }



    plm_model.change(
        fn=update_components_based_on_model,
        inputs=[plm_model],
        outputs=[model_path, eval_method, pooling_method, num_labels, problem_type, otg_message]
    )

    # Configuration Import Handler
    def handle_config_import(config_path: str):
        """
        Loads a prediction configuration from a JSON file and updates the UI components.
        """
        try:
            if not config_path or not config_path.strip():
                gr.Warning("Please provide a configuration file path")
                return [gr.update() for _ in range(6)]
                
            if not os.path.exists(config_path):
                gr.Warning(f"Configuration file not found: {config_path}")
                return [gr.update() for _ in range(6)]
            
            with open(config_path, "r", encoding='utf-8') as f:
                config = json.load(f)
            
            def get_config_val(key, default):
                return config.get(key, default)
            
            # Extract and validate model_path
            model_path_value = get_config_val("model_path", "ckpt/demo/demo_solubility.pt")
            
            # Extract and map plm_model (config stores full path, UI uses key)
            plm_model_path = get_config_val("plm_model", "")
            plm_model_value = None
            # Try to find the key by matching the path value
            for key, path in plm_models.items():
                if path == plm_model_path:
                    plm_model_value = key
                    break
            # If not found, use the first available model
            if not plm_model_value and plm_models:
                plm_model_value = list(plm_models.keys())[0]
            
            # Validate eval_method
            eval_method_value = get_config_val("training_method", "freeze")
            valid_eval_methods = ["full", "freeze", "ses-adapter", "plm-lora", "plm-qlora", "plm_adalora", "plm_dora", "plm_ia3"]
            if eval_method_value not in valid_eval_methods:
                eval_method_value = "freeze"
            
            # Validate pooling_method
            pooling_method_value = get_config_val("pooling_method", "mean")
            if pooling_method_value not in ["mean", "attention1d", "light_attention"]:
                pooling_method_value = "mean"
            
            # Validate problem_type
            problem_type_value = get_config_val("problem_type", "single_label_classification")
            valid_problem_types = ["single_label_classification", "multi_label_classification", "regression", "residue_single_label_classification", "residue_regression"]
            if problem_type_value not in valid_problem_types:
                problem_type_value = "single_label_classification"
            
            
            # Extract num_labels
            num_labels_value = get_config_val("num_labels", 2)
            
            gr.Info(f"✅ Configuration successfully imported from {config_path}")
            
            return [
                gr.update(value=model_path_value),
                gr.update(value=plm_model_value),
                gr.update(value=eval_method_value),
                gr.update(value=pooling_method_value),
                gr.update(value=problem_type_value),
                gr.update(value=num_labels_value),
            ]
        
        except (json.JSONDecodeError, KeyError) as e:
            gr.Warning(f"❌ Error parsing configuration file: {str(e)}")
            return [gr.update() for _ in range(6)]
        except Exception as e:
            gr.Warning(f"❌ An unexpected error occurred during import: {str(e)}")
            return [gr.update() for _ in range(6)]
    
    # Bind import config button
    import_config_button.click(
        fn=handle_config_import,
        inputs=[config_path_input],
        outputs=[model_path, plm_model, eval_method, pooling_method, problem_type, num_labels]
    )

    return {
        "predict_sequence": predict_sequence,
        "predict_batch": predict_batch,
        "handle_abort": handle_predict_tab_abort
    }