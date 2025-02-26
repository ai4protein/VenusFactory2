import threading
import queue
import subprocess
import matplotlib.pyplot as plt
import numpy as np
import os
import signal
import re
import time
from typing import Dict, Any, Optional
from .command import build_command_list

class TrainingMonitor:
    def __init__(self):
        """Initialize training monitor."""
        # Queues for thread-safe data exchange
        self.stats_queue = queue.Queue()
        self.message_queue = queue.Queue()
        self.is_training = False
        self.stop_thread = False
        self.process = None
        self.training_thread = None
        self.debug_progress = False  # Enable for debug info
        
        # Metrics tracking
        self._reset_tracking()
        
        # Progress tracking
        self.current_progress = {
            'stage': 'Waiting',  # Training/Validation/Testing
            'progress': '',      # Progress bar text
            'epoch': 0,
            'current': 0,
            'total': 100,
            'total_epochs': 0,   # Add total_epochs field, for storing total training rounds
            'val_accuracy': 0.0,
            'best_accuracy': 0.0,
            'best_epoch': 0,
            'best_metric_name': 'accuracy',  # Name of the best metric
            'best_metric_value': 0.0,        # Value of the best metric
            'progress_detail': '',           # Detailed progress information
            'elapsed_time': '',              # Elapsed time
            'remaining_time': '',            # Remaining time
            'it_per_sec': 0.0,               # Iterations per second
            'grad_step': 0,                  # Gradient steps
            'loss': 0.0,                     # Loss value
            'test_metrics': {},              # Add test metrics container
            'test_progress': 0.0,            # Test progress percentage
            'test_results_html': ''          # HTML formatted test results
        }
        
        self.error_message = None
        self.skip_output_patterns = [
            r"Model Parameters Statistics:",
            r"Dataset Statistics:",
            r"Sample \d+ data points from train dataset:"
        ]
        
        # Simplified regex patterns
        self.patterns = {
            # Basic training log patterns
            'train': r'Epoch (\d+) Train Loss: ([\d.]+)',
            'val': r'Epoch (\d+) Val Loss: ([\d.]+)',
            'val_metric': r'Epoch (\d+) Val ([a-zA-Z_\-]+(?:\s[a-zA-Z_\-]+)*): ([\d.]+)',
            'epoch_header': r'---------- Epoch (\d+) ----------',
            'best_save': r'Saving model with best val ([a-zA-Z_\-]+(?:\s[a-zA-Z_\-]+)*): ([\d.]+)',
            
            # Test result patterns - improved to match log format exactly
            'test_header': r'Test Results:',
            'test_phase_start': r'---------- Starting Test Phase ----------',
            # Fix parsing for metrics like f1
            'test_metric': r'Test\s+([a-zA-Z0-9_\-]+(?:\s[a-zA-Z0-9_\-]+)*):?\s*([\d.]+)',
            # Add pattern for matching common metrics
            'test_common_metrics': r'Test\s+((?:f1|accuracy|precision|recall|auroc|mcc)):\s*([\d.]+)',
            # Specific pattern for test loss that matches "Test Loss: 0.3370" format
            'test_loss': r'Test\s+Loss:\s*([\d.]+)',
            # Alternative format pattern
            'test_alt_format': r'([a-zA-Z0-9_\-]+(?:\s[a-zA-Z0-9_\-]+)*)\s+on\s+test:\s*([\d.]+)',
            
            # Model parameter statistics
            'model_param': r'([A-Za-z\s]+):\s*([\d,.]+[KM]?)',
        }
        
        # Progress bar patterns - Updated to handle both Validating and Testing phases
        self.progress_patterns = {
            'train': r'Training:\s*(\d+)%\|[^|]*\|\s*(\d+)/(\d+)\s*\[([\d:]+)<([\d:]+),\s*([\d.]+)it/s(?:,\s*grad_step=(\d+),\s*train_loss=([\d.]+))?\]',
            # Combined pattern for both Validating and Testing since they use same tqdm format
            'valid_or_test': r'(?:Validating|Valid|Testing|Test):\s*(\d+)%\|[^|]*\|\s*(\d+)/(\d+)\s*\[([\d:]+)<([\d:]+),\s*([\d.]+)it/s(?:[^\]]*)\]',
        }
        
        # Test results storage
        self.test_results = {}
        self.parsing_test_results = False
        self.test_results_table = None
        self.test_results_html = None

    def _should_skip_line(self, line: str) -> bool:
        """Check if the line should be skipped from output."""
        for pattern in self.skip_output_patterns:
            if re.search(pattern, line):
                return True
        return False
    
    def _process_output(self, process):
        """Process output from training process in real-time."""
        while True:
            if self.stop_thread:
                break
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                line = output.strip()
                if not self._should_skip_line(line):
                    self.message_queue.put(line)
                self._process_output_line(line)
        process.stdout.close()
        
    def start_training(self, args: Dict[str, Any]):
        """Start training process with given arguments."""
        if self.is_training:
            return
            
        self.is_training = True
        self.stop_thread = False
        self._reset_tracking()
        self._reset_stats()  # Ensure stats are reset
        self.error_message = None
        
        # Extract total epochs from training parameters and save to current_progress
        self.current_progress['total_epochs'] = args.get('num_epochs', 100)
        
        try:
            # Build command
            cmd = build_command_list(args)
            
            # Start process
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                preexec_fn=os.setsid
            )
            
            # Start output processing thread
            self.training_thread = threading.Thread(
                    target=self._process_output,
                    args=(self.process,)
            )
            self.training_thread.daemon = True
            self.training_thread.start()
                
        except Exception as e:
            self.error_message = f"Error starting training: {str(e)}"
            self.is_training = False
            self.message_queue.put(f"ERROR: {self.error_message}")
    
    def abort_training(self):
        """Abort current training process."""
        if self.process:
            try:
                self.stop_thread = True
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                except:
                    pass
            finally:
                self.is_training = False
                self.process = None
                self._reset_stats()  # Ensure all stats are reset on termination
            self.message_queue.put("Training aborted by user.")
    
    def get_messages(self) -> str:
        """Get all messages from queue."""
        messages = []
        while not self.message_queue.empty():
            try:
                messages.append(self.message_queue.get_nowait())
            except queue.Empty:
                break
        
        message_text = "\n".join(messages)
        
        if self.error_message:
            message_text += f"\n\nERROR: {self.error_message}"
            
        return message_text
    
    def get_plot(self):
        """
        Generate a plot showing training and validation losses.
        
        Returns:
            matplotlib Figure object, or None if insufficient data
        """
        # Return None if insufficient data
        if not self.epochs or (not self.train_losses and not self.val_losses):
            return None
        
        # Ensure training losses array matches epochs array length
        if len(self.train_losses) < len(self.epochs):
            while len(self.train_losses) < len(self.epochs):
                self.train_losses.append(None)
        
        # Create new plot with improved style
        plt.close('all')  # Close previous plots
        try:
            plt.style.use('seaborn-v0_8-whitegrid')  # Use seaborn style for better visuals
        except:
            try:
                plt.style.use('seaborn-whitegrid')
            except:
                pass
        
        fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
        
        # Set background color for better visibility
        ax.set_facecolor('#f8f9fa')
        fig.patch.set_facecolor('white')
        
        # Initialize variables
        x_train, y_train = [], []
        x_val, y_val = [], []
        val_loss_to_epoch = {}  # Initialize here to fix reference error
        
        # Process training loss data
        if self.train_losses:
            for i, (ep, loss) in enumerate(zip(self.epochs, self.train_losses)):
                if loss is not None:
                    x_train.append(ep)
                    y_train.append(loss)
            
            if x_train:
                ax.plot(x_train, y_train, 
                        label='Training Loss',
                        color='#1f77b4',  # Modern blue color
                        marker='o',
                        markersize=5,
                        linewidth=2.5,
                        alpha=0.9,
                        zorder=3)
        
        # Process validation loss data
        if self.val_losses:
            # Ensure val_losses array is properly sized
            while len(self.val_losses) < len(self.epochs):
                self.val_losses.append(None)
                
            for i, ep in enumerate(self.epochs):
                if i < len(self.val_losses) and self.val_losses[i] is not None:
                    x_val.append(ep)
                    y_val.append(self.val_losses[i])
                    val_loss_to_epoch[ep] = self.val_losses[i]
            
            if x_val:
                ax.plot(x_val, y_val, 
                        label='Validation Loss',
                        color='#ff7f0e',  # Modern orange color
                        marker='s',
                        markersize=5,
                        linewidth=2.5,
                        alpha=0.9,
                        zorder=4)
        
        # Set plot title and labels with improved styling
        ax.set_title('Training and Validation Loss', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax.set_ylabel('Loss', fontsize=12, fontweight='bold')
        
        # Improve grid appearance
        ax.grid(True, linestyle='--', alpha=0.7, zorder=0)
        
        # Improve tick labels
        ax.tick_params(axis='both', which='major', labelsize=10)
        
        # Improve legend
        leg = ax.legend(loc='upper right', fontsize=11, frameon=True, fancybox=True, 
                       framealpha=0.9, edgecolor='gray')
        leg.get_frame().set_linewidth(1.0)
        
        # Set y-axis to start from 0 if all values are positive
        if self.train_losses and self.val_losses:
            all_losses = [l for l in self.train_losses + self.val_losses if l is not None]
            if all_losses and min(all_losses) >= 0:
                y_min = 0
                y_max = max(all_losses) * 1.1  # Add 10% padding at the top
                ax.set_ylim(y_min, y_max)
        
        # Mark best model position (if available)
        best_epoch = self.current_progress.get('best_epoch', 0)
        best_metric_name = self.current_progress.get('best_metric_name', '')
        best_metric_value = self.current_progress.get('best_metric_value', 0.0)
        
        # Only mark best model if validation loss and best epoch are valid
        if best_epoch > 0 and val_loss_to_epoch and best_epoch in val_loss_to_epoch:
            best_y = val_loss_to_epoch[best_epoch]
            ax.scatter([best_epoch], [best_y], color='#d62728', s=120, zorder=5, 
                      marker='*', edgecolor='white', linewidth=1.5)
            
            # Add an annotation for the best model
            label_text = f'Best Model\n{best_metric_name}: {best_metric_value:.4f}'
            ax.annotate(label_text, 
                       xy=(best_epoch, best_y),
                       xytext=(10, -20),
                       textcoords='offset points',
                       color='#d62728',
                       fontsize=10,
                       fontweight='bold',
                       arrowprops=dict(arrowstyle='->',
                                      connectionstyle='arc3,rad=.2',
                                      color='#d62728'))
        
        # Remove top and right spines for cleaner look
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
        
        # Add light horizontal lines to improve readability
        if self.train_losses or self.val_losses:
            ax.yaxis.grid(True, linestyle='-', alpha=0.1, color='gray')
        
        # Ensure the x-axis shows integer epochs
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        
        # Tight layout to optimize use of space
        fig.tight_layout(pad=3.0)
        
        return fig
    
    def get_progress(self) -> Dict[str, Any]:
        """Return current progress information."""
        return self.current_progress.copy()
    
    def _process_output_line(self, line: str):
        """Process training output line for metric tracking."""
        try:
            # Always check for test progress if in Testing stage
            if self.current_progress.get('stage') == 'Testing':
                if self._process_test_progress(line):
                    return
                    
            # Check for test phase start
            if re.search(self.patterns['test_phase_start'], line):
                self.current_progress['stage'] = 'Testing'
                # Reset test metrics at the start of test phase
                self.current_progress['test_metrics'] = {}
                self.current_progress['test_results_html'] = ''
                return
                
            # Check for epoch header pattern (e.g., "---------- Epoch 1 ----------")
            epoch_header_match = re.search(self.patterns['epoch_header'], line)
            if epoch_header_match:
                new_epoch = int(epoch_header_match.group(1))
                # Update current epoch
                self.current_epoch = new_epoch
                self.current_progress['epoch'] = new_epoch
                if self.debug_progress:
                    print(f"Detected epoch header, setting current epoch to: {new_epoch}")
                return
            
            # Detect test results header
            if re.search(self.patterns['test_header'], line):
                self.parsing_test_results = True
                self.test_results = {}
                # Set stage to 'Testing' when we see the test results header
                self.current_progress['stage'] = 'Testing'
                return
                
            # Parse test results - handle log lines with timestamps and INFO prefixes
            # Extract the actual content part of the log line if it contains timestamp and INFO
            log_content = line
            log_match = re.search(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} - [a-zA-Z]+ - INFO - (.*)', line)
            if log_match:
                log_content = log_match.group(1)
                
            if self.parsing_test_results:
                collected_new_metric = False
                
                # Try to match test loss value
                test_loss_match = re.search(self.patterns['test_loss'], log_content)
                if test_loss_match:
                    loss_value = float(test_loss_match.group(1))
                    self.test_results['loss'] = loss_value
                    collected_new_metric = True
                
                # First try to match common metrics (f1, accuracy, etc.)
                if not test_loss_match:
                    common_metric_match = re.search(self.patterns['test_common_metrics'], log_content)
                    if common_metric_match:
                        metric_name, metric_value = common_metric_match.groups()
                        metric_name = metric_name.strip().lower()
                        try:
                            self.test_results[metric_name] = float(metric_value)
                            collected_new_metric = True
                        except ValueError:
                            if self.debug_progress:
                                print(f"Failed to parse value for common metric {metric_name}: {metric_value}")
                
                # Match other test metrics - Fix parsing for metrics like f1
                if not test_loss_match and not (locals().get('common_metric_match')):  # Avoid duplicate matching
                    test_metric_match = re.search(self.patterns['test_metric'], log_content)
                    if test_metric_match:
                        metric_name, metric_value = test_metric_match.groups()
                        metric_name = metric_name.strip().lower()
                        
                        # Special handling for f1 metric to ensure correct parsing
                        if metric_name == "f" and metric_value.startswith("1"):
                            # Fix the case where f1 is incorrectly split
                            metric_name = "f1"
                            metric_value = metric_value[1:].strip()  # Remove the leading "1"
                        
                        try:
                            self.test_results[metric_name] = float(metric_value)
                            collected_new_metric = True
                        except ValueError:
                            if self.debug_progress:
                                print(f"Failed to parse value for metric {metric_name}: {metric_value}")
                
                # Try to match alternative format
                if not test_loss_match and not test_metric_match:
                    alt_format_match = re.search(self.patterns['test_alt_format'], log_content)
                    if alt_format_match:
                        metric_name, metric_value = alt_format_match.groups()
                        metric_name = metric_name.strip().lower()
                        try:
                            self.test_results[metric_name] = float(metric_value)
                            collected_new_metric = True
                        except ValueError:
                            if self.debug_progress:
                                print(f"Failed to parse value for alt metric {metric_name}: {metric_value}")
                
                # If we collected a new metric, update the display
                if collected_new_metric:
                    self._update_test_results_display()
                
                # Determine if we should end test results parsing
                # Only end parsing when line doesn't start with "Test", is not empty, and we've collected metrics, or if line is empty
                if ((not log_content.strip().startswith("Test") and 
                    len(log_content.strip()) > 0 and 
                    self.test_results) or 
                    log_content.strip() == ""):
                    
                    # Ensure we've collected at least some metrics before ending parsing
                    if self.test_results:
                        self.parsing_test_results = False
                        # Final update of the display
                        self._update_test_results_display()
                
                return
            
            # Parse model parameter statistics
            if "Model Parameters Statistics:" in line:
                self.current_stats = {}
                self.parsing_stats = True
                self.skipped_first_separator = False
                return
            
            if self.parsing_stats:
                # Handle separator line
                if "------------------------" in line:
                    # If this is the first separator line, skip it
                    if not self.skipped_first_separator:
                        self.skipped_first_separator = True
                        return
                    
                    # If it's the last separator line, check if we have enough information
                    required_keys = ["adapter_total", "adapter_trainable", 
                                    "pretrain_total", "pretrain_trainable", 
                                    "combined_total", "combined_trainable", 
                                    "trainable_percentage"]
                    
                    missing_keys = [key for key in required_keys if key not in self.current_stats]
                    
                    if not missing_keys:
                        # Put statistics in queue
                        self.stats_queue.put(self.current_stats.copy())
                        # Update cache
                        self.last_stats.update(self.current_stats)
                    
                    self.parsing_stats = False
                    self.current_model = None
                    self.skipped_first_separator = False
                    return
                
                # If first separator not yet skipped, don't process other lines
                if not self.skipped_first_separator:
                    return
                
                # Match model name sections
                if "Adapter Model:" in line:
                    self.current_model = "adapter"
                    return
                elif "Pre-trained Model:" in line:
                    self.current_model = "pretrain"
                    return
                elif "Combined:" in line:
                    self.current_model = "combined"
                    return
                
                # Parse parameter information
                param_match = re.search(self.patterns['model_param'], line)
                if param_match and self.current_model:
                    stat_name, stat_value = param_match.groups()
                    stat_name = stat_name.strip().lower()
                    
                    if "total parameters" in stat_name:
                        self.current_stats[f"{self.current_model}_total"] = stat_value
                    elif "trainable parameters" in stat_name:
                        self.current_stats[f"{self.current_model}_trainable"] = stat_value
                    elif "trainable percentage" in stat_name and self.current_model == "combined":
                        self.current_stats["trainable_percentage"] = stat_value
                return
            
            # Process training progress
            train_progress_match = re.search(self.progress_patterns['train'], line)
            if train_progress_match:
                percentage, current, total, elapsed, remaining, it_per_sec = train_progress_match.groups()[:6]
                grad_step = train_progress_match.group(7) if len(train_progress_match.groups()) >= 7 and train_progress_match.group(7) else "0"
                loss = train_progress_match.group(8) if len(train_progress_match.groups()) >= 8 and train_progress_match.group(8) else "0.0"
                
                # Update progress information
                self.current_progress['stage'] = 'Training'
                self.current_progress['current'] = int(current)
                self.current_progress['total'] = int(total)
                self.current_progress['progress_detail'] = f"{current}/{total}[{elapsed}<{remaining},{it_per_sec}it/s"
                if grad_step:
                    self.current_progress['progress_detail'] += f",grad_step={grad_step}"
                self.current_progress['progress_detail'] += f",train_loss={loss}]"
                self.current_progress['elapsed_time'] = elapsed
                self.current_progress['remaining_time'] = remaining
                self.current_progress['it_per_sec'] = float(it_per_sec)
                if grad_step:
                    self.current_progress['grad_step'] = int(grad_step)
                if loss and float(loss) > 0:
                    self.current_progress['loss'] = float(loss)
                return
            
            # Validation or Testing progress - consolidated since they use same tqdm format
            valid_or_test_match = re.search(self.progress_patterns['valid_or_test'], line)
            if valid_or_test_match:
                percentage, current, total, elapsed, remaining, it_per_sec = valid_or_test_match.groups()
                
                # Determine stage based on current context and line content
                # If line contains 'Test' or we've already detected test phase, set to 'Testing'
                if 'Test' in line or self.current_progress.get('stage') == 'Testing' or self.parsing_test_results:
                    self.current_progress['stage'] = 'Testing'
                else:
                    self.current_progress['stage'] = 'Validation'
                
                self.current_progress['current'] = int(current)
                self.current_progress['total'] = int(total)
                self.current_progress['progress_detail'] = f"{current}/{total}[{elapsed}<{remaining},{it_per_sec}it/s]"
                self.current_progress['elapsed_time'] = elapsed
                self.current_progress['remaining_time'] = remaining
                self.current_progress['it_per_sec'] = float(it_per_sec)
                return
            
            # Parse training loss
            train_match = re.search(self.patterns['train'], line)
            if train_match:
                epoch, loss = train_match.groups()
                current_epoch = int(epoch)
                
                self.current_progress['epoch'] = current_epoch
                self.current_progress['loss'] = float(loss)
                self.current_epoch = current_epoch
                
                # Add new epoch to epochs list
                if current_epoch not in self.epochs:
                    self.epochs.append(current_epoch)
                    self.train_losses.append(float(loss))
                else:
                    # Update existing epoch
                    idx = self.epochs.index(current_epoch)
                    self.train_losses[idx] = float(loss)
                return
            
            # Parse validation loss
            val_match = re.search(self.patterns['val'], line)
            if val_match:
                epoch, loss = val_match.groups()
                current_epoch = int(epoch)
                
                # Ensure current epoch exists
                if current_epoch not in self.epochs:
                    self.epochs.append(current_epoch)
                    if len(self.train_losses) < len(self.epochs):
                        self.train_losses.append(None)
                
                idx = self.epochs.index(current_epoch)
                
                # Ensure val_losses list matches epochs list length
                while len(self.val_losses) < len(self.epochs):
                    self.val_losses.append(None)
                
                # Update val_losses at correct position
                self.val_losses[idx] = float(loss)
                
                # Also update val_metrics dictionary
                if 'loss' not in self.val_metrics:
                    self.val_metrics['loss'] = []
                
                # Ensure val_metrics['loss'] matches epochs length
                while len(self.val_metrics['loss']) < len(self.epochs):
                    self.val_metrics['loss'].append(None)
                
                # Update val_metrics['loss'] at correct position
                self.val_metrics['loss'][idx] = float(loss)
                return
            
            # Parse validation metrics
            val_metric_match = re.search(self.patterns['val_metric'], line)
            if val_metric_match:
                epoch, metric_name, metric_value = val_metric_match.groups()
                current_epoch = int(epoch)
                metric_name = metric_name.strip().lower()
                
                # Handle different possible metrics
                if metric_name == 'accuracy' or metric_name == 'acc':
                    self.current_progress['val_accuracy'] = float(metric_value)
                
                # Ensure current epoch exists
                if current_epoch not in self.epochs:
                    self.epochs.append(current_epoch)
                    if len(self.train_losses) < len(self.epochs):
                        self.train_losses.append(None)
                
                # Add to corresponding metric list
                if metric_name not in self.val_metrics:
                    self.val_metrics[metric_name] = []
                
                # Ensure list length matches epochs
                while len(self.val_metrics[metric_name]) < len(self.epochs):
                    self.val_metrics[metric_name].append(None)
                
                idx = self.epochs.index(current_epoch)
                self.val_metrics[metric_name][idx] = float(metric_value)
                return
            
            # Match best model save info: e.g., "Saving model with best val accuracy: 0.9088"
            best_save_match = re.search(self.patterns['best_save'], line)
            if best_save_match:
                metric_name, metric_value = best_save_match.groups()
                metric_name = metric_name.strip().lower()
                metric_value = float(metric_value)
                
                self.current_progress['best_metric_name'] = metric_name
                self.current_progress['best_metric_value'] = metric_value
                self.current_progress['best_epoch'] = self.current_epoch
                
                # If accuracy metric
                if metric_name == 'accuracy':
                    self.current_progress['best_accuracy'] = metric_value
                
                if self.debug_progress:
                    print(f"Best model updated - Metric: {metric_name}, Value: {metric_value}, Epoch: {self.current_epoch}")
                return
                
        except Exception as e:
            # Don't raise exception to avoid interrupting the processing thread
            # But record the error message
            self.error_message = f"Failed to parse output line: {str(e)}"
            if self.debug_progress:
                print(f"Error parsing line: {str(e)}")
                print(f"Line content: {line}")
    
    def _reset_tracking(self):
        """Reset metrics tracking."""
        self.train_losses = []
        self.val_losses = []
        self.val_metrics = {}
        self.epochs = []
        self.current_epoch = 0

    def get_stats(self) -> Dict:
        """Get collected statistics."""
        # Save last retrieved statistics to avoid emptying queue every time
        if not hasattr(self, 'last_stats'):
            self.last_stats = {}
        
        try:
            # Check if there's new data in the queue
            if not self.stats_queue.empty():
                # Get the latest statistics data
                while not self.stats_queue.empty():
                    stat = self.stats_queue.get_nowait()
                    self.last_stats.update(stat)
        except queue.Empty:
            pass
        except Exception as e:
            print(f"Error getting statistics data: {str(e)}")
        
        return self.last_stats

    def _reset_stats(self):
        """Reset statistics tracking."""
        # Clear statistics queue
        while not self.stats_queue.empty():
            try:
                self.stats_queue.get_nowait()
            except queue.Empty:
                break
                
        # Reset current statistics
        self.current_stats = {}
        self.parsing_stats = False
        self.current_model = None
        self.skipped_first_separator = False  # Reset flag
        
        # Reset cached statistics
        self.last_stats = {}
        
        # Reset training and validation metrics
        self._reset_tracking()
        
        # Reset progress info
        self.current_progress = {
            'stage': 'Waiting',
            'progress': '',
            'epoch': 0,
            'current': 0,
            'total': 100,
            'total_epochs': 0,  # Ensure total_epochs is reset
            'val_accuracy': 0.0,
            'best_accuracy': 0.0,
            'best_epoch': 0,
            'best_metric_name': 'accuracy',
            'best_metric_value': 0.0,
            'progress_detail': '',
            'elapsed_time': '',
            'remaining_time': '',
            'it_per_sec': 0.0,
            'grad_step': 0,
            'loss': 0.0,
            'test_metrics': {},
            'test_progress': 0.0,
            'test_results_html': ''
        }

    def _update_test_results_display(self):
        """Update the display of test results, in both HTML and text formats."""
        if not self.test_results:
            return
            
        # Count number of metrics
        metrics_count = len(self.test_results)
        
        # Create a more beautiful HTML table with summary information
        html_content = f"""
        <div style="max-width: 800px; margin: 0 auto; font-family: Arial, sans-serif;">
            <h3 style="text-align: center; margin-bottom: 15px; color: #333;">Test Results</h3>
            <p style="text-align: center; margin-bottom: 15px; color: #666;">{metrics_count} metrics found</p>
            <table style="width: 100%; border-collapse: collapse; font-size: 14px; border: 1px solid #ddd; box-shadow: 0 2px 3px rgba(0,0,0,0.1);">
                <thead>
                    <tr style="background-color: #f0f0f0;">
                        <th style="padding: 12px; text-align: left; border: 1px solid #ddd; font-weight: bold;">Metric</th>
                        <th style="padding: 12px; text-align: right; border: 1px solid #ddd; font-weight: bold;">Value</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        # Sort by priority and alphabetically to ensure important metrics are displayed first
        priority_metrics = ['loss', 'accuracy', 'f1', 'precision', 'recall', 'auroc', 'mcc']
        
        # Build priority sorting key
        def get_priority(item):
            name = item[0]
            if name in priority_metrics:
                return priority_metrics.index(name)
            return len(priority_metrics)
        
        # Sort by priority
        sorted_metrics = sorted(self.test_results.items(), key=get_priority)
        
        # Add a row for each metric, using alternating row colors
        for i, (metric_name, metric_value) in enumerate(sorted_metrics):
            row_style = 'background-color: #f9f9f9;' if i % 2 == 0 else ''
            color_style = self._get_metric_color(metric_name, metric_value)
            
            # Use bold for priority metrics
            is_priority = metric_name in priority_metrics
            name_style = 'font-weight: bold;' if is_priority else ''
            
            html_content += f"""
            <tr style="{row_style}">
                <td style="padding: 10px; text-align: left; border: 1px solid #ddd; {name_style}">{metric_name}</td>
                <td style="padding: 10px; text-align: right; border: 1px solid #ddd; {color_style}">{metric_value:.4f}</td>
            </tr>
            """
            
        html_content += """
                </tbody>
            </table>
            <p style="text-align: right; margin-top: 10px; color: #888; font-size: 12px;">Test completed at: """ + time.strftime("%Y-%m-%d %H:%M:%S") + """</p>
        </div>
        """
        
        # Save to current_progress for UI access
        self.current_progress['test_metrics'] = self.test_results.copy()
        self.current_progress['test_results_html'] = html_content
        
        # Generate text representation for logging
        text_results = "\nTest Results:\n" + "-" * 30 + "\n"
        
        # Display in same order as HTML
        for metric_name, metric_value in sorted_metrics:
            text_results += f"{metric_name.ljust(15)}: {metric_value:.4f}\n"
            
        text_results += "-" * 30
        text_results += f"\nTotal {metrics_count} metrics"
        
        # Add text results to message queue
        self.message_queue.put(text_results)

    def _process_test_progress(self, line: str):
        """Process test progress from output lines during testing phase."""
        # Parse intermediate test results if available
        test_metric_interim_match = re.search(r'Batch (\d+)/(\d+): ([a-zA-Z_\-]+) = ([\d.]+)', line)
        if test_metric_interim_match:
            batch, total_batches, metric_name, metric_value = test_metric_interim_match.groups()
            progress = int(batch) / int(total_batches) * 100
            self.current_progress['test_progress'] = progress
            
            # Update test metrics with interim values
            if 'interim_metrics' not in self.current_progress:
                self.current_progress['interim_metrics'] = {}
            
            self.current_progress['interim_metrics'][metric_name] = float(metric_value)
            return True
            
        return False

    def _get_metric_color(self, metric_name, value):
        """Get color styling for metrics based on their value"""
        # For metrics that are better when their value is higher (accuracy, F1, etc.)
        higher_better = ['accuracy', 'acc', 'f1', 'recall', 'precision', 'auroc', 'auprc', 'spearman', 
                        'pearson', 'spearman_corr', 'mcc', 'specificity', 'sensitivity']
        # For metrics that are better when their value is lower (loss, error rate, etc.)
        lower_better = ['loss', 'error', 'rmse', 'mae', 'mse']
        
        if any(better in metric_name.lower() for better in higher_better):
            if value > 0.9:
                return 'color: #0a3; font-weight: bold;'
            elif value > 0.8:
                return 'color: #4a4;'
            elif value > 0.7:
                return 'color: #6a6;'
            else:
                return ''
        elif any(better in metric_name.lower() for better in lower_better):
            if value < 0.1:
                return 'color: #0a3; font-weight: bold;'
            elif value < 0.3:
                return 'color: #4a4;'
            elif value < 0.5:
                return 'color: #6a6;'
            else:
                return ''
        return ''