import threading
import queue
import subprocess
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any, Optional
from .command import build_command_list
import re

class TrainingMonitor:
    def __init__(self):
        self.message_queue = queue.Queue()
        self.process: Optional[subprocess.Popen] = None
        self.is_training = False
        self.training_thread: Optional[threading.Thread] = None
        
        # Training metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.val_metrics = {}  # 存储所有验证指标
        self.epochs = []
        self.current_epoch = 0
        
        self.error_message = None  # 添加错误消息存储
        
    def start_training(self, args: Dict[str, Any]):
        """Start training process with given arguments."""
        if self.is_training:
            return
            
        self.is_training = True
        self._reset_tracking()
        self.error_message = None  # 重置错误消息
        
        # Create and start training thread
        self.training_thread = threading.Thread(
            target=self._run_training,
            args=(args,)
        )
        self.training_thread.start()
    
    def abort_training(self):
        """Abort current training process."""
        if self.process:
            self.process.terminate()
            self.is_training = False
            self.message_queue.put("Training aborted by user.")
    
    def get_messages(self) -> str:
        """Get all messages from queue."""
        messages = []
        while not self.message_queue.empty():
            messages.append(self.message_queue.get())
        
        message_text = "\n".join(messages)
        
        # 如果有错误消息，添加到输出末尾
        if self.error_message:
            message_text += f"\n\nERROR: {self.error_message}"
            
        return message_text
    
    def get_plot(self):
        """Generate and return training progress plot."""
        if not self.train_losses:
            return None
            
        # 创建子图网格
        num_metrics = len(self.val_metrics) + 1  # +1 for loss
        fig, axes = plt.subplots(num_metrics, 1, figsize=(10, 4*num_metrics))
        if num_metrics == 1:
            axes = [axes]
        
        # 绘制损失图
        axes[0].plot(self.epochs, self.train_losses, label='Train Loss')
        if self.val_losses:
            axes[0].plot(self.epochs, self.val_losses, label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # 绘制其他指标
        for idx, (metric_name, metric_values) in enumerate(self.val_metrics.items(), 1):
            axes[idx].plot(self.epochs, metric_values, label=f'Validation {metric_name}')
            axes[idx].set_xlabel('Epoch')
            axes[idx].set_ylabel(metric_name)
            axes[idx].set_title(f'Validation {metric_name}')
            axes[idx].legend()
            axes[idx].grid(True)
        
        plt.tight_layout()
        return fig
    
    def _run_training(self, args: Dict[str, Any]):
        """Run training process."""
        try:
            # Build command
            cmd = build_command_list(args)
            
            # Start process
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # 合并stderr到stdout
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Read output
            while True:
                line = self.process.stdout.readline()
                if not line and self.process.poll() is not None:
                    break
                    
                if line:
                    line = line.strip()
                    self.message_queue.put(line)
                    self._process_output_line(line)
            
            # Check process return code
            if self.process.returncode != 0:
                error_msg = f"Training process exited with code {self.process.returncode}"
                self.error_message = error_msg
                self.message_queue.put(f"\nERROR: {error_msg}")
                
        except Exception as e:
            error_msg = f"Error during training: {str(e)}"
            self.error_message = error_msg
            self.message_queue.put(f"\nERROR: {error_msg}")
            
        finally:
            self.is_training = False
            self.process = None
    
    def _process_output_line(self, line: str):
        """Process training output line for metrics tracking."""
        try:
            # 处理训练损失
            if "Train Loss:" in line:
                epoch = int(re.search(r"Epoch (\d+)", line).group(1))
                train_loss = float(re.search(r"Train Loss: ([\d.]+)", line).group(1))
                self.current_epoch = epoch
                self.epochs.append(epoch)
                self.train_losses.append(train_loss)
            
            # 处理验证损失和指标
            elif "Validation Results:" in line:
                # 下一行会是验证结果，设置标志
                self.next_lines_are_metrics = True
            elif hasattr(self, 'next_lines_are_metrics') and self.next_lines_are_metrics:
                if "Loss:" in line:
                    val_loss = float(re.search(r"Loss: ([\d.]+)", line).group(1))
                    self.val_losses.append(val_loss)
                elif ":" in line:  # 其他指标
                    metric_name, value = line.split(":", 1)
                    metric_name = metric_name.strip()
                    value = float(value.strip())
                    if metric_name not in self.val_metrics:
                        self.val_metrics[metric_name] = []
                    self.val_metrics[metric_name].append(value)
                else:
                    self.next_lines_are_metrics = False
                
        except (ValueError, AttributeError, IndexError) as e:
            print(f"Error processing line: {line}, Error: {str(e)}")
    
    def _reset_tracking(self):
        """Reset metrics tracking."""
        self.train_losses = []
        self.val_losses = []
        self.val_metrics = {}
        self.epochs = []
        self.current_epoch = 0