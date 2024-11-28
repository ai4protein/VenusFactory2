import sys
import queue
import subprocess
import signal
import os
import seaborn as sns
from threading import Thread
import matplotlib.pyplot as plt
import io
import re

class TrainingMonitor:
    def __init__(self):
        self.message_queue = queue.Queue()
        self.is_training = False
        self.messages = []
        self.train_losses = []
        self.val_losses = []
        self.epochs = []
        self.current_epoch = 0
        self.total_epochs = 0
        self.progress = 0

    def start_training(self, args_dict):
        if not self.is_training:
            self.is_training = True
            self.messages = []
            self.train_losses = []
            self.val_losses = []
            self.epochs = []
            Thread(target=self._run_training, args=(args_dict,), daemon=True).start()

    def abort_training(self):
        if self.is_training:
            self.is_training = False
            print("Training aborted!")
            Thread(target=self._abort_training, daemon=True).start()
            
    def _abort_training(self):
        # 发送SIGINT信号到训练进程
        os.kill(self.process.pid, signal.SIGINT)
            
    
    def _run_training(self, args_dict):
        cmd = [sys.executable, "src/train.py"]
        for k, v in args_dict.items():
            if v is True:
                cmd.append(f"--{k}")
            elif v is not False and v is not None:
                cmd.append(f"--{k}")
                cmd.append(str(v))
        print(" ".join(cmd))
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True
            )
            
            self.process = process

            for line in process.stdout:
                self.message_queue.put(line.strip())
                if "EPOCH" in line and "TRAIN loss:" in line:
                    try:
                        epoch = int(re.search(r"EPOCH\s+(\d+)", line).group(1))
                        train_loss = float(re.search(r"TRAIN loss:\s+([\d.]+)", line).group(1))
                        self.epochs.append(epoch)
                        self.train_losses.append(train_loss)
                    except:
                        pass
                    
                if "VAL loss:" in line:
                    try:
                        val_loss = float(re.search(r"VAL loss:\s+([\d.]+)", line).group(1))
                        self.val_losses.append(val_loss)
                    except:
                        pass

            process.wait()
            if process.returncode == 0:
                self.message_queue.put("Training completed successfully!")
            else:
                self.message_queue.put(f"Training failed with return code {process.returncode}")
        except Exception as e:
            self.message_queue.put(f"Error occurred: {str(e)}")
        finally:
            self.is_training = False

    def get_messages(self):
        while not self.message_queue.empty():
            message = self.message_queue.get()
            self.messages.append(message)
            
            # 解析训练信息
            if "EPOCH" in message and "TRAIN loss:" in message:
                try:
                    epoch = int(re.search(r"EPOCH\s+(\d+)", message).group(1))
                    train_loss = float(re.search(r"TRAIN loss:\s+([\d.]+)", message).group(1))
                    self.epochs.append(epoch)
                    self.train_losses.append(train_loss)
                except:
                    pass
                    
            if "VAL loss:" in message:
                try:
                    val_loss = float(re.search(r"VAL loss:\s+([\d.]+)", message).group(1))
                    self.val_losses.append(val_loss)
                except:
                    pass
        
        return "\n".join(self.messages) if self.messages else ""

    def get_plot(self):
        # 创建训练曲线图
        plt.figure(figsize=(6, 4), dpi=100)
        
        sns.set_style("whitegrid")  # 使用seaborn样式美化图表
        
        if self.train_losses:
            plt.plot(self.epochs, self.train_losses, 
                    label='Train Loss',
                    color='#2196F3',  # 使用蓝色
                    marker='o',
                    markersize=4,
                    linewidth=2,
                    drawstyle='default')
        if self.val_losses:
            plt.plot(self.epochs[:len(self.val_losses)], self.val_losses,
                    label='Val Loss', 
                    color='#F44336',  # 使用红色
                    marker='s',
                    markersize=4,
                    linewidth=2,
                    drawstyle='default')
        
        plt.xlabel('Epoch', fontsize=10)
        plt.ylabel('Loss', fontsize=10)
        plt.legend(frameon=True, fontsize=9)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 设置背景色和边框
        plt.gca().set_facecolor('#f8f9fa')
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        
        plt.tight_layout()
        
        # 将图转换为可显示格式
        return plt

    def get_progress(self):
        """返回训练进度的HTML表示"""
        if not self.is_training:
            return ""
        
        if self.total_epochs > 0:
            progress = (self.current_epoch / self.total_epochs) * 100
            return f"""
                <div style="width: 100%; height: 20px; background-color: #f0f0f0; border-radius: 10px; overflow: hidden;">
                    <div style="width: {progress}%; height: 100%; background-color: #2196F3; transition: width 0.5s ease;">
                    </div>
                </div>
                <div style="text-align: center; margin-top: 5px;">
                    Progress: {self.current_epoch}/{self.total_epochs} epochs ({progress:.1f}%)
                </div>
            """
        return ""