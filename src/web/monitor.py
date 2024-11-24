import sys
import queue
import subprocess
from threading import Thread

class TrainingMonitor:
    def __init__(self):
        self.message_queue = queue.Queue()
        self.is_training = False
        self.current_epoch = 0
        self.total_epochs = 0
        self.current_step = 0
        self.total_steps = 0

    def start_training(self, args_dict):
        self.is_training = True
        self.current_epoch = 0
        self.total_epochs = int(args_dict.get("train_epoch", 0))
        thread = Thread(target=self._run_training, args=(args_dict,))
        thread.start()
        return thread

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

            for line in process.stdout:
                self.message_queue.put(line.strip())
                if "EPOCH" in line and "TRAIN loss:" in line:
                    try:
                        self.current_epoch = int(line.split("EPOCH")[1].split()[0])
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
        messages = []
        while not self.message_queue.empty():
            messages.append(self.message_queue.get())
        return "\n".join(messages)

    def get_progress(self):
        if not self.is_training:
            return ""
        progress = f"Training Progress: Epoch {self.current_epoch}/{self.total_epochs}"
        percentage = (self.current_epoch / self.total_epochs) * 100 if self.total_epochs > 0 else 0
        progress_bar = f"""
        <div style="width:100%; height:20px; background-color:#f0f0f0; border-radius:5px;">
            <div style="width:{percentage}%; height:100%; background-color:#4CAF50; border-radius:5px;">
            </div>
        </div>
        <p style="text-align:center;">{progress}</p>
        """
        return progress_bar