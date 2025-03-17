from typing import Dict, Any
import os
import json
import sys

def build_command_list(args: Dict[str, Any]) -> list:
    """Build command list for training script."""
    cmd = ["python", "src/train.py"]
    
    for key, value in args.items():
        if value is None or value == "":
            continue
            
        if isinstance(value, bool):
            if value:
                cmd.append(f"--{key}")
        elif key == "lora_target_modules":
            if value:
                cmd.append(f"--{key}")
                cmd.extend(value)
        else:
            cmd.extend([f"--{key}", str(value)])
    
    return cmd

def preview_command(args: Dict[str, Any]) -> str:
    """Generate preview of training command."""
    cmd = build_command_list(args)
    return " ".join(cmd)

def save_arguments(args: Dict[str, Any], output_dir: str):
    """Save training arguments to file."""
    os.makedirs(output_dir, exist_ok=True)
    args_file = os.path.join(output_dir, "training_args.json")
    
    with open(args_file, 'w') as f:
        json.dump(args, f, indent=2)

def build_eval_command_list(args: Dict[str, Any]) -> list:
    """构建评估脚本的命令行列表"""
    cmd = ["python", "src/eval.py"]
    
    for key, value in args.items():
        if value is None or value == "":
            continue
            
        if isinstance(value, bool):
            if value:
                cmd.append(f"--{key}")
        else:
            cmd.extend([f"--{key}", str(value)])
    
    return cmd

def preview_eval_command(args: Dict[str, Any]) -> str:
    """生成评估命令的预览"""
    cmd = build_eval_command_list(args)
    return " ".join(cmd)

def build_predict_command_list(args: Dict[str, Any], is_batch: bool = False) -> list:
    """构建预测脚本的命令行列表"""
    # 根据是否为批量预测选择不同的脚本
    script = "src/predict_batch.py" if is_batch else "src/predict.py"
    cmd = ["python", script]
    
    for key, value in args.items():
        if value is None or value == "":
            continue
            
        if isinstance(value, bool):
            if value:
                cmd.append(f"--{key}")
        elif isinstance(value, list):
            if value:
                cmd.append(f"--{key}")
                cmd.extend([str(v) for v in value])
        else:
            cmd.extend([f"--{key}", str(value)])
    
    return cmd

def preview_predict_command(args: Dict[str, Any], is_batch: bool = False) -> str:
    """生成预测命令的预览"""
    cmd = build_predict_command_list(args, is_batch)
    return " ".join(cmd) 