from typing import Dict, Any
import os
import json

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