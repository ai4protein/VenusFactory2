import os
import json
import sys
import gradio as gr

def preview_command(args_dict, cmd_list):
    # build final command
    final_cmd = ""
    for i, part in enumerate(cmd_list):
        if i > 0:  # Skip first element (python path)
            if part.startswith("--"):
                final_cmd += "\n\t" + part
            else:
                # add '\' at end of part if not the last one
                final_cmd += " " + part + (" \\" if i != len(cmd_list) - 1 else "")
        else:
            final_cmd += part
    return gr.update(value=final_cmd, visible=True)

def save_arguments(save_path, args_dict):
    if not save_path:
        return gr.update(value="Please enter a save path!", visible=True)
    
    try:
        # ensure directory exists
        save_path = os.path.join('configs', save_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # save arguments to json file
        with open(save_path, 'w') as f:
            json.dump(args_dict, f, indent=4)
        return gr.update(value=f"Arguments saved to {save_path}", visible=True)
    except Exception as e:
        return gr.update(value=f"Error saving arguments: {str(e)}", visible=True)

def load_arguments(load_path):
    try:
        with open(load_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        return None

def build_command_list(args_dict):
    cmd = [sys.executable, "src/train.py"]
    for k, v in args_dict.items():
        if v is True:
            cmd.append(f"--{k}")
        elif v is not False and v is not None:
            cmd.append(f"--{k}")
            cmd.append(str(v))
    return cmd