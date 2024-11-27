import os
import json
import sys
import gradio as gr

def preview_command(args, constant):
    # 构建命令字符串
    cmd = [sys.executable, "src/train.py"]  # 开始用 python 路径
    
    for k, v in args.items():
        if v is not None:
            if k == "plm_model":
                v = constant["plm_models"][v]
            elif k == "dataset_config":
                v = constant["dataset_configs"][v]
            if isinstance(v, bool):
                if v:  # 对于布尔值，只在True时添加参数名
                    cmd.append(f"--{k}")
            else:
                cmd.append(f"--{k}")
                cmd.append(str(v))
    
    # 将命令列表转换为字符串，每个部分用空格连接
    return " ".join(cmd)


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