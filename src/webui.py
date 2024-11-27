import json
import time
import gradio as gr
from web.monitor import TrainingMonitor
from web.train_tab import create_train_tab
from web.eval_tab import create_inference_tab

def load_constant():
    """Load constant values from config files"""
    return json.load(open("src/constant.json"))

def create_ui():
    monitor = TrainingMonitor()
    constant = load_constant()
    
    with gr.Blocks() as demo:
        gr.Markdown("# ProFactory")
        
        # Create tabs
        train_components = create_train_tab(monitor, constant)
        inference_components = create_inference_tab(constant)
        

    return demo

if __name__ == "__main__":
    demo = create_ui()
    demo.launch(server_name="0.0.0.0", share=True)