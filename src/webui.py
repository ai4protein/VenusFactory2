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
    
    def update_output():
        if monitor.is_training:
            messages = monitor.get_messages()
            plot = monitor.get_plot()
            return messages, plot
        else:
            return "Click Start to begin training!", None
    
    with gr.Blocks() as demo:
        gr.Markdown("# ProFactory")
        
        # Create tabs
        train_components = create_train_tab(monitor, constant)
        inference_components = create_inference_tab(constant)
        
        demo.load(
            fn=update_output,
            inputs=None,
            outputs=[
                train_components["output_text"], 
                train_components["plot_output"]
            ]
        )
        
    return demo

if __name__ == "__main__":
    demo = create_ui()
    demo.launch(server_name="0.0.0.0", share=True)