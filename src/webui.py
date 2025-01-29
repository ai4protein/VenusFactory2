import json
import time
import gradio as gr
from web.utils.monitor import TrainingMonitor
from web.train_tab import create_train_tab
from web.eval_tab import create_inference_tab
from web.download_tab import create_download_tab

def load_constant():
    """Load constant values from config files"""
    try:
        return json.load(open("src/constant.json"))
    except Exception as e:
        return {"error": f"Failed to load constant.json: {str(e)}"}

def create_ui():
    monitor = TrainingMonitor()
    constant = load_constant()
    
    def update_output():
        try:
            if monitor.is_training:
                messages = monitor.get_messages()
                plot = monitor.get_plot()
                return messages, plot
            else:
                if monitor.error_message:
                    return f"Training stopped with error:\n{monitor.error_message}", None
                return "Click Start to begin training!", None
        except Exception as e:
            return f"Error in UI update: {str(e)}", None
    
    with gr.Blocks() as demo:
        gr.Markdown("# VenusFactory")
        
        # Create tabs
        with gr.Tabs():
            try:
                train_components = create_train_tab(constant)
                inference_components = create_inference_tab(constant)
                download_components = create_download_tab(constant)
            except Exception as e:
                gr.Markdown(f"Error creating UI components: {str(e)}")
                return
        
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
    try:
        demo = create_ui()
        demo.launch(server_name="0.0.0.0", share=True)
    except Exception as e:
        print(f"Failed to launch UI: {str(e)}")