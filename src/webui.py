import json
import time
import gradio as gr
from web.utils.monitor import TrainingMonitor
from web.train_tab import create_train_tab
from web.eval_tab import create_eval_tab
from web.download_tab import create_download_tab
from web.predict_tab import create_predict_tab
from web.manual_tab import create_manual_tab

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
                loss_plot = monitor.get_loss_plot()
                metrics_plot = monitor.get_metrics_plot()
                return messages, loss_plot, metrics_plot
            else:
                if monitor.error_message:
                    return f"Training stopped with error:\n{monitor.error_message}", None, None
                return "Click Start to begin training!", None, None
        except Exception as e:
            return f"Error in UI update: {str(e)}", None, None
    
    with gr.Blocks() as demo:
        gr.Markdown("# VenusFactory")
        
        # Create tabs
        with gr.Tabs():
            try:
                train_components = {"output_text": None, "loss_plot": None, "metrics_plot": None}
                train_tab = create_train_tab(constant)
                if train_components["output_text"] is not None and train_components["loss_plot"] is not None and train_components["metrics_plot"] is not None:
                    train_components["output_text"] = train_tab["output_text"]
                    train_components["loss_plot"] = train_tab["loss_plot"]
                    train_components["metrics_plot"] = train_tab["metrics_plot"]
                eval_components = create_eval_tab(constant)
                predict_components = create_predict_tab(constant)
                download_components = create_download_tab(constant)
                manual_components = create_manual_tab(constant)
            except Exception as e:
                gr.Markdown(f"Error creating UI components: {str(e)}")
                train_components = {"output_text": None, "loss_plot": None, "metrics_plot": None}
        
        if train_components["output_text"] is not None and train_components["loss_plot"] is not None and train_components["metrics_plot"] is not None:
            demo.load(
                fn=update_output,
                inputs=None,
                outputs=[
                    train_components["output_text"], 
                    train_components["loss_plot"],
                    train_components["metrics_plot"]
                ]
            )
        
    return demo

if __name__ == "__main__":
    try:
        demo = create_ui()
        demo.launch(server_name="0.0.0.0", share=True, allowed_paths=["img"])
    except Exception as e:
        print(f"Failed to launch UI: {str(e)}")