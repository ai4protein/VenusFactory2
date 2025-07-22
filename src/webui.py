import json
import time
import gradio as gr
from web.utils.monitor import TrainingMonitor
from web.train_tab import create_train_tab
from web.eval_tab import create_eval_tab
from web.download_tab import create_download_tab
from web.predict_tab import create_predict_tab
from web.manual_tab import create_manual_tab
from web.zero_shot_tab import create_zero_shot_tab
from web.function_predict_tab import create_protein_function_tab

def load_constant():
    """Load constant values from config files"""
    try:
        return json.load(open("src/constant.json"))
    except Exception as e:
        print(f"Error loading constant.json: {e}")
        return {"error": f"Failed to load constant.json: {str(e)}"}

def create_ui():
    """Creates the main Gradio UI with a nested tab layout."""
    monitor = TrainingMonitor()
    constant = load_constant()
    
    def update_output():
        """Callback function to update the training monitor UI components."""
        try:
            if monitor.is_training:
                messages = monitor.get_messages()
                loss_plot = monitor.get_loss_plot()
                metrics_plot = monitor.get_metrics_plot()
                return messages, loss_plot, metrics_plot
            else:
                if monitor.error_message:
                    return f"Training stopped with error:\n{monitor.error_message}", None, None
                return "Training not in progress. Configure on the 'Training' tab and click Start.", None, None
        except Exception as e:
            return f"An error occurred in the UI update loop: {str(e)}", None, None
    
    with gr.Blocks(title="VenusFactory") as demo:
        gr.Markdown("# VenusFactory")
        
        # Initialize train_components to None to handle potential creation errors
        train_components = None

        # --- Top-Level Tabs for Main Categories ---
        with gr.Tabs():
            
            # Group 1: Model Train and Prediction
            with gr.TabItem("🚀 Model Train and Prediction (For Advanced Users)"):
                # Nested (Secondary) Tabs for sub-functions
                with gr.Tabs():
                    with gr.TabItem("Training"):
                        try:
                            # This function must return a dictionary with the required components
                            train_components = create_train_tab(constant) 
                        except Exception as e:
                            gr.Markdown(f"**Error creating Training tab:**\n```\n{e}\n```")
                    
                    with gr.TabItem("Evaluation"):
                        try:
                            eval_components = create_eval_tab(constant)
                        except Exception as e:
                            gr.Markdown(f"**Error creating Evaluation tab:**\n```\n{e}\n```")

                    with gr.TabItem("Prediction"):
                        try:
                            predict_components = create_predict_tab(constant)
                        except Exception as e:
                            gr.Markdown(f"**Error creating Prediction tab:**\n```\n{e}\n```")

            # Group 2: Quick Start
            with gr.TabItem("⚡ Quick Start (For Biologists to predict)"):
                # Nested (Secondary) Tabs for sub-functions
                with gr.Tabs():
                    with gr.TabItem("Zero-Shot Mutation Prediction (Get better mutants before wet-lab experiments)"):
                        try:
                            zero_shot_components = create_zero_shot_tab(constant)
                        except Exception as e:
                            gr.Markdown(f"**Error creating Zero-Shot Prediction tab:**\n```\n{e}\n```")

                    with gr.TabItem("Protein Function Prediction (Get the function of a protein before wet-lab experiments)"):
                        try:
                            protein_function_components = create_protein_function_tab(constant)
                        except Exception as e:
                            gr.Markdown(f"**Error creating Protein Function Prediction tab:**\n```\n{e}\n```")

                    with gr.TabItem("Download (Download PDB/FASTA/InterPro... Files)"):
                        try:
                            download_components = create_download_tab(constant)
                        except Exception as e:
                            gr.Markdown(f"**Error creating Download tab:**\n```\n{e}\n```")

            # Group 3: Manual (no nested tabs needed)
            with gr.TabItem("📖 Manual (More details about the platform)"):
                try:
                    manual_components = create_manual_tab(constant)
                except Exception as e:
                    gr.Markdown(f"**Error creating Manual tab:**\n```\n{e}\n```")
        
        # Check if the training components were created successfully before setting up the monitor loop
        if train_components and all(k in train_components for k in ["output_text", "loss_plot", "metrics_plot"]):
            demo.load(
                fn=update_output,
                inputs=None,
                outputs=[
                    train_components["output_text"], 
                    train_components["loss_plot"],
                    train_components["metrics_plot"]
                ],

            )
        else:
            # This message will be printed to the console where the script is running
            print("Warning: Training monitor components not found. The live update feature for training will be disabled.")
            
    return demo

if __name__ == "__main__":
    try:
        demo = create_ui()
        demo.queue().launch(server_name="0.0.0.0", server_port=7860, share=True, allowed_paths=["img"], mcp_server=True)
    except Exception as e:
        print(f"Failed to launch UI: {str(e)}")