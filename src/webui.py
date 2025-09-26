import json
import time
import threading
import os
import gradio as gr
from web.utils.monitor import TrainingMonitor
from web.train_tab import create_train_tab
from web.eval_tab import create_eval_tab
from web.download_tab import create_download_tab
from web.predict_tab import create_predict_tab
from web.manual_tab import create_manual_tab
from web.chat_tab import create_chat_tab
from web.venus_factory_advanced_tool_tab import create_advanced_tool_tab
from web.venus_factory_download_tab import create_download_tool_tab
from web.venus_factory_quick_tool_tab import create_quick_tool_tab
from web.venus_factory_comprehensive_tab import create_comprehensive_tab

def delete_old_files():
    try:
        target_date = datetime.now() - timedelta(days=2)
        base_path = Path("temp_outputs")
        dir_to_delete = base_path / target_date.strftime('%Y') / target_date.strftime('%m') / target_date.strftime('%d')
        if dir_to_delete.is_dir():
            shutil.rmtree(dir_to_delete)

    except Exception as e:
        print(f"Clean file error: {e}")

def run_cleanup_schedule():
    while True:
        delete_old_files()
        time.sleep(60 * 60)

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
    
    # Read CSS files and embed them directly
    def read_css_file(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                print(f"✅ Successfully loaded CSS: {file_path} ({len(content)} characters)")
                return content
        except Exception as e:
            print(f"❌ Warning: Could not read CSS file {file_path}: {e}")
            return ""
    
    # Get the directory of this file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    assets_dir = os.path.join(current_dir, "web", "assets")
    
    # Read CSS files
    custom_css = read_css_file(os.path.join(assets_dir,"css", "custom_ui.css"))
    manual_css = read_css_file(os.path.join(assets_dir, "css", "manual_ui.css"))
    manual_js = read_css_file(os.path.join(assets_dir, "js", "manual_ui.js"))
    
    # Combine all CSS
    css_links = f"""
    <style>
    {custom_css}
    {manual_css}
    {manual_js}
    </style>
    """
    
    with gr.Blocks(css=css_links, title="VenusFactory") as demo:
        # Header with GitHub icon
        header_html = """
        <div class="header-container" style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 20px;">
            <div style="display: flex; align-items: center; gap: 15px;">
                <h1 style="margin: 0; font-size: 2.5em; font-weight: 700; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;">
                    VenusFactory
                </h1>
                <span style="font-size: 1.6em; color: #666; font-weight: 400;">The most powerful Open-source AI protein engineering platform</span>
            </div>
            <div style="display: flex; gap: 10px;">
                <a href="https://github.com/ai4protein/VenusFactory" target="_blank" class="github-button" style="text-decoration: none; display: flex; align-items: center; padding: 10px 15px; background: #f4f4f4; color: #333; border: 2px solid #e0e0e0; border-radius: 8px; font-weight: 600; transition: all 0.3s ease; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="#333" style="margin-right: 8px;">
                        <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/>
                    </svg>
                    GitHub
                </a>
                <a href="https://huggingface.co/AI4Protein" target="_blank" class="huggingface-button" style="text-decoration: none; display: flex; align-items: center; padding: 10px 15px; background: #f4f4f4; color: #333; border: 2px solid #e0e0e0; border-radius: 8px; font-weight: 600; transition: all 0.3s ease; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                    🤗 Hugging Face
                </a>
            </div>
        </div>
        """
        gr.HTML(header_html)
        
        # Initialize train_components to None to handle potential creation errors
        train_components = None

        # --- Top-Level Tabs for Main Categories ---
        with gr.Tabs():
            with gr.TabItem("📋 VenusScope"):
                comprehensive_componsents = create_comprehensive_tab(constant)
            
            with gr.TabItem("🤖 VenusAgent"):
                try:
                    chat_components = create_chat_tab(constant)
                except Exception as e:
                    gr.Markdown(f"**Error creating Chat tab:**\n```\n{e}\n```")
                    
            # Model Train and Prediction
            with gr.TabItem("🚀 Model Train and Prediction"):
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

            

            # Quick Tools
            with gr.TabItem("🔧 Quick Tools "):
                try:
                    easy_use_components = create_quick_tool_tab(constant)
                except Exception as e:
                    gr.Markdown(f"**Error creating Easy-Use tab:**\n```\n{e}\n```")
                
            # Advanced Tools
            with gr.TabItem("⚡ Advanced Tools"):
                try:
                    advanced_tool_components = create_advanced_tool_tab(constant)
                except Exception as e:
                    gr.Markdown(f"**Error creating Advanced Tools tab:**\n```\n{e}\n```")

            # Download
            with gr.TabItem("💾 Download "):
                try:
                    download_components = create_download_tool_tab(constant)
                except Exception as e:
                    gr.Markdown(f"**Error creating Download tab:**\n```\n{e}\n```")

            # Manual (no nested tabs needed)
            with gr.TabItem("📖 Manual "):
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
        demo.queue().launch(
            server_port=7860, 
            share=True, 
            allowed_paths=["img"],
            show_error=True,
            inbrowser=True,
        )

    except Exception as e:
        print(f"Failed to launch UI: {str(e)}")
