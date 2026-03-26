import argparse
import json
import time
import shutil
import threading
import os
import asyncio
from typing import Tuple

import gradio as gr
import datetime
from pathlib import Path
from web.train_tab import create_train_tab
from web.eval_tab import create_eval_tab
from web.predict_tab import create_predict_tab
from web.manual_tab import create_manual_tab
from web.chat_tab import create_chat_tab, AGENT_TAB_CSS
from web.advanced_tool_tab import create_advanced_tool_tab
from web.download_tool_tab import create_download_tool_tab
from web.quick_tool_tab import create_quick_tool_tab
from web.comprehensive_tab import create_comprehensive_tab
from web.utils.monitor import TrainingMonitor
from web.utils.html_ui import load_html_template
from api_server import app as fastapi_app
from mcp_server import start_http_server
from fastapi_mcp import FastApiMCP
import uvicorn


def parse_args():
    """Parse command line arguments for startup mode."""
    parser = argparse.ArgumentParser(
        description="VenusFactory - Unified platform for protein engineering"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["mcp", "fastapi", "server", "all"],
        default="all",
        help="Startup mode: mcp (MCP server only), fastapi (FastAPI only), server (Gradio only), all (all three servers, default)"
    )
    return parser.parse_args()

_fastapi_server_thread = None
_fastapi_server_lock = threading.Lock()
_fastapi_server = None

mcp_server = FastApiMCP(fastapi_app)
mcp_server.mount_http()
mcp_server.mount_sse()  

def start_fastapi_server(host: str = None, port: int = None) -> Tuple[str, int]:
    """Launch the FastAPI application in a background thread if not already running."""
    global _fastapi_server_thread, _fastapi_server

    default_host = os.getenv("FASTAPI_HOST", "0.0.0.0")
    default_port = int(os.getenv("FASTAPI_PORT", "5000"))

    host = host or default_host
    port = port or default_port

    with _fastapi_server_lock:
        if _fastapi_server_thread and _fastapi_server_thread.is_alive():
            return host, port

        config = uvicorn.Config(
            fastapi_app,
            host=host,
            port=port,
            log_level=os.getenv("FASTAPI_LOG_LEVEL", "info"),
            reload=False,
            access_log=True,
            loop="asyncio",
            lifespan="auto",
        )
        config.handle_signals = False

        server = uvicorn.Server(config)
        _fastapi_server = server

        def run_server():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(server.serve())
            except Exception as exc:
                print(f"[FastAPI] Server exited unexpectedly: {exc}")
            finally:
                loop.close()

        thread = threading.Thread(target=run_server, name="FastAPIThread", daemon=True)
        thread.start()
        _fastapi_server_thread = thread

    startup_timeout = float(os.getenv("FASTAPI_STARTUP_TIMEOUT", "5.0"))
    start_time = time.time()
    while getattr(server, "started", False) is False and thread.is_alive():
        if time.time() - start_time > startup_timeout:
            break
        time.sleep(0.1)

    print(f"[FastAPI] Serving on http://{host}:{port}")
    return host, port

def delete_old_files():
    try:
        target_date = datetime.datetime.now() - datetime.timedelta(days=2)
        base_path = Path(os.getenv("TEMP_OUTPUTS_DIR", "temp_outputs"))
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

    # Combine all CSS (Agent tab CSS lives in chat_tab.py)
    css_links = f"""
    <style>
    {custom_css}
    {manual_css}
    {AGENT_TAB_CSS}
    {manual_js}
    </style>
    """
    
    # Gradio 6: css moved from Blocks() to launch()
    with gr.Blocks(title="VenusFactory") as demo:
        # Header with GitHub icon
        header_html = load_html_template("header.html")
        gr.HTML(header_html, padding=True)
        
        # Initialize train_components to None to handle potential creation errors
        train_components = None

        # --- Top-Level Tabs for Main Categories ---
        with gr.Tabs():
            # Agent
            with gr.Tab("🤖 Agent"):
                try:
                    chat_components = create_chat_tab(constant)
                except Exception as e:
                    gr.Markdown(f"**Error creating Chat tab:**\n```\n{e}\n```")
            
            # Report
            with gr.Tab("📋 Report"):
                comprehensive_componsents = create_comprehensive_tab(constant)
                    
            # Model Train and Prediction
            with gr.Tab("🚀 Model Train and Prediction"):
                # Nested (Secondary) Tabs for sub-functions
                with gr.Tabs():
                    with gr.Tab("Training"):
                        try:
                            # This function must return a dictionary with the required components
                            train_components = create_train_tab(constant) 
                        except Exception as e:
                            gr.Markdown(f"**Error creating Training tab:**\n```\n{e}\n```")
                    
                    with gr.Tab("Evaluation"):
                        try:
                            eval_components = create_eval_tab(constant)
                        except Exception as e:
                            gr.Markdown(f"**Error creating Evaluation tab:**\n```\n{e}\n```")

                    with gr.Tab("Prediction"):
                        try:
                            predict_components = create_predict_tab(constant)
                        except Exception as e:
                            gr.Markdown(f"**Error creating Prediction tab:**\n```\n{e}\n```")

            # Quick Tools
            with gr.Tab("🔧 Quick Tools "):
                try:
                    easy_use_components = create_quick_tool_tab(constant)
                except Exception as e:
                    gr.Markdown(f"**Error creating Easy-Use tab:**\n```\n{e}\n```")
                
            # Advanced Tools
            with gr.Tab("⚡ Advanced Tools"):
                try:
                    advanced_tool_components = create_advanced_tool_tab(constant)
                except Exception as e:
                    gr.Markdown(f"**Error creating Advanced Tools tab:**\n```\n{e}\n```")

            # Download
            with gr.Tab("💾 Download "):
                try:
                    download_components = create_download_tool_tab(constant)
                except Exception as e:
                    gr.Markdown(f"**Error creating Download tab:**\n```\n{e}\n```")

            # Manual (no nested tabs needed)
            with gr.Tab("📖 Manual "):
                try:
                    manual_components = create_manual_tab(constant)
                except Exception as e:
                    gr.Markdown(f"**Error creating Manual tab:**\n```\n{e}\n```")
        
        # Footer (visible on all pages)
        footer_html = load_html_template("footer.html")
        gr.HTML(footer_html)
        
        # NOTE: demo.load() disabled for Gradio 6 compatibility.
        # In Gradio 6, components in inactive tabs (e.g. Training tab) may not be in the DOM yet
        # when the load event fires, causing "Component with ID not found". Training monitor
        # still works during actual training via progress callbacks.
            
    return demo, css_links

def run_mcp_server(host: str = None, port: int = None):
    """Run MCP server in foreground (blocking)."""
    host = host or os.getenv("MCP_HTTP_HOST", "0.0.0.0")
    port = port or int(os.getenv("MCP_HTTP_PORT", "8080"))
    print(f"[MCP] Starting server on http://{host}:{port}/mcp")
    from mcp_server import mcp
    app_with_route = mcp.http_app()
    uvicorn.run(app_with_route, host=host, port=port)


def run_fastapi_server(host: str = None, port: int = None):
    """Run FastAPI server in foreground (blocking)."""
    host = host or os.getenv("FASTAPI_HOST", "0.0.0.0")
    port = port or int(os.getenv("FASTAPI_PORT", "5000"))
    print(f"[FastAPI] Starting server on http://{host}:{port}")
    uvicorn.run(
        fastapi_app,
        host=host,
        port=port,
        log_level=os.getenv("FASTAPI_LOG_LEVEL", "info"),
    )


def run_gradio_server():
    """Run Gradio server in foreground (blocking)."""
    demo, css_links = create_ui()
    demo.queue().launch(
        server_port=7860,
        share=True,
        allowed_paths=["img"],
        show_error=True,
        inbrowser=True,
        mcp_server=True,
        css=css_links,
    )


if __name__ == "__main__":
    args = parse_args()

    try:
        if args.mode == "mcp":
            run_mcp_server()

        elif args.mode == "fastapi":
            run_fastapi_server()

        elif args.mode == "server":
            run_gradio_server()

        elif args.mode == "all":
            # Start cleanup thread
            cleanup_thread = threading.Thread(target=run_cleanup_schedule, daemon=True)
            cleanup_thread.start()

            # Start FastAPI in background thread
            fastapi_host, fastapi_port = start_fastapi_server()
            print(f"[FastAPI] Background API available at http://{fastapi_host}:{fastapi_port}")

            # Start MCP in background thread
            mcp_host, mcp_port = start_http_server()
            print(f"[MCP] HTTP server available at http://{mcp_host}:{mcp_port}/mcp")

            # Run Gradio in main thread
            run_gradio_server()

    except Exception as e:
        print(f"Failed to launch: {str(e)}")