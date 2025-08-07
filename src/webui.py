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
from web.index_tab import create_index_tab
from web.chat_tab import create_chat_tab
from web.venus_factory_advanced_tool_tab import create_advanced_tool_tab
from web.venus_factory_download_tab import create_download_tool_tab
from web.venus_factory_quick_tool_tab import create_quick_tool_tab

from stats_manager import get_stats_api, track_usage_api, stats_manager

def load_constant():
    try:
        return json.load(open("src/constant.json"))
    except Exception as e:
        print(f"Error loading constant.json: {e}")
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
                return "Training not in progress. Configure on the 'Training' tab and click Start.", None, None
        except Exception as e:
            return f"An error occurred in the UI update loop: {str(e)}", None, None
    
    def get_stats():
        return get_stats_api()
    
    def track_usage(module):
        return track_usage_api(module)
    
    def track_agent_usage():
        return track_usage_api('agent_usage')
    
    def track_mutation_quick():
        return track_usage_api('mutation_prediction_quick')
    
    def track_mutation_advanced():
        return track_usage_api('mutation_prediction_advanced')
    
    def track_function_quick():
        return track_usage_api('function_prediction_quick')
    
    def track_function_advanced():
        return track_usage_api('function_prediction_advanced')
    
    def read_css_file(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                print(f"✅ Successfully loaded CSS: {file_path} ({len(content)} characters)")
                return content
        except Exception as e:
            print(f"❌ Warning: Could not read CSS file {file_path}: {e}")
            return ""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    assets_dir = os.path.join(current_dir, "web", "assets")
    custom_css = read_css_file(os.path.join(assets_dir, "custom_ui.css"))
    manual_css = read_css_file(os.path.join(assets_dir, "manual_ui.css"))
    manual_js = read_css_file(os.path.join(assets_dir, "manual_ui.js"))
    css_links = f"""
    <style>
    {custom_css}
    {manual_css}
    {manual_js}
    </style>
    """
    with gr.Blocks(css=css_links) as demo:
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
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor" style="margin-right: 8px;">
                        <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/>
                    </svg>
                    GitHub
                </a>
                <a href="https://huggingface.co/AI4Protein" target="_blank" class="huggingface-button" style="text-decoration: none; display: flex; align-items: center; padding: 10px 15px; background: #f4f4f4; color: #333; border: 2px solid #e0e0e0; border-radius: 8px; font-weight: 600; transition: all 0.3s ease; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                    🤗
                    Hugging Face
                </a>
            </div>
        </div>
        """
        gr.HTML(header_html)
        train_components = None
        with gr.Tabs():
            with gr.TabItem("🏠 Index"):
                try:
                    index_components = create_index_tab(constant)
                except Exception as e:
                    gr.Markdown(f"""**Error creating Index tab:**\n```
{e}\n```""")
            with gr.TabItem("🚀 Model Train and Prediction (For Advanced Users)"):
                with gr.Tabs():
                    with gr.TabItem("Training"):
                        try:
                            train_components = create_train_tab(constant)
                        except Exception as e:
                            gr.Markdown(f"""**Error creating Training tab:**\n```
{e}\n```""")
                    with gr.TabItem("Evaluation"):
                        try:
                            eval_components = create_eval_tab(constant)
                        except Exception as e:
                            gr.Markdown(f"""**Error creating Evaluation tab:**\n```
{e}\n```""")
                    with gr.TabItem("Prediction"):
                        try:
                            predict_components = create_predict_tab(constant)
                        except Exception as e:
                            gr.Markdown(f"""**Error creating Prediction tab:**\n```
{e}\n```""")
            with gr.TabItem("🤖 VenusAgent-0.1 (Beta Version)"):
                try:
                    chat_components = create_chat_tab(constant)
                except Exception as e:
                    gr.Markdown(f"""**Error creating Chat tab:**\n```
{e}\n```""")
            with gr.TabItem("🔧 Quick Tools "):
                try:
                    easy_use_components = create_quick_tool_tab(constant)
                except Exception as e:
                    gr.Markdown(f"""**Error creating Easy-Use tab:**\n```
{e}\n```""")
            with gr.TabItem("⚡ Advanced Tools"):
                try:
                    advanced_tool_components = create_advanced_tool_tab(constant)
                except Exception as e:
                    gr.Markdown(f"""**Error creating Advanced Tools tab:**\n```
{e}\n```""")
            with gr.TabItem("💾 Download "):
                try:
                    download_components = create_download_tool_tab(constant)
                except Exception as e:
                    gr.Markdown(f"""**Error creating Download tab:**\n```
{e}\n```""")
            with gr.TabItem("📖 Manual "):
                try:
                    manual_components = create_manual_tab(constant)
                except Exception as e:
                    gr.Markdown(f"""**Error creating Manual tab:**\n```
{e}\n```""")
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
            print("Warning: Training monitor components not found. The live update feature for training will be disabled.")

        # 注册统计API函数到Gradio - 使用更简单的方式
        stats_output = gr.JSON(label="Statistics", visible=False)
        demo.load(fn=get_stats, inputs=None, outputs=stats_output)
        
        track_input = gr.Textbox(label="Module", visible=False)
        track_output = gr.JSON(label="Track Result", visible=False)
        demo.load(fn=track_usage, inputs=track_input, outputs=track_output)
        
        # 添加统计记录接口
        gr.HTML("""
        <script>
        // 页面加载时自动记录访问
        document.addEventListener('DOMContentLoaded', function() {
            // 记录页面访问
            trackPageVisit();
            
            // 监听Agent的send按钮
            setupAgentTracking();
            
            // 监听功能按钮
            setupFunctionTracking();
        });
        
        function setupAgentTracking() {
            // 监听Agent的send按钮点击
            document.addEventListener('click', function(e) {
                if (e.target && (e.target.textContent.includes('Send') || e.target.textContent.includes('发送'))) {
                    // 检查是否在Agent标签页内
                    const agentTab = document.querySelector('[data-testid="tab"]:has-text("VenusAgent")');
                    if (agentTab && agentTab.classList.contains('selected')) {
                        trackUsage('agent_usage');
                    }
                }
            });
        }
        
        function setupFunctionTracking() {
            // 监听突变预测按钮
            document.addEventListener('click', function(e) {
                const buttonText = e.target.textContent || '';
                
                // 突变预测按钮
                if (buttonText.includes('Mutation') || buttonText.includes('突变') || buttonText.includes('Predict')) {
                    const quickToolsTab = document.querySelector('[data-testid="tab"]:has-text("Quick Tools")');
                    const advancedToolsTab = document.querySelector('[data-testid="tab"]:has-text("Advanced Tools")');
                    
                    if (quickToolsTab && quickToolsTab.classList.contains('selected')) {
                        trackUsage('mutation_prediction_quick');
                    } else if (advancedToolsTab && advancedToolsTab.classList.contains('selected')) {
                        trackUsage('mutation_prediction_advanced');
                    }
                }
                
                // 功能预测按钮
                if (buttonText.includes('Function') || buttonText.includes('功能') || buttonText.includes('Analysis')) {
                    const quickToolsTab = document.querySelector('[data-testid="tab"]:has-text("Quick Tools")');
                    const advancedToolsTab = document.querySelector('[data-testid="tab"]:has-text("Advanced Tools")');
                    
                    if (quickToolsTab && quickToolsTab.classList.contains('selected')) {
                        trackUsage('function_prediction_quick');
                    } else if (advancedToolsTab && advancedToolsTab.classList.contains('selected')) {
                        trackUsage('function_prediction_advanced');
                    }
                }
            });
        }
        
        function trackPageVisit() {
            // 使用localStorage避免重复记录
            const today = new Date().toDateString();
            const lastVisit = localStorage.getItem('lastVisit');
            
            if (lastVisit !== today) {
                localStorage.setItem('lastVisit', today);
                
                // 使用Gradio API调用统计函数
                fetch('/api/predict/', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        fn_index: 1,  // 使用数字索引
                        data: ['total_visits']
                    })
                }).catch(error => {
                    console.log('Failed to track page visit:', error);
                });
            }
        }
        
        function trackUsage(module) {
            // 使用Gradio API调用统计函数
            fetch('/api/predict/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    fn_index: 1,  // 使用数字索引
                    data: [module]
                })
            }).catch(error => {
                console.log('Failed to track usage:', error);
            });
        }
        </script>
        """)
        
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
