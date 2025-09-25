import gradio as gr
import os
import re
import markdown
from typing import Dict, Any

def create_manual_tab(constant: Dict[str, Any]) -> Dict[str, Any]:
    # Add CSS styles from external file
    css_path = os.path.join(os.path.dirname(__file__), "assets", "css", "manual_ui.css")
    custom_css = open(css_path, "r").read()
    
    js_path = os.path.join(os.path.dirname(__file__), "assets", "js", "manual_ui.js")
    custom_js = open(js_path, "r").read()
    
    gr.HTML(f"<style>{custom_css}</style><script>{custom_js}</script>", visible=False)
    
    # convert markdown to html
    def markdown_to_html(markdown_content, base_path="src/web/manual"):
        """Convert Markdown content to HTML, and embed images as base64 encoded"""
        # Process image paths, use base64 encoding to directly embed images
        def embed_image(match):
            alt_text = match.group(1)
            img_path = match.group(2)
            
            # Check if the path is an external URL
            if img_path.startswith(('http://', 'https://')):
                return f'<img src="{img_path}" alt="{alt_text}" />'
            
            # Process local image paths
            try:
                # Remove the leading / to get the correct path
                if img_path.startswith('/'):
                    img_path = img_path[1:]
                
                # Get the absolute path
                current_dir = os.path.dirname(os.path.abspath(__file__))
                project_root = os.path.dirname(os.path.dirname(current_dir))
                abs_img_path = os.path.join(project_root, img_path)
                
                # Read the image and convert it to base64
                import base64
                from pathlib import Path
                
                image_path = Path(abs_img_path)
                if image_path.exists():
                    image_type = image_path.suffix.lstrip('.').lower()
                    if image_type == 'jpg':
                        image_type = 'jpeg'
                        
                    with open(image_path, "rb") as img_file:
                        encoded_string = base64.b64encode(img_file.read()).decode('utf-8')
                        
                    return f'<img src="data:image/{image_type};base64,{encoded_string}" alt="{alt_text}" style="max-width:100%; height:auto;" />'
                else:
                    print(f"Image file does not exist: {abs_img_path}")
                    return f'<span style="color:red;">[Image does not exist: {img_path}]</span>'
                
            except Exception as e:
                print(f"Error processing image: {e}, path: {img_path}")
                return f'<span style="color:red;">[Image loading error: {img_path}]</span>'
        
        # Use regular expression to process all image tags
        pattern = r'!\[(.*?)\]\((.*?)\)'
        processed_content = re.sub(pattern, embed_image, markdown_content)
        
        # Use Python's markdown library to convert
        html = markdown.markdown(
            processed_content, 
            extensions=[
                'tables', 
                'fenced_code', 
                'codehilite', 
                'nl2br', 
                'extra',
                'mdx_truly_sane_lists'
            ],
            extension_configs={
                'mdx_truly_sane_lists': {
                    'nested_indent': 2,
                    'truly_sane': True
                }
            }
        )
        
        return html

    # Generate HTML navigation bar and process content from Markdown content
    def generate_toc_and_content(markdown_content):
        """Generate HTML navigation bar and process content from Markdown content"""
        # Extract all headers
        headers = re.findall(r'^(#{1,3})\s+(.+)$', markdown_content, re.MULTILINE)
        
        if not headers:
            return "<div class='manual-nav'><p>目录加载中...</p></div>", markdown_content
        
        toc_html = "<div class='manual-nav'><ul>"
        
        # Create navigation items for each header
        for i, (level, title) in enumerate(headers):
            level_num = len(level)
            header_id = f"header-{i}"
            
            # Add class based on header level
            css_class = ""
            if level_num == 2:
                css_class = "nav-h2"
            elif level_num == 3:
                css_class = "nav-h3"
            
            toc_html += f"<li><a href='#{header_id}' class='{css_class}'>{title}</a></li>"
        
        toc_html += "</ul></div>"
        
        # Add ID to headers in Markdown content
        processed_content = markdown_content
        for i, (level, title) in enumerate(headers):
            header_id = f"header-{i}"
            header_pattern = f"{level} {title}"
            header_replacement = f"{level} <span id='{header_id}'></span>{title}"
            processed_content = processed_content.replace(header_pattern, header_replacement, 1)
        
        # Convert processed Markdown to HTML
        html_content = markdown_to_html(processed_content)
        
        return toc_html, html_content

    with gr.Row():
        language = gr.Dropdown(choices=['English', 'Chinese'], value='English', label='Language', interactive=True)
    

    with gr.Tab("Index"): 
        gr.HTML(f'''
                <script>
                // Track page visits asynchronously without blocking page load
                (function() {{
                    // Use setTimeout to ensure page is fully loaded before tracking
                    setTimeout(() => {{
                        fetch('/api/stats/track', {{
                            method: 'POST',
                            headers: {{'Content-Type': 'application/json'}},
                            body: JSON.stringify({{module: 'total_visits'}})
                        }}).catch(error => {{
                            console.log('Stats API call failed, using mock data');
                        }});
                    }}, 1000);
                }})();
                </script>
                <style>
                @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700&display=swap');
                
                /* CSS Variables for Theme Switching */
                :root {{
                    /* Light theme variables */
                    --bg-primary: #f4f7fa;
                    --bg-secondary: #fff;
                    --bg-card: #f4f7fa;
                    --bg-stats: linear-gradient(135deg, #f6fbff 0%, #e3f3fb 100%);
                    --bg-stats-item: rgba(255, 255, 255, 0.8);
                    --bg-stats-item-hover: rgba(255, 255, 255, 0.9);
                    --text-primary: #222;
                    --text-secondary: #1e293b;
                    --text-muted: #444;
                    --accent-color: #2563eb;
                    --border-color: #e2e8f0;
                    --shadow-color: rgba(30,41,59,0.07);
                    --card-shadow: rgba(30,41,59,0.04);
                    --accent-shadow: rgba(37, 99, 235, 0.07);
                    --accent-border: rgba(37, 99, 235, 0.15);
                    --accent-shadow-light: rgba(37, 99, 235, 0.08);
                    --accent-shadow-hover: rgba(37, 99, 235, 0.15);
                }}
                
                /* Dark theme variables */
                @media (prefers-color-scheme: dark) {{
                    :root {{
                        --bg-primary: #0f172a;
                        --bg-secondary: #1e293b;
                        --bg-card: #334155;
                        --bg-stats: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
                        --bg-stats-item: rgba(51, 65, 85, 0.8);
                        --bg-stats-item-hover: rgba(51, 65, 85, 0.9);
                        --text-primary: #f1f5f9;
                        --text-secondary: #e2e8f0;
                        --text-muted: #94a3b8;
                        --accent-color: #60a5fa;
                        --border-color: #475569;
                        --shadow-color: rgba(0, 0, 0, 0.3);
                        --card-shadow: rgba(0, 0, 0, 0.2);
                        --accent-shadow: rgba(96, 165, 250, 0.15);
                        --accent-border: rgba(96, 165, 250, 0.3);
                        --accent-shadow-light: rgba(96, 165, 250, 0.2);
                        --accent-shadow-hover: rgba(96, 165, 250, 0.3);
                    }}
                }}
                
                body, .gradio-container {{
                    font-family: 'Inter', 'Roboto', Arial, sans-serif !important;
                    background: var(--bg-primary);
                    color: var(--text-primary);
                    transition: background-color 0.3s ease, color 0.3s ease;
                }}
                .main-content {{
                    margin-top: 80px;
                    max-width: 1400px;
                    margin-left: auto;
                    margin-right: auto;
                    background: var(--bg-secondary);
                    border-radius: 16px;
                    box-shadow: 0 4px 24px var(--shadow-color);
                    padding: 40px 36px 32px 36px;
                    transition: background-color 0.3s ease, box-shadow 0.3s ease;
                }}
                h1, h2, h3 {{
                    color: var(--accent-color);
                    font-weight: 700;
                    margin-bottom: 0.5em;
                    transition: color 0.3s ease;
                }}
                p, li, ul {{
                    font-size: 1.08em;
                    line-height: 1.7;
                    color: var(--text-primary);
                    transition: color 0.3s ease;
                }}
                .card {{
                    background: var(--bg-card);
                    border-radius: 10px;
                    box-shadow: 0 2px 8px var(--card-shadow);
                    padding: 18px 22px;
                    margin-bottom: 18px;
                    transition: background-color 0.3s ease, box-shadow 0.3s ease;
                }}
                .stats-container {{
                    background: var(--bg-stats);
                    border-radius: 16px;
                    padding: 32px;
                    margin-top: 40px;
                    color: var(--text-secondary);
                    box-shadow: 0 8px 32px var(--accent-shadow);
                    transition: background 0.3s ease, color 0.3s ease, box-shadow 0.3s ease;
                }}
                .stats-grid {{
                    display: grid;
                    grid-template-columns: repeat(4, 1fr);
                    gap: 32px;
                    margin-top: 24px;
                    max-width: 1200px;
                    margin-left: auto;
                    margin-right: auto;
                }}
                .stat-item {{
                    background: var(--bg-stats-item);
                    border-radius: 12px;
                    padding: 16px 10px;
                    text-align: center;
                    backdrop-filter: blur(10px);
                    border: 1px solid var(--accent-border);
                    transition: all 0.3s ease;
                    box-shadow: 0 4px 16px var(--accent-shadow-light);
                }}
                .stat-item:hover {{
                    transform: translateY(-4px);
                    box-shadow: 0 12px 40px var(--accent-shadow-hover);
                    background: var(--bg-stats-item-hover);
                }}
                .stat-number {{
                    font-size: 1.5em;
                    font-weight: 900;
                    margin-bottom: 4px;
                    text-shadow: 0 1px 2px var(--accent-shadow);
                    color: var(--text-primary);
                    transition: color 0.3s ease;
                }}
                .stat-label {{
                    font-size: 0.95em;
                    opacity: 0.9;
                    font-weight: 500;
                    color: var(--text-secondary);
                    transition: color 0.3s ease;
                }}
                .stat-icon {{
                    font-size: 1.3em;
                    margin-bottom: 6px;
                    display: block;
                }}
                
                /* Enhanced dark mode styles */
                @media (prefers-color-scheme: dark) {{
                    /* Link colors */
                    a {{
                        color: var(--accent-color);
                    }}
                    
                    /* Code block styling */
                    pre {{
                        background: var(--bg-card);
                        color: var(--text-primary);
                        border: 1px solid var(--border-color);
                    }}
                    
                    /* Horizontal rule */
                    hr {{
                        border-color: var(--border-color);
                    }}
                    
                    /* Table styling */
                    table {{
                        background: var(--bg-secondary);
                        color: var(--text-primary);
                    }}
                    
                    th, td {{
                        border-color: var(--border-color);
                    }}
                    
                    /* Button styling */
                    button, .btn {{
                        background: var(--accent-color);
                        color: var(--bg-secondary);
                        border: 1px solid var(--accent-color);
                    }}
                    
                    button:hover, .btn:hover {{
                        background: var(--accent-color);
                        opacity: 0.9;
                    }}
                    
                    /* Enhanced dark mode for specific elements */
                    .stat-item {{
                        background: rgba(51, 65, 85, 0.9);
                        border-color: var(--accent-border);
                    }}
                    
                    .stat-item:hover {{
                        background: rgba(51, 65, 85, 1);
                        box-shadow: 0 12px 40px var(--accent-shadow-hover);
                    }}
                    
                    /* Dark mode for images */
                    img {{
                        filter: brightness(0.9) contrast(1.1);
                    }}
                    
                    /* Dark mode for text elements */
                    h1, h2, h3 {{
                        color: var(--accent-color);
                    }}
                    
                    /* Dark mode for cards */
                    .stats-container {{
                        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
                        color: var(--text-secondary);
                    }}
                }}
                </style>
                <div class="main-content">
            <!-- Top section: VenusFactory introduction -->
            <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 0.5em;">
                <img id="venusfactory-logo" src="https://blog-img-1259433191.cos.ap-shanghai.myqcloud.com/venus/img/venusfactory_logo.png" alt="Venus Head" style="height: 150px; margin-left: 10px;" />
                <img id="venus-logo" src="https://blog-img-1259433191.cos.ap-shanghai.myqcloud.com/venus/img/venus_logo.png" alt="Venus Logo" style="height: 100px; margin-right: 10px; margin-top: 20px;" />
            </div>
            <style>
                @media (prefers-color-scheme: dark) {{
                    #venusfactory-logo {{
                        content: url('https://blog-img-1259433191.cos.ap-shanghai.myqcloud.com/venus/img/venusfactory_logo_darkmode.png');
                    }}
                    #venus-logo {{
                        content: url('https://blog-img-1259433191.cos.ap-shanghai.myqcloud.com/venus/img/venus_logo_darkmode.png');
                    }}
                }}
            </style>
            <div style="text-align: center; margin-top: -80px; margin-bottom: 40px;">
                <h1 style="font-size:3.5em; font-weight:900;">Welcome to <span style='font-weight:900;'>VenusFactory</span>!</h1>
            </div>
            <div style="max-width: 1400px; margin: 0 auto; font-size: 1.2em; text-align: left;">
                <p style="font-size:1.2em; margin-bottom: 0.7em;"><b>VenusFactory</b> is a unified open-source platform for protein engineering, designed to simplify data acquisition, model fine-tuning, and functional analysis for both biologists and AI researchers.<br>
                The Web UI features four core modules:</p>
                <ul style="font-size:1.0em;">
                    <li>🤖 <b>VenusAgent-0.1</b> is an integrated AI assistant that answers questions related to the platform and protein AI.</li>
                    <li>🛠️ <b>Quick Tools</b> offers one-click protein analysis tools designed as a convenient method, making common tasks easy and accessible.</li>
                    <li>⚡ <b>Advanced Tools</b> enables zero-shot prediction, function analysis, and advanced data options for experienced users.</li>
                    <li>💾 <b>Download</b> allows you to get various protein data like AlphaFold2 Structures, RCSB PDB and InterPro.</li>
                </ul>
            </div>
            <hr style="margin: 40px 0; border: 1px solid var(--border-color);">
            <!-- Middle section: How to Use VenusFactory -->
            <div style="text-align: left; max-width: 1400px; margin: 0 auto;">
                <h1 style="font-size:2.2em; font-weight:900; color:var(--text-primary); margin-bottom: 0.7em;">
                    How to Use VenusFactory
                </h1>
                <div style="font-size:1.2em;">
                    <p style="font-size:1.2em;">Depending on your needs, VenusFactory can provide different services.</p>
                    <p style="font-size:1.2em;">If you want a quick answer about protein mutations, use VenusAgent-0.1. Upload your file, and the AI Assistant will give you a helpful reply.</p>
                    <p style="font-size:1.2em;">If you want to know possible mutation methods or protein functions, go to Quick Tools, choose the task you need, and you will get the result in a few minutes.</p>
                    <p style="font-size:1.2em;">If you have some knowledge about different protein models, you can use the Advanced Tools tab. All major models are available to meet your needs.</p>
                    <p style="font-size:1.2em;">If you want to get some protein data files, click the download tab, input the PDB ID or UniProt ID, and download it for further research.</p>
                    
                    <p style="font-size:1.2em; font-weight:bold; margin-top:2em;">Module Demonstrations:</p>
                    
                    <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; margin-top: 1em;">
                        <!-- VenusAgent-0.1 GIF -->
                        <div style="text-align: center;">
                            <h3 style="color: var(--accent-color); margin-bottom: 10px;">🤖 VenusAgent-0.1</h3>
                            <img src="https://blog-img-1259433191.cos.ap-shanghai.myqcloud.com/venus/gif/agent.gif" alt="VenusAgent-0.1 Demo" style="max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.1);" />
                        </div>
                        
                        <!-- Quick Tools GIF -->
                        <div style="text-align: center;">
                            <h3 style="color: var(--accent-color); margin-bottom: 10px;">🛠️ Quick Tools</h3>
                            <img src="https://blog-img-1259433191.cos.ap-shanghai.myqcloud.com/venus/gif/quick_tool.gif" alt="Quick Tools Demo" style="max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.1);" />
                        </div>
                        
                        <!-- Advanced Tools GIF -->
                        <div style="text-align: center;">
                            <h3 style="color: var(--accent-color); margin-bottom: 10px;">⚡ Advanced Tools</h3>
                            <img src="https://blog-img-1259433191.cos.ap-shanghai.myqcloud.com/venus/gif/advanced_tool.gif" alt="Advanced Tools Demo" style="max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.1);" />
                        </div>
                        
                        <!-- Download GIF -->
                        <div style="text-align: center;">
                            <h3 style="color: var(--accent-color); margin-bottom: 10px;">💾 Download</h3>
                            <img src="https://blog-img-1259433191.cos.ap-shanghai.myqcloud.com/venus/gif/download.gif" alt="Download Demo" style="max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.1);" />
                        </div>
                    </div>
                </div>
            </div>
            <hr style="margin: 40px 0; border: 1px solid var(--border-color);">

                <!-- Survey and Partner Institutions Display Area -->
                <div style="background: var(--bg-stats); border-radius: 16px; padding: 40px; margin: 40px auto; max-width: 1400px; box-shadow: 0 4px 20px var(--shadow-color);">
                    
                    <!-- Upper Section: Research Questionnaires -->
                    <div style="margin-bottom: 40px;">
                        <div style="text-align: center; margin-bottom: 25px;">
                            <h2 style="font-size: 1.5em; font-weight: 700; color: var(--text-secondary); margin-bottom: 10px;">Research Questionnaires</h2>
                        </div>
                        
                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 30px;">
                            <!-- Google Forms Survey -->
                            <div style="background: var(--bg-secondary); border-radius: 12px; padding: 25px; box-shadow: 0 2px 12px var(--card-shadow); border: 1px solid var(--border-color); display: flex; flex-direction: column; align-items: center;">
                                <img src="https://blog-img-1259433191.cos.ap-shanghai.myqcloud.com/venus/img/venusfactory_googleform.png" alt="Google Survey" style="width: auto; height: 120px; border-radius: 8px; margin-bottom: 15px;">
                                <h4 style="font-size: 1.2em; font-weight: 600; color: var(--text-secondary); margin: 0;">Google Survey</h4>
                            </div>

                            <!-- Wenjuanxing Survey -->
                            <div style="background: var(--bg-secondary); border-radius: 12px; padding: 25px; box-shadow: 0 2px 12px var(--card-shadow); border: 1px solid var(--border-color); display: flex; flex-direction: column; align-items: center;">
                                <img src="https://blog-img-1259433191.cos.ap-shanghai.myqcloud.com/venus/img/venusfactory_wenjuanxing.png" alt="问卷星" style="width: auto; height: 120px; border-radius: 8px; margin-bottom: 15px;">
                                <h4 style="font-size: 1.2em; font-weight: 600; color: var(--text-secondary); margin: 0;">问卷星</h4>
                            </div>
                        </div>
                    </div>

                    <div style="margin-bottom: 40px;">
                        <div style="text-align: center; margin-bottom: 25px;">
                            <h2 style="font-size: 1.5em; font-weight: 700; color: var(--text-secondary); margin-bottom: 10px;">Partner Institutions</h2>
                        </div>
                        
                        <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px;">
                            <!-- Card 1: Shanghai Jiao Tong University -->
                            <a href="https://www.sjtu.edu.cn/" target="_blank" style="text-decoration: none; color: inherit; display: flex;">
                                <!-- The 'justify-content' property below has been changed from 'center' to 'flex-start' -->
                                <div style="background: var(--bg-secondary); border-radius: 12px; padding: 20px; box-shadow: 0 2px 12px var(--card-shadow); transition: all 0.3s ease; border: 1px solid var(--border-color); display: flex; flex-direction: column; align-items: center; text-align: center; gap: 15px; min-height: 120px; justify-content: flex-start; width: 100%;">
                                    <img src="https://blog-img-1259433191.cos.ap-shanghai.myqcloud.com/venus/img/sjtu_logo.jpg" alt="SJTU" style="height: 50px;">
                                    <div style="font-size: 1.0em; font-weight: 600; color: var(--text-secondary);">Shanghai Jiao Tong University</div>
                                </div>
                            </a>
                            <!-- Card 2: East China University of Science and Technology -->
                            <a href="https://www.ecust.edu.cn/" target="_blank" style="text-decoration: none; color: inherit; display: flex;">
                                <!-- The 'justify-content' property below has been changed from 'center' to 'flex-start' -->
                                <div style="background: var(--bg-secondary); border-radius: 12px; padding: 20px; box-shadow: 0 2px 12px var(--card-shadow); transition: all 0.3s ease; border: 1px solid var(--border-color); display: flex; flex-direction: column; align-items: center; text-align: center; gap: 15px; min-height: 120px; justify-content: flex-start; width: 100%;">
                                    <img src="https://blog-img-1259433191.cos.ap-shanghai.myqcloud.com/venus/img/ecust_logo.jpg" alt="ECUST" style="height: 50px;">
                                    <div style="font-size: 1.0em; font-weight: 600; color: var(--text-secondary);">East China University of Science and Technology</div>
                                </div>
                            </a>
                            <!-- Card 3: Shanghai AI Laboratory -->
                            <a href="https://www.shlab.org.cn/" target="_blank" style="text-decoration: none; color: inherit; display: flex;">
                                <!-- The 'justify-content' property below has been changed from 'center' to 'flex-start' -->
                                <div style="background: var(--bg-secondary); border-radius: 12px; padding: 20px; box-shadow: 0 2px 12px var(--card-shadow); transition: all 0.3s ease; border: 1px solid var(--border-color); display: flex; flex-direction: column; align-items: center; text-align: center; gap: 15px; min-height: 120px; justify-content: flex-start; width: 100%;">
                                    <img src="https://blog-img-1259433191.cos.ap-shanghai.myqcloud.com/venus/img/shailab_logo.jpg" alt="SHAILab" style="height: 50px;">
                                    <div style="font-size: 1.0em; font-weight: 600; color: var(--text-secondary);">Shanghai AI Laboratory</div>
                                </div>
                            </a>
                        </div>
                    </div>
                </div>
                
                <!-- Developer Information Area -->
                <div style="background: var(--bg-card); border-radius: 12px; padding: 30px; margin: 30px auto; max-width: 1400px;">
                    <div style="text-align: center; margin-bottom: 25px;">
                        <h2 style="font-size: 1.5em; font-weight: 700; color: var(--text-secondary); margin-bottom: 10px;">Cooperation Platform & Developer Information</h2>
                    </div>
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; font-size: 1.1em;">
                        <div>  
                            <div style="margin-bottom: 12px;"><b>🤝 Cooperation Platform:</b> <a href="https://hyper.ai/cn/tutorials/38568" target="_blank" style="color: var(--accent-color);">HyperAI</a></div>
                            <div style="margin-bottom: 12px;"><b>🧬 Few-shot mutation prediction tool:</b> <a href="https://github.com/ai4protein/Pro-FSFP" target="_blank" style="color: var(--accent-color);">Pro-FSFP</a></div>
                            <div style="margin-bottom: 12px;"><b>⚡ The most advanced zero-shot protein prediction tool:</b> <a href="https://github.com/ai4protein/VenusREM" target="_blank" style="color: var(--accent-color);">VenusREM</a></div>
                        </div>
                        <div>
                            <div style="margin-bottom: 12px;"><b>🏠 Developer homepage:</b> <a href="https://tyang816.github.io/" target="_blank" style="color: var(--accent-color);">https://tyang816.github.io/</a></div>
                            <div style="margin-bottom: 12px;"><b>✉️ Developer contact information:</b> <a href="mailto:tanyang.august@sjtu.edu.cn" style="color: var(--accent-color);">tanyang.august@sjtu.edu.cn</a>, <a href="mailto:zlr_zmm@163.com" style="color: var(--accent-color);">zlr_zmm@163.com</a></div>
                        </div>
                    </div>
                </div>

                <!-- Citation Area -->
                <div style="background: var(--bg-card); border-radius: 12px; padding: 30px; margin: 30px auto; max-width: 1400px;">
                    <div style="text-align: center; margin-bottom: 25px;">
                        <h2 style="font-size: 1.5em; font-weight: 700; color: var(--text-secondary); margin-bottom: 10px;">Citation</h2>
                    </div>
                    <pre style="background: var(--background-fill-secondary); color: var(--text-primary); border: 1px solid var(--border-color-primary); border-radius: 8px; padding: 18px; font-size: 1.2em; overflow-x: auto;">
<code>@inproceedings&#123;&#123;tan-etal-2025-venusfactory,
    title = "&#123;V&#125;enus&#123;F&#125;actory: An Integrated System for Protein Engineering with Data Retrieval and Language Model Fine-Tuning",
    author = "Tan, Yang and Liu, Chen and Gao, Jingyuan and Wu, Banghao and Li, Mingchen and Wang, Ruilin and Zhang, Lingrong and Yu, Huiqun and Fan, Guisheng and Hong, Liang and Zhou, Bingxin",
    editor = "Mishra, Pushkar and Muresan, Smaranda and Yu, Tao",
    booktitle = "Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 3: System Demonstrations)",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.acl-demo.23/",
    doi = "10.18653/v1/2025.acl-demo.23",
    pages = "230--241",
    ISBN = "979-8-89176-253-4",
&#125;&#125;</code>
                    </pre>
                </div>
            </div>

            

            <hr style="margin: 40px 0; border: 1px solid var(--border-color);">
            <!-- Additional Information Section -->
            <div style="background: var(--bg-card); border-radius: 14px; box-shadow: 0 2px 12px var(--card-shadow); padding: 32px 28px 24px 28px; max-width: 1400px; margin: 40px auto 0 auto;">
                <h1 style="font-size:2em; font-weight:900; color:var(--accent-color); margin-bottom: 0.5em; border-bottom: 2px solid var(--border-color); padding-bottom: 0.2em; letter-spacing: 1px;">Additional Information</h1>
                <div style="display: flex; flex-wrap: wrap; gap: 40px;">
                    <!-- Supported Models -->
                    <div style="flex:1; min-width: 320px;">
                        <b style="font-size:1.1em; color:var(--text-secondary);">Supported Models:</b>
                        <ul style="margin: 18px 0 0 0; padding: 0; list-style: none;">
                            <li style="margin-bottom: 12px;">
                                <span style="font-weight:bold; color:var(--accent-color);">ESM-1v/ESM-1b/ESM-650M</span>
                                <span style="font-size:0.87em; color:var(--text-muted);"> – State-of-the-art protein language models from Meta AI for sequence-based prediction.</span>
                                <a href="https://www.biorxiv.org/content/10.1101/622803v4.full" target="_blank" style="margin-left:8px; color:var(--accent-color); text-decoration:underline; font-size:0.87em;">[Paper]</a>
                                <a href="https://github.com/facebookresearch/esm" target="_blank" style="margin-left:8px; color:var(--accent-color); text-decoration:underline; font-size:0.87em;">[Code]</a>
                            </li>
                            <li style="margin-bottom: 12px;">
                                <span style="font-weight:bold; color:var(--accent-color);">SaProt</span>
                                <span style="font-size:0.87em; color:var(--text-muted);"> – A model for protein sequence analysis and function prediction.</span>
                                <a href="https://www.biorxiv.org/content/10.1101/2023.10.01.560349v5" target="_blank" style="margin-left:8px; color:var(--accent-color); text-decoration:underline; font-size:0.87em;">[Paper]</a>
                                <a href="https://github.com/westlake-repl/SaProt" target="_blank" style="margin-left:8px; color:var(--accent-color); text-decoration:underline; font-size:0.87em;">[Code]</a>
                            </li>
                            <li style="margin-bottom: 12px;">
                                <span style="font-weight:bold; color:var(--accent-color);">MIF-ST</span>
                                <span style="font-size:0.87em; color:var(--text-muted);"> – Structure-informed models for protein fitness and mutation effect prediction.</span>
                                <a href="https://www.biorxiv.org/content/10.1101/2022.05.25.493516v1.full" target="_blank" style="margin-left:8px; color:var(--accent-color); text-decoration:underline; font-size:0.87em;">[Paper]</a>
                                <a href="https://github.com/microsoft/protein-sequence-models" target="_blank" style="margin-left:8px; color:var(--accent-color); text-decoration:underline; font-size:0.87em;">[Code]</a>
                            </li>
                            <li style="margin-bottom: 12px;">
                                <span style="font-weight:bold; color:var(--accent-color);">ProSST-2048</span>
                                <span style="font-size:0.87em; color:var(--text-muted);"> – Large-scale protein sequence-structure models for zero-shot and supervised tasks.</span>
                                <a href="https://www.biorxiv.org/content/10.1101/2024.04.15.589672v2.full" target="_blank" style="margin-left:8px; color:var(--accent-color); text-decoration:underline; font-size:0.87em;">[Paper]</a>
                                <a href="https://github.com/ai4protein/ProSST" target="_blank" style="margin-left:8px; color:var(--accent-color); text-decoration:underline; font-size:0.87em;">[Code]</a>
                            </li>
                            <li style="margin-bottom: 12px;">
                                <span style="font-weight:bold; color:var(--accent-color);">ProtSSN</span>
                                <span style="font-size:0.87em; color:var(--text-muted);"> – Protein structure and sequence network for protein structure prediction.</span>
                                <a href="https://elifesciences.org/reviewed-preprints/98033" target="_blank" style="margin-left:8px; color:var(--accent-color); text-decoration:underline; font-size:0.87em;">[Paper]</a>
                                <a href="https://github.com/ai4protein/ProtSSN" target="_blank" style="margin-left:8px; color:var(--accent-color); text-decoration:underline; font-size:0.87em;">[Code]</a>
                            </li>
                            <li style="margin-bottom: 12px;">
                                <span style="font-weight:bold; color:var(--accent-color);">Ankh-large</span>
                                <span style="font-size:0.87em; color:var(--text-muted);"> – Transformer-based protein language model for structure and function tasks.</span>
                                <a href="https://www.biorxiv.org/content/10.1101/2023.01.16.524265v1.full" target="_blank" style="margin-left:8px; color:var(--accent-color); text-decoration:underline; font-size:0.87em;">[Paper]</a>
                                <a href="https://github.com/agemagician/Ankh" target="_blank" style="margin-left:8px; color:var(--accent-color); text-decoration:underline; font-size:0.87em;">[Code]</a>
                            </li>
                            <li style="margin-bottom: 12px;">
                                <span style="font-weight:bold; color:var(--accent-color);">ProtBert-uniref50</span>
                                <span style="font-size:0.87em; color:var(--text-muted);"> – BERT-based protein model trained on UniRef50.</span>
                                <a href="https://www.biorxiv.org/content/10.1101/2021.05.24.445464v1" target="_blank" style="margin-left:8px; color:var(--accent-color); text-decoration:underline; font-size:0.87em;">[Paper]</a>
                                <a href="https://github.com/agemagician/ProtTrans" target="_blank" style="margin-left:8px; color:var(--accent-color); text-decoration:underline; font-size:0.87em;">[Code]</a>
                            </li>
                            <li style="margin-bottom: 12px;">
                                <span style="font-weight:bold; color:var(--accent-color);">ProtT5-xl-uniref50</span>
                                <span style="font-size:0.87em; color:var(--text-muted);"> – T5-based protein model for sequence and structure prediction.</span>
                                <a href="https://arxiv.org/abs/2007.06225" target="_blank" style="margin-left:8px; color:var(--accent-color); text-decoration:underline; font-size:0.87em;">[Paper]</a>
                                <a href="https://github.com/agemagician/ProtTrans" target="_blank" style="margin-left:8px; color:var(--accent-color); text-decoration:underline; font-size:0.87em;">[Code]</a>
                            </li>
                        </ul>
                    </div>
                    <!-- Supported Datasets -->
                    <div style="flex:1; min-width: 320px;">
                        <b style="font-size:1.1em; color:var(--text-secondary);">Supported Datasets:</b>
                        <ul style="margin: 18px 0 0 0; padding: 0; list-style: none;">
                            <li style="margin-bottom: 12px;">
                                <span style="font-weight:bold; color:var(--accent-color);">DeepSol</span>
                                <span style="font-size:0.87em; color:var(--text-muted);"> – Dataset for protein solubility prediction.</span>
                                <a href="https://academic.oup.com/bioinformatics/article/34/15/2605/4938490" target="_blank" style="margin-left:8px; color:var(--accent-color); text-decoration:underline; font-size:0.87em;">[Paper]</a>
                                <a href="https://zenodo.org/records/1162886" target="_blank" style="margin-left:8px; color:var(--accent-color); text-decoration:underline; font-size:0.87em;">[Datasets]</a>
                            </li>
                            <li style="margin-bottom: 12px;">
                                <span style="font-weight:bold; color:var(--accent-color);">DeepSoluE</span>
                                <span style="font-size:0.87em; color:var(--text-muted);"> – Enhanced solubility dataset for benchmarking.</span>
                                <a href="https://bmcbiol.biomedcentral.com/articles/10.1186/s12915-023-01510-8" target="_blank" style="margin-left:8px; color:var(--accent-color); text-decoration:underline; font-size:0.87em;">[Paper]</a>
                                <a href="https://github.com/wangchao-malab/DeepSoluE" target="_blank" style="margin-left:8px; color:var(--accent-color); text-decoration:underline; font-size:0.87em;">[Datasets]</a>
                            </li>
                            <li style="margin-bottom: 12px;">
                                <span style="font-weight:bold; color:var(--accent-color);">ProtSolM</span>
                                <span style="font-size:0.87em; color:var(--text-muted);"> – Solubility dataset for machine learning tasks.</span>
                                <a href="https://arxiv.org/abs/2406.19744" target="_blank" style="margin-left:8px; color:var(--accent-color); text-decoration:underline; font-size:0.87em;">[Paper]</a>
                                <a href="https://huggingface.co/datasets/AI4Protein/ProtSolM_ESMFold_PDB" target="_blank" style="margin-left:8px; color:var(--accent-color); text-decoration:underline; font-size:0.87em;">[Datasets]</a>
                            </li>
                            <li style="margin-bottom: 12px;">
                                <span style="font-weight:bold; color:var(--accent-color);">DeepLocBinary</span>
                                <span style="font-size:0.87em; color:var(--text-muted);"> – Dataset for binary protein subcellular localization prediction.</span>
                                <a href="https://academic.oup.com/bioinformatics/article/33/21/3387/4099600" target="_blank" style="margin-left:8px; color:var(--accent-color); text-decoration:underline; font-size:0.87em;">[Paper]</a>
                                <a href="https://huggingface.co/datasets/AI4Protein/DeepLocBinary" target="_blank" style="margin-left:8px; color:var(--accent-color); text-decoration:underline; font-size:0.87em;">[Datasets]</a>
                            </li>
                            <li style="margin-bottom: 12px;">
                                <span style="font-weight:bold; color:var(--accent-color);">DeepLocMulti</span>
                                <span style="font-size:0.87em; color:var(--text-muted);"> – Dataset for multi-class protein subcellular localization prediction.</span>
                                <a href="https://academic.oup.com/bioinformatics/article/33/21/3387/4099600" target="_blank" style="margin-left:8px; color:var(--accent-color); text-decoration:underline; font-size:0.87em;">[Paper]</a>
                                <a href="https://huggingface.co/datasets/AI4Protein/DeepLocMulti" target="_blank" style="margin-left:8px; color:var(--accent-color); text-decoration:underline; font-size:0.87em;">[Datasets]</a>
                            </li>
                            <li style="margin-bottom: 12px;">
                                <span style="font-weight:bold; color:var(--accent-color);">MetalIonBinding</span>
                                <span style="font-size:0.87em; color:var(--text-muted);"> – Dataset for protein metal ion binding site prediction.</span>
                                <a href="https://www.biorxiv.org/content/10.1101/2023.10.01.560349v5" target="_blank" style="margin-left:8px; color:var(--accent-color); text-decoration:underline; font-size:0.87em;">[Paper]</a>
                                <a href="https://huggingface.co/datasets/AI4Protein/MetalIonBinding" target="_blank" style="margin-left:8px; color:var(--accent-color); text-decoration:underline; font-size:0.87em;">[Datasets]</a>
                            </li>
                            <li style="margin-bottom: 12px;">
                                <span style="font-weight:bold; color:var(--accent-color);">Thermostability</span>
                                <span style="font-size:0.87em; color:var(--text-muted);"> – Dataset for protein thermostability prediction.</span>
                                <a href="https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/file/2b44928ae11fb9384c4cf38708677c48-Paper-round2.pdf" target="_blank" style="margin-left:8px; color:var(--accent-color); text-decoration:underline; font-size:0.87em;">[Paper]</a>
                                <a href="https://huggingface.co/datasets/AI4Protein/Thermostability" target="_blank" style="margin-left:8px; color:var(--accent-color); text-decoration:underline; font-size:0.87em;">[Datasets]</a>
                            </li>
                            <li style="margin-bottom: 12px;">
                                <span style="font-weight:bold; color:var(--accent-color);">SortingSignal</span>
                                <span style="font-size:0.87em; color:var(--text-muted);"> – Dataset for protein sorting signal prediction.</span>
                                <a href="https://www.nature.com/articles/s41587-019-0036-z" target="_blank" style="margin-left:8px; color:var(--accent-color); text-decoration:underline; font-size:0.87em;">[Paper]</a>
                                <a href="https://huggingface.co/datasets/AI4Protein/SortingSignal" target="_blank" style="margin-left:8px; color:var(--accent-color); text-decoration:underline; font-size:0.87em;">[Datasets]</a>
                            </li>
                            <li>
                                <span style="font-weight:bold; color:var(--accent-color);">DeepET_Topt</span>
                                <span style="font-size:0.87em; color:var(--text-muted);"> – Dataset for optimal growth temperature (Topt) prediction.</span>
                                <a href="https://academic.oup.com/bib/article/26/2/bbaf114/8074761" target="_blank" style="margin-left:8px; color:var(--accent-color); text-decoration:underline; font-size:0.87em;">[Paper]</a>
                                <a href="https://huggingface.co/datasets/AI4Protein/DeepET_Topt" target="_blank" style="margin-left:8px; color:var(--accent-color); text-decoration:underline; font-size:0.87em;">[Datasets]</a>
                            </li>
                        </ul>
                    </div>
                </div>
            </div>
            
            ''')


    with gr.Tab("Training"):
        training_content = load_manual_training(language.value)
        toc_html, html_content = generate_toc_and_content(training_content)
        training_md = gr.HTML(f"""
            <div class="manual-container">
                {toc_html}
                <div class="manual-content">{html_content}</div>
            </div>
        """)
    
    with gr.Tab("Prediction"):
        prediction_content = load_manual_prediction(language.value)
        toc_html, html_content = generate_toc_and_content(prediction_content)
        prediction_md = gr.HTML(f"""
            <div class="manual-container">
                {toc_html}
                <div class="manual-content">{html_content}</div>
            </div>
        """)
    
    with gr.Tab("Evaluation"):
        evaluation_content = load_manual_evaluation(language.value)
        toc_html, html_content = generate_toc_and_content(evaluation_content)
        evaluation_md = gr.HTML(f"""
            <div class="manual-container">
                {toc_html}
                <div class="manual-content">{html_content}</div>
            </div>
        """)
    
    with gr.Tab("Download"):
        download_content = load_manual_download(language.value)
        toc_html, html_content = generate_toc_and_content(download_content)
        download_md = gr.HTML(f"""
            <div class="manual-container">
                {toc_html}
                <div class="manual-content">{html_content}</div>
            </div>
        """)
    
    with gr.Tab("FAQ"):
        faq_content = load_manual_faq(language.value)
        toc_html, html_content = generate_toc_and_content(faq_content)
        faq_md = gr.HTML(f"""
            <div class="manual-container">
                {toc_html}
                <div class="manual-content">{html_content}</div>
            </div>
        """)
    
    # correctly bind language switch event
    language.change(
        fn=update_manual,
        inputs=[language],
        outputs=[training_md, prediction_md, evaluation_md, download_md, faq_md]
    )
    
    return {"training_md": training_md, "prediction_md": prediction_md, "evaluation_md": evaluation_md, "download_md": download_md, "faq_md": faq_md}

def update_manual(language):
    """Update the manual content
    Args:
        language: language
    Returns:
        training_md: training manual
        prediction_md: prediction manual
        evaluation_md: evaluation manual
        download_md: download manual
        faq_md: faq manual
    """
    training_content = load_manual_training(language)
    prediction_content = load_manual_prediction(language)
    evaluation_content = load_manual_evaluation(language)
    download_content = load_manual_download(language)
    faq_content = load_manual_faq(language)
    
    # Use Python's markdown library to convert Markdown to HTML
    def markdown_to_html(markdown_content, base_path="src/web/manual"):
        """Convert Markdown content to HTML, and embed images as base64 encoded"""
        # Process image paths, use base64 encoding to directly embed images
        def embed_image(match):
            alt_text = match.group(1)
            img_path = match.group(2)
            
            # Check if the path is an external URL
            if img_path.startswith(('http://', 'https://')):
                return f'<img src="{img_path}" alt="{alt_text}" />'
            
            # Process local image paths
            try:
                # Remove the leading / to get the correct path
                if img_path.startswith('/'):
                    img_path = img_path[1:]
                
                # Get the absolute path
                current_dir = os.path.dirname(os.path.abspath(__file__))
                project_root = os.path.dirname(os.path.dirname(current_dir))
                abs_img_path = os.path.join(project_root, img_path)
                
                # Read the image and convert it to base64
                import base64
                from pathlib import Path
                
                image_path = Path(abs_img_path)
                if image_path.exists():
                    image_type = image_path.suffix.lstrip('.').lower()
                    if image_type == 'jpg':
                        image_type = 'jpeg'
                        
                    with open(image_path, "rb") as img_file:
                        encoded_string = base64.b64encode(img_file.read()).decode('utf-8')
                        
                    return f'<img src="data:image/{image_type};base64,{encoded_string}" alt="{alt_text}" style="max-width:100%; height:auto;" />'
                else:
                    print(f"Image file does not exist: {abs_img_path}")
                    return f'<span style="color:red;">[Image does not exist: {img_path}]</span>'
                
            except Exception as e:
                print(f"Error processing image: {e}, path: {img_path}")
                return f'<span style="color:red;">[Image loading error: {img_path}]</span>'
        
        # Use regular expression to process all image tags
        pattern = r'!\[(.*?)\]\((.*?)\)'
        processed_content = re.sub(pattern, embed_image, markdown_content)
        
        # Use Python's markdown library to convert
        html = markdown.markdown(
            processed_content, 
            extensions=[
                'tables', 
                'fenced_code', 
                'codehilite', 
                'nl2br', 
                'extra',
                'mdx_truly_sane_lists'
            ],
            extension_configs={
                'mdx_truly_sane_lists': {
                    'nested_indent': 2,
                    'truly_sane': True
                }
            }
        )
        
        return html
    
    # Generate HTML navigation bar and process content from Markdown content
    def generate_toc_and_content(markdown_content):
        """Generate HTML navigation bar and process content from Markdown content"""
        # Extract all headers
        headers = re.findall(r'^(#{1,3})\s+(.+)$', markdown_content, re.MULTILINE)
        
        if not headers:
            return "<div class='manual-nav'><p>目录加载中...</p></div>", markdown_content
        
        toc_html = "<div class='manual-nav'><ul>"
        
        # Create navigation items for each header
        for i, (level, title) in enumerate(headers):
            level_num = len(level)
            header_id = f"header-{i}"
            
            # Add class based on header level
            css_class = ""
            if level_num == 2:
                css_class = "nav-h2"
            elif level_num == 3:
                css_class = "nav-h3"
            
            toc_html += f"<li><a href='#{header_id}' class='{css_class}'>{title}</a></li>"
        
        toc_html += "</ul></div>"
        
        # Add ID to headers in Markdown content
        processed_content = markdown_content
        for i, (level, title) in enumerate(headers):
            header_id = f"header-{i}"
            header_pattern = f"{level} {title}"
            header_replacement = f"{level} <span id='{header_id}'></span>{title}"
            processed_content = processed_content.replace(header_pattern, header_replacement, 1)
        
        # Convert processed Markdown to HTML
        html_content = markdown_to_html(processed_content)
        
        return toc_html, html_content
    
    # Generate HTML with navigation bar
    training_toc, training_html = generate_toc_and_content(training_content)
    prediction_toc, prediction_html = generate_toc_and_content(prediction_content)
    evaluation_toc, evaluation_html = generate_toc_and_content(evaluation_content)
    download_toc, download_html = generate_toc_and_content(download_content)
    faq_toc, faq_html = generate_toc_and_content(faq_content)
    
    training_output = f"""
        <div class="manual-container">
            {training_toc}
            <div class="manual-content">{training_html}</div>
        </div>
    """
    
    prediction_output = f"""
        <div class="manual-container">
            {prediction_toc}
            <div class="manual-content">{prediction_html}</div>
        </div>
    """
    
    evaluation_output = f"""
        <div class="manual-container">
            {evaluation_toc}
            <div class="manual-content">{evaluation_html}</div>
        </div>
    """
    
    download_output = f"""
        <div class="manual-container">
            {download_toc}
            <div class="manual-content">{download_html}</div>
        </div>
    """
    
    faq_output = f"""
        <div class="manual-container">
            {faq_toc}
            <div class="manual-content">{faq_html}</div>
        </div>
    """
    
    return training_output, prediction_output, evaluation_output, download_output, faq_output

def load_manual_training(language):
    if language == 'Chinese':
        manual_path = os.path.join("src/web/manual", "TrainingManual_ZH.md")
    else:
        manual_path = os.path.join("src/web/manual", "TrainingManual_EN.md")
    try:
        with open(manual_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"# Error loading manual\n\n{str(e)}"

def load_manual_prediction(language):
    if language == 'Chinese':
        manual_path = os.path.join("src/web/manual", "PredictionManual_ZH.md")
    else:
        manual_path = os.path.join("src/web/manual", "PredictionManual_EN.md")
    try:
        with open(manual_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"# Error loading manual\n\n{str(e)}"

def load_manual_evaluation(language):
    if language == 'Chinese':
        manual_path = os.path.join("src/web/manual", "EvaluationManual_ZH.md")
    else:
        manual_path = os.path.join("src/web/manual", "EvaluationManual_EN.md")
    try:
        with open(manual_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"# Error loading manual\n\n{str(e)}"
    
def load_manual_download(language):
    if language == 'Chinese':
        manual_path = os.path.join("src/web/manual", "DownloadManual_ZH.md")
    else:
        manual_path = os.path.join("src/web/manual", "DownloadManual_EN.md")
    try:
        with open(manual_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"# Error loading manual\n\n{str(e)}"

def load_manual_faq(language):
    if language == 'Chinese':
        manual_path = os.path.join("src/web/manual", "QAManual_ZH.md")
    else:
        manual_path = os.path.join("src/web/manual", "QAManual_EN.md")
    try:
        with open(manual_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"# FAQ\n\n{str(e)}"
