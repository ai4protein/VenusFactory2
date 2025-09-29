import gradio as gr
import os
import re
import markdown
from typing import Dict, Any
from .index_tab import create_index_tab

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
        gr.HTML(create_index_tab(constant))

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
    quicktools_content = load_manual_quicktools(language)
    advancedtools_content = load_manual_advancedtools(language)
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
