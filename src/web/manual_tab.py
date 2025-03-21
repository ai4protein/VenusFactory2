import gradio as gr
import os
import re
import markdown
from typing import Dict, Any

def create_manual_tab(constant: Dict[str, Any]) -> Dict[str, Any]:
    # 添加自定义CSS，增大文本大小并添加左侧导航栏样式
    custom_css = """
    <style>
        /* 增大整体文本大小 */
        .manual-content {
            font-size: 16px !important;
            line-height: 1.6 !important;
        }
        
        /* 标题样式 */
        .manual-content h1 {
            font-size: 28px !important;
            margin-top: 30px !important;
            margin-bottom: 20px !important;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }
        
        .manual-content h2 {
            font-size: 24px !important;
            margin-top: 25px !important;
            margin-bottom: 15px !important;
            border-bottom: 1px solid #ddd;
            padding-bottom: 8px;
        }
        
        .manual-content h3 {
            font-size: 20px !important;
            margin-top: 20px !important;
            margin-bottom: 10px !important;
        }
        
        .manual-content h4 {
            font-size: 18px !important;
            margin-top: 15px !important;
            margin-bottom: 10px !important;
        }
        
        /* 段落和列表样式 */
        .manual-content p, .manual-content li {
            font-size: 16px !important;
            margin-bottom: 10px !important;
        }
        
        /* 嵌套列表样式 */
        .manual-content ul, .manual-content ol {
            padding-left: 25px !important;
            margin-bottom: 15px !important;
            list-style-position: outside !important;
        }
        
        .manual-content ul ul, 
        .manual-content ol ol,
        .manual-content ul ol,
        .manual-content ol ul {
            margin-top: 5px !important;
            margin-bottom: 5px !important;
            padding-left: 25px !important;
        }
        
        .manual-content ul {
            list-style-type: disc !important;
        }
        
        .manual-content ul ul {
            list-style-type: circle !important;
        }
        
        .manual-content ul ul ul {
            list-style-type: square !important;
        }
        
        .manual-content ol {
            list-style-type: decimal !important;
        }
        
        .manual-content ol ol {
            list-style-type: lower-alpha !important;
        }
        
        .manual-content ol ol ol {
            list-style-type: lower-roman !important;
        }
        
        /* 确保列表项正确显示 */
        .manual-content li {
            display: list-item !important;
            margin-bottom: 5px !important;
        }
        
        .manual-content li p {
            margin-bottom: 5px !important;
            display: inline-block !important;
        }
        
        /* 代码块样式 */
        .manual-content pre {
            background-color: #f5f5f5 !important;
            padding: 15px !important;
            border-radius: 5px !important;
            overflow-x: auto !important;
            margin: 15px 0 !important;
        }
        
        .manual-content code {
            font-family: 'Courier New', Courier, monospace !important;
            font-size: 15px !important;
        }
        
        /* 表格样式 */
        .manual-content table {
            width: 100% !important;
            border-collapse: collapse !important;
            margin: 20px 0 !important;
        }
        
        .manual-content th, .manual-content td {
            border: 1px solid #ddd !important;
            padding: 12px !important;
            text-align: left !important;
        }
        
        .manual-content th {
            background-color: #f2f2f2 !important;
            font-weight: bold !important;
        }
        
        /* 左侧导航栏样式 */
        .manual-container {
            display: flex !important;
            width: 100% !important;
            position: relative !important;
        }
        
        .manual-nav {
            width: 250px !important;
            padding: 18px !important;
            background-color: #f8f9fa !important;
            border-right: 1px solid #ddd !important;
            border-radius: 5px !important;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1) !important;
            font-size: 14px !important;
            line-height: 1.4 !important;
            align-self: flex-start !important;
            position: sticky !important;
            top: 20px !important;
            max-height: 100% !important;
            overflow-y: auto !important;
        }
        
        .manual-nav ul {
            list-style-type: none !important;
            padding: 0 !important;
            margin: 0 !important;
        }
        
        .manual-nav li {
            margin-bottom: 5px !important;
        }
        
        .manual-nav a {
            display: block !important;
            padding: 6px 8px !important;
            color: #333 !important;
            text-decoration: none !important;
            border-radius: 4px !important;
            line-height: 1.4 !important;
            transition: all 0.2s ease !important;
        }
        
        .manual-nav a:hover {
            background-color: #e9ecef !important;
            transform: translateX(2px) !important;
        }
        
        .manual-nav .nav-h2 {
            padding-left: 15px !important;
            font-size: 13px !important;
        }
        
        .manual-nav .nav-h3 {
            padding-left: 30px !important;
            font-size: 12px !important;
        }
        
        .manual-content {
            flex: 1 !important;
            padding: 20px !important;
            overflow-y: auto !important;
            margin-left: 20px !important;
        }
        
        /* 响应式设计 */
        @media (max-width: 768px) {
            .manual-container {
                flex-direction: column !important;
            }
            
            .manual-nav {
                position: static !important;
                width: 100% !important;
                height: auto !important;
                max-height: 300px !important;
                margin-bottom: 20px !important;
            }
            
            .manual-content {
                margin-left: 0 !important;
                width: 100% !important;
            }
        }
        
        /* 滚动条样式 */
        .manual-nav::-webkit-scrollbar {
            width: 6px !important;
        }
        
        .manual-nav::-webkit-scrollbar-track {
            background: #f1f1f1 !important;
            border-radius: 10px !important;
        }
        
        .manual-nav::-webkit-scrollbar-thumb {
            background: #c1c1c1 !important;
            border-radius: 10px !important;
        }
        
        .manual-nav::-webkit-scrollbar-thumb:hover {
            background: #a8a8a8 !important;
        }
        
        /* 强化列表样式 */
        .manual-content ul li, .manual-content ol li {
            margin-bottom: 8px !important;
            line-height: 1.5 !important;
        }
        
        .manual-content ul li:last-child, .manual-content ol li:last-child {
            margin-bottom: 0 !important;
        }
        
        .manual-content ul ul, .manual-content ol ol, .manual-content ul ol, .manual-content ol ul {
            margin-top: 8px !important;
        }
        
        /* 强调列表项内容 */
        .manual-content li strong, .manual-content li b {
            color: #2c3e50 !important;
        }
        
        /* 确保列表项内的代码块正确显示 */
        .manual-content li code {
            background-color: #f5f5f5 !important;
            padding: 2px 4px !important;
            border-radius: 3px !important;
            font-size: 14px !important;
            color: #e83e8c !important;
        }
        
        /* 添加到您现有的custom_css字符串中 */
        .manual-content img {
            max-width: 100% !important;
            height: auto !important;
            display: block !important;
            margin: 20px auto !important;
            border-radius: 5px !important;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1) !important;
        }
        
        /* 为图片添加描述样式 */
        .manual-content p img + em {
            display: block !important;
            text-align: center !important;
            color: #666 !important;
            font-size: 14px !important;
            margin-top: 8px !important;
        }
        
        /* 图片点击放大效果相关样式 */
        .manual-content img:hover {
            cursor: pointer !important;
            transform: scale(1.01) !important;
            transition: transform 0.2s ease !important;
        }
    </style>
    """

    # 添加JavaScript代码，用于处理导航点击
    custom_js = """
    <script>
    document.addEventListener('DOMContentLoaded', function() {
        // 为所有导航链接添加点击事件
        document.querySelectorAll('.manual-nav a').forEach(function(link) {
            link.addEventListener('click', function(e) {
                e.preventDefault();
                const targetId = this.getAttribute('href').substring(1);
                const targetElement = document.getElementById(targetId);
                if (targetElement) {
                    targetElement.scrollIntoView({
                        behavior: 'smooth'
                    });
                }
            });
        });
        
        // 为所有手册内容中的图片添加点击事件
        document.querySelectorAll('.manual-content img').forEach(function(img) {
            img.addEventListener('click', function() {
                // 创建一个模态框来显示大图
                const modal = document.createElement('div');
                modal.style.cssText = 'position:fixed; top:0; left:0; width:100%; height:100%; background-color:rgba(0,0,0,0.8); display:flex; justify-content:center; align-items:center; z-index:9999;';
                
                // 创建大图元素
                const largeImg = document.createElement('img');
                largeImg.src = this.src;
                largeImg.style.cssText = 'max-width:90%; max-height:90%; object-fit:contain;';
                
                // 将大图添加到模态框
                modal.appendChild(largeImg);
                
                // 点击模态框关闭它
                modal.addEventListener('click', function() {
                    document.body.removeChild(modal);
                });
                
                // 将模态框添加到body
                document.body.appendChild(modal);
            });
        });
    });
    </script>
    """

    # 使用Python的markdown库将Markdown转换为HTML
    def markdown_to_html(markdown_content, base_path="src/web/manual"):
        """将Markdown内容转换为HTML，并将图片嵌入为base64编码"""
        # 处理图片路径，使用base64编码直接嵌入图片
        def embed_image(match):
            alt_text = match.group(1)
            img_path = match.group(2)
            
            # 检查路径是否为外部URL
            if img_path.startswith(('http://', 'https://')):
                return f'<img src="{img_path}" alt="{alt_text}" />'
            
            # 处理本地图片路径
            try:
                # 去掉开头的/以获取正确的路径
                if img_path.startswith('/'):
                    img_path = img_path[1:]
                
                # 获取绝对路径
                current_dir = os.path.dirname(os.path.abspath(__file__))
                project_root = os.path.dirname(os.path.dirname(current_dir))
                abs_img_path = os.path.join(project_root, img_path)
                
                # 读取图片并转换为base64
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
                    print(f"图片文件不存在: {abs_img_path}")
                    return f'<span style="color:red;">[图片不存在: {img_path}]</span>'
                
            except Exception as e:
                print(f"处理图片时出错: {e}, 路径: {img_path}")
                return f'<span style="color:red;">[图片加载错误: {img_path}]</span>'
        
        # 使用正则表达式处理所有图片标记
        pattern = r'!\[(.*?)\]\((.*?)\)'
        processed_content = re.sub(pattern, embed_image, markdown_content)
        
        # 使用Python的markdown库进行转换
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

    # 从Markdown内容生成HTML导航栏和处理内容
    def generate_toc_and_content(markdown_content):
        """从Markdown内容生成HTML导航栏和处理内容"""
        # 提取所有标题
        headers = re.findall(r'^(#{1,3})\s+(.+)$', markdown_content, re.MULTILINE)
        
        if not headers:
            return "<div class='manual-nav'><p>目录加载中...</p></div>", markdown_content
        
        toc_html = "<div class='manual-nav'><ul>"
        
        # 为每个标题创建导航项
        for i, (level, title) in enumerate(headers):
            level_num = len(level)
            header_id = f"header-{i}"
            
            # 根据标题级别添加类
            css_class = ""
            if level_num == 2:
                css_class = "nav-h2"
            elif level_num == 3:
                css_class = "nav-h3"
            
            toc_html += f"<li><a href='#{header_id}' class='{css_class}'>{title}</a></li>"
        
        toc_html += "</ul></div>"
        
        # 为Markdown内容中的标题添加ID
        processed_content = markdown_content
        for i, (level, title) in enumerate(headers):
            header_id = f"header-{i}"
            header_pattern = f"{level} {title}"
            header_replacement = f"{level} <span id='{header_id}'></span>{title}"
            processed_content = processed_content.replace(header_pattern, header_replacement, 1)
        
        # 将处理后的Markdown转换为HTML
        html_content = markdown_to_html(processed_content)
        
        return toc_html, html_content

    with gr.Tab("Manual"):
        # 添加自定义CSS和JavaScript
        gr.HTML(custom_css + custom_js)
        
        with gr.Row():
           language = gr.Dropdown(choices=['English', 'Chinese'], value='English', label='Language', interactive=True)
        
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
        
        # 正确绑定语言切换事件
        language.change(
            fn=update_manual,
            inputs=[language],
            outputs=[training_md, prediction_md, evaluation_md, download_md, faq_md]
        )
    
    return {"training_md": training_md, "prediction_md": prediction_md, "evaluation_md": evaluation_md, "download_md": download_md, "faq_md": faq_md}

def update_manual(language):
    """更新手册内容"""
    training_content = load_manual_training(language)
    prediction_content = load_manual_prediction(language)
    evaluation_content = load_manual_evaluation(language)
    download_content = load_manual_download(language)
    faq_content = load_manual_faq(language)
    
    # 使用Python的markdown库将Markdown转换为HTML
    def markdown_to_html(markdown_content, base_path="src/web/manual"):
        """将Markdown内容转换为HTML，并将图片嵌入为base64编码"""
        # 处理图片路径，使用base64编码直接嵌入图片
        def embed_image(match):
            alt_text = match.group(1)
            img_path = match.group(2)
            
            # 检查路径是否为外部URL
            if img_path.startswith(('http://', 'https://')):
                return f'<img src="{img_path}" alt="{alt_text}" />'
            
            # 处理本地图片路径
            try:
                # 去掉开头的/以获取正确的路径
                if img_path.startswith('/'):
                    img_path = img_path[1:]
                
                # 获取绝对路径
                current_dir = os.path.dirname(os.path.abspath(__file__))
                project_root = os.path.dirname(os.path.dirname(current_dir))
                abs_img_path = os.path.join(project_root, img_path)
                
                # 读取图片并转换为base64
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
                    print(f"图片文件不存在: {abs_img_path}")
                    return f'<span style="color:red;">[图片不存在: {img_path}]</span>'
                
            except Exception as e:
                print(f"处理图片时出错: {e}, 路径: {img_path}")
                return f'<span style="color:red;">[图片加载错误: {img_path}]</span>'
        
        # 使用正则表达式处理所有图片标记
        pattern = r'!\[(.*?)\]\((.*?)\)'
        processed_content = re.sub(pattern, embed_image, markdown_content)
        
        # 使用Python的markdown库进行转换
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
    
    # 为每个内容生成导航栏和HTML内容
    def generate_toc_and_content(markdown_content):
        """从Markdown内容生成HTML导航栏和处理内容"""
        # 提取所有标题
        headers = re.findall(r'^(#{1,3})\s+(.+)$', markdown_content, re.MULTILINE)
        
        if not headers:
            return "<div class='manual-nav'><p>目录加载中...</p></div>", markdown_content
        
        toc_html = "<div class='manual-nav'><ul>"
        
        # 为每个标题创建导航项
        for i, (level, title) in enumerate(headers):
            level_num = len(level)
            header_id = f"header-{i}"
            
            # 根据标题级别添加类
            css_class = ""
            if level_num == 2:
                css_class = "nav-h2"
            elif level_num == 3:
                css_class = "nav-h3"
            
            toc_html += f"<li><a href='#{header_id}' class='{css_class}'>{title}</a></li>"
        
        toc_html += "</ul></div>"
        
        # 为Markdown内容中的标题添加ID
        processed_content = markdown_content
        for i, (level, title) in enumerate(headers):
            header_id = f"header-{i}"
            header_pattern = f"{level} {title}"
            header_replacement = f"{level} <span id='{header_id}'></span>{title}"
            processed_content = processed_content.replace(header_pattern, header_replacement, 1)
        
        # 将处理后的Markdown转换为HTML
        html_content = markdown_to_html(processed_content)
        
        return toc_html, html_content
    
    # 生成带导航栏的HTML
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