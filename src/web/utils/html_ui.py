import os
import json
import gradio as gr
from datasets import load_dataset

def load_html_template(name: str, **kwargs) -> str:
    """Load and format an HTML fragment from assets/html_fragments/
    Args:
        name: name of the HTML fragment
        **kwargs: keyword arguments to format the HTML fragment
    Returns:
        formatted HTML fragment
    """
    path = os.path.join("src", "web", "assets", "html_fragments", name)
    with open(path, "r", encoding="utf-8") as f:
        html = f.read()
    if kwargs:
        return html.format(**kwargs)
    return html
