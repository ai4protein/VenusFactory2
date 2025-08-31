import os
from pathlib import Path

def load_css_file(css_filename: str) -> str:
    """
    Load CSS file content from the assets directory
    
    Args:
        css_filename: Name of the CSS file (e.g., 'eval_predict_ui.css')
        
    Returns:
        CSS content as string
    """
    # Get the current file's directory
    current_dir = Path(__file__).parent.parent
    
    # Construct path to assets directory
    assets_dir = current_dir / "assets"
    css_path = assets_dir / css_filename
    
    try:
        with open(css_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Warning: CSS file {css_filename} not found at {css_path}")
        return ""
    except Exception as e:
        print(f"Error loading CSS file {css_filename}: {e}")
        return ""

def get_css_style_tag(css_filename: str) -> str:
    """
    Get CSS content wrapped in a style tag
    
    Args:
        css_filename: Name of the CSS file
        
    Returns:
        CSS content wrapped in <style> tag
    """
    css_content = load_css_file(css_filename)
    if css_content:
        return f"<style>\n{css_content}\n</style>"
    return ""
