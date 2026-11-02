from pathlib import Path
def load_html_template(name: str, **kwargs) -> str:
    """Load and format an HTML fragment from assets/html_fragments/
    Args:
        name: name of the HTML fragment
        **kwargs: keyword arguments to format the HTML fragment
    Returns:
        formatted HTML fragment
    """
    path = Path(__file__).parent / "report_fragments" / name
    with open(path, "r", encoding="utf-8") as f:
        html = f.read()
    if kwargs:
        return html.format(**kwargs)
    return html

def create_progress_html(message: str = "Processing...") -> str:
    """Create a breathing progress indicator HTML."""
    return f"""
    <div style="text-align: center; padding: 20px;">
        <div style="display: inline-block;">
            <div style="width: 60px; height: 60px; border: 4px solid #f3f3f3; border-top: 4px solid #3498db; 
                        border-radius: 50%; animation: spin 1s linear infinite;"></div>
            <p style="margin-top: 10px; color: #666; font-weight: 500;">{message}</p>
        </div>
        <style>
            @keyframes spin {{
                0% {{ transform: rotate(0deg); }}
                100% {{ transform: rotate(360deg); }}
            }}
            @keyframes breathe {{
                0%, 100% {{ opacity: 0.6; }}
                50% {{ opacity: 1; }}
            }}
        </style>
    </div>
    """

def create_status_html(status: str, color: str = "#666") -> str:
    """Create a status indicator HTML."""
    return f"<div style='text-align: center; padding: 10px;'><span style='color: {color}; font-weight: 500;'>{status}</span></div>"
