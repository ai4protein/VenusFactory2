"""Helper functions for chat tab functionality."""

import os
import html as _html
import smtplib
import gradio as gr
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from .common_utils import get_save_path


def send_feedback_email(feedback_text: str) -> str:
    """Send feedback email via SMTP"""
    if not feedback_text.strip():
        return "❌ Please enter your feedback before submitting."
    
    sender_email = "zmm.zlr@qq.com"
    receiver_email = "zlr_zmm@163.com"
    subject = "VenusFactory User Feedback"
    password = os.getenv("EMAIL_PASSWORD")

    server = None
    try:
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = receiver_email
        msg['Subject'] = subject

        body = f"""
VenusFactory User Feedback
--------------------------
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Feedback:
{feedback_text}
--------------------------
        """
        msg.attach(MIMEText(body, 'plain', 'utf-8'))

        smtp_server = "smtp.qq.com"
        smtp_port = 465
        
        server = smtplib.SMTP_SSL(smtp_server, smtp_port)
        server.login(sender_email, password)
        server.send_message(msg)
        
        return "✅ Thank you for your feedback! Email sent successfully."

    except Exception as e:
        return f"❌ Failed to send feedback: {str(e)}"

    finally:
        if server:
            try:
                server.quit()
            except Exception:
                pass


def handle_feedback_submit(feedback_text: str) -> tuple:
    """Handle feedback submission and return status updates."""
    result = send_feedback_email(feedback_text)
    return "", result, gr.update(visible=True)


def make_chat_html(path: Path, history_list: List[Dict[str, Any]]) -> None:
    """Create HTML file from chat history."""
    css = """
    body { font-family: Arial, sans-serif; background: #f6f8fb; padding: 20px; }
    .chat-container { max-width: 900px; margin: 0 auto; }
    .message { display: flex; margin: 8px 0; }
    .message.user { justify-content: flex-end; }
    .bubble { max-width: 75%; padding: 12px 16px; border-radius: 12px; white-space: pre-wrap; }
    .bubble.user { background: #0b93f6; color: #fff; border-bottom-right-radius: 4px; }
    .bubble.assistant { background: #eef1f7; color: #111; border-bottom-left-radius: 4px; }
    .meta { font-size: 12px; color: #666; margin: 4px 8px; }
    """
    with open(path, "w", encoding="utf-8") as f:
        f.write("<!doctype html><html><head><meta charset='utf-8'><title>Chat Export</title>")
        f.write(f"<style>{css}</style></head><body>")
        f.write("<div class='chat-container'>")
        f.write("<h2>VenusFactory Chat Export</h2>\n")
        for msg in history_list:
            role = str(msg.get("role", "")).lower()
            content = str(msg.get("content", ""))
            ts = ""
            if isinstance(msg.get("timestamp"), str):
                ts = msg.get("timestamp")
            elif isinstance(msg.get("timestamp"), datetime):
                ts = msg.get("timestamp").strftime("%Y-%m-%d %H:%M:%S")
            content_esc = _html.escape(content)
            cls = "assistant" if role != "user" else "user"
            f.write(f"<div class='message {cls}'>")
            f.write(f"<div class='bubble {cls}'>{content_esc}</div>")
            f.write(f"</div>")
            if ts:
                text_align = 'right' if role == 'user' else 'left'
                f.write(f"<div class='meta' style='text-align: {text_align}'>{_html.escape(ts)}</div>")
        f.write("</div></body></html>")


def export_chat_history_html(session_state_value: Dict[str, Any]) -> tuple:
    """Create (or overwrite) session-scoped HTML file on server and return status update and download update."""
    try:
        ss = session_state_value
        history_list = ss.get('history', [])
        out_dir = get_save_path("VenusAgent", "Expert_Chat")
        session_id = ss.get('session_id') or "anon"
        filename = out_dir / f"chat_history_{session_id}.html"
        make_chat_html(filename, history_list)
        # return two outputs: (markdown update, downloadbutton update)
        return (
            gr.update(visible=True, value=f"Exported HTML to: {str(filename)}"),
            gr.update(visible=True, value=str(filename))
        )
    except Exception as e:
        return (
            gr.update(visible=True, value=f"Export failed: {e}"),
            gr.update(visible=False, value="")
        )


def save_chat_history_to_server(session_state_value: Dict[str, Any]) -> gr.update:
    """Return the path to the session-scoped HTML file for download; create it if missing."""
    try:
        ss = session_state_value
        out_dir = get_save_path("VenusAgent", "Expert_Chat")
        session_id = ss.get('session_id') or "anon"
        filename = out_dir / f"chat_history_{session_id}.html"
        # create file if not exists
        if not filename.exists():
            history_list = ss.get('history', [])
            make_chat_html(filename, history_list)
        # return update to trigger DownloadButton
        return gr.update(visible=True, value=str(filename))
    except Exception as e:
        return gr.update(visible=False, value="")

