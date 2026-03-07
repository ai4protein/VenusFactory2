import json
from typing import Any, Dict, Optional

def query_success_response(
    content: str,
    query_time_ms: int = 0,
    source: str = "unknown",
) -> str:
    """Build JSON for query success: status, content, execution_context."""
    out: Dict[str, Any] = {
        "status": "success",
        "content": content,
        "execution_context": {"query_time_ms": query_time_ms, "source": source},
    }
    return json.dumps(out, ensure_ascii=False)

def error_response(error_type: str, message: str, suggestion: Optional[str] = None) -> str:
    """Build JSON for error."""
    out: Dict[str, Any] = {
        "status": "error",
        "error": {"type": error_type, "message": message},
        "file_info": None,
    }
    if suggestion:
        out["error"]["suggestion"] = suggestion
    return json.dumps(out, ensure_ascii=False)
