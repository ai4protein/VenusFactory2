"""Lightweight Web session state; inference belongs to the execution worker."""
from datetime import datetime, timezone
import uuid

from mcp_tools.common.common_utils import get_web_v2_area_dir


def initialize_session_state():
    identity = str(uuid.uuid4())
    workspace = get_web_v2_area_dir("sessions", tool="chat", session_id=identity)
    workspace.mkdir(parents=True, exist_ok=True)
    return {
        "session_id": identity, "agent_session_dir": str(workspace),
        "runtime_kind": "native", "chat_mode": "science_agent", "status": "idle",
        "created_at": datetime.now(timezone.utc),
        "history": [], "conversation_log": [], "tool_executions": [],
        "dialogue_memory": [], "temp_files": [], "tool_cache": {},
    }


def ensure_runtime_state(state):
    if state.get("runtime_kind") == "native" or state.get("agent_execution"):
        return state
    # Only historical role-graph sessions require the old LLM and chain objects.
    from agent.chat_agent import ensure_runtime_state as restore_legacy
    return restore_legacy(state)
