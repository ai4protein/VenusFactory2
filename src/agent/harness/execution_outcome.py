"""Interpret process outcomes without mistaking tool transport success for execution."""
import json

EXECUTION_TOOLS = {"run_session_python", "run_session_shell", "run_research_python"}


def execution_error(value):
    if isinstance(value, str):
        if len(value) > 200000:
            return None
        try:
            value, _ = json.JSONDecoder().raw_decode(value.lstrip())
        except (TypeError, ValueError):
            return None
    if not isinstance(value, dict):
        return None
    code = value.get("exit_code")
    failed = type(code) is int and code != 0
    failed = failed or value.get("status") in {"failed", "timeout", "cancelled"}
    if not failed:
        return None
    detail = value.get("stderr_tail") or value.get("stderr") or value.get("error") or ""
    detail = detail[-1800:] if isinstance(detail, str) else ""
    label = f"Execution exited with code {code}" if type(code) is int else "Execution "+value["status"]
    return label+(": "+detail if detail.strip() else "")
