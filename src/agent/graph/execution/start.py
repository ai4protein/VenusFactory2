"""MLS execution preamble: emit 'executing Step N' UI line before tool runs."""
from __future__ import annotations

from langchain_core.runnables import RunnableConfig

from agent.graph.common.lang import _detect_ui_lang
from agent.graph.common.ui_text import _ui_text
from agent.graph.helpers.plan_helpers import _normalize_step_number
from agent.graph.state import AgentState


def _summarize_tool_inputs(tool_name: str, inputs: dict) -> str:
    """Build a short, user-readable summary of the most important parameters
    for the tool being invoked. Designed for the per-step UI announcement so
    the user sees WHAT is happening (which file, which model, which residue
    range) instead of just an opaque "executing step N…" message.
    """
    if not isinstance(inputs, dict):
        return ""

    KEY_PRIORITY = [
        # Identifiers / inputs
        "pdb_id", "uniprot_id", "ec_number", "gene_name", "fasta_file",
        "pdb_file", "pdb_path", "structure_file", "csv_file", "config_path",
        # Tool-specific knobs the user cares about
        "model_name", "task", "num_sequences", "temperatures",
        "designed_chains", "fixed_chains", "skill_id",
        # Outputs (less critical but useful)
        "out_dir", "output_dir", "out_path",
    ]
    parts = []
    seen = set()
    for k in KEY_PRIORITY:
        if k in inputs and inputs[k] is not None and k not in seen:
            v = inputs[k]
            seen.add(k)
            if isinstance(v, str):
                # Shorten very long paths to ~/sessions/.../file
                if "temp_outputs/web_v2/sessions/" in v:
                    try:
                        rel = v.split("temp_outputs/web_v2/sessions/", 1)[1]
                        bits = rel.split("/", 4)
                        if len(bits) >= 5:
                            v = f"~/sessions/{bits[0][:8]}/{bits[4]}"
                    except Exception:
                        pass
                if len(v) > 60:
                    v = v[:30] + "…" + v[-25:]
                parts.append(f"`{k}`=`{v}`")
            elif isinstance(v, (list, tuple)):
                if len(v) <= 4:
                    parts.append(f"`{k}`={list(v)}")
                else:
                    parts.append(f"`{k}`=[{len(v)} items]")
            elif isinstance(v, dict):
                parts.append(f"`{k}`={{{len(v)} keys}}")
            else:
                parts.append(f"`{k}`={v}")
        if len(parts) >= 4:
            break
    return ", ".join(parts)


async def execute_start_node(state: AgentState, config: RunnableConfig):
    """Show 'MLS is executing step N' so UI updates before tool runs.

    Now augmented with a one-line tool + key-input summary so the user
    sees WHICH tool is being invoked and with what payload — replaces
    the opaque "executing step N…" with a substantive progress signal.
    """
    plan = state.get("plan", [])
    idx = state.get("current_step_index", 0)
    history = list(state.get("history", []))
    ui_lang = state.get("ui_lang") or _detect_ui_lang(state["messages"][-1].content)
    if idx >= len(plan):
        return {}
    step = plan[idx]
    step_num = _normalize_step_number(step.get("step"), idx + 1)
    task_desc = step.get("task_description", "…")
    tool_name = step.get("tool_name", "")
    tool_input = step.get("tool_input", {}) or {}
    inputs_summary = _summarize_tool_inputs(tool_name, tool_input) if isinstance(tool_input, dict) else ""

    base_msg = _ui_text(ui_lang, "executing_step", step_num=step_num, task_desc=task_desc)
    if tool_name:
        tool_line = (
            f"\n\n→ 调用工具 `{tool_name}`" if ui_lang == "zh"
            else f"\n\n→ Tool `{tool_name}`"
        )
        if inputs_summary:
            tool_line += f" ({inputs_summary})"
        content = base_msg + tool_line
    else:
        content = base_msg

    history.append({
        "role": "assistant",
        "content": content,
        "role_id": "machine_learning_specialist",
    })
    return {"history": history}
