"""PI chat-mode nodes: direct conversational reply without tool execution."""
from __future__ import annotations

from langchain_core.runnables import RunnableConfig

from agent.graph.common.lang import _detect_ui_lang
from agent.graph.common.streaming import _stream_chain
from agent.graph.helpers.chat_history import _get_chat_history_messages
from agent.graph.state import AgentState


async def chat_start_node(state: AgentState, config: RunnableConfig):
    """Show 'PI is responding' for simple chat/greeting inputs."""
    history = list(state.get("history", []))
    ui_lang = state.get("ui_lang") or _detect_ui_lang(state["messages"][-1].content)
    history.append({
        "role": "assistant",
        "content": "🤔 思考中..." if ui_lang == "zh" else "🤔 Thinking...",
        "role_id": "principal_investigator",
    })
    return {"history": history, "ui_lang": ui_lang}


async def chat_node(state: AgentState, config: RunnableConfig):
    """Chat mode: use pi_chat_chain for direct responses (greetings, simple questions)."""
    chains = config.get("configurable", {}).get("chains", {})
    text = state["messages"][-1].content
    ui_lang = state.get("ui_lang") or _detect_ui_lang(text)
    history = list(state.get("history", []))

    if history and ("Thinking" in history[-1].get("content", "") or "思考中" in history[-1].get("content", "")):
        history.pop()

    try:
        chat_history = _get_chat_history_messages(chains, history, text)
        response = await _stream_chain(
            chains["pi_chat"],
            {"input": text, "chat_history": chat_history},
            role_id="principal_investigator",
        )
        history.append({"role": "assistant", "content": response, "role_id": "principal_investigator"})
    except Exception as e:
        err_msg = (
            f"抱歉，我遇到了错误：{str(e)}"
            if ui_lang == "zh"
            else f"I apologize, but I encountered an error: {str(e)}"
        )
        history.append({"role": "assistant", "content": err_msg, "role_id": "principal_investigator"})

    return {"history": history, "status": "completed"}
