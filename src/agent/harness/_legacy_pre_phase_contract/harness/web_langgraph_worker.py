"""One Web turn in the native LangGraph agent; stdio only transports UI events.

The native create_agent owns model/tool selection. A process boundary lets the
Web host stop a turn and its local tool subprocesses without abandoning threads.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import sys


def emit(value):
    print(json.dumps(value, ensure_ascii=False), file=sys.__stdout__, flush=True)


def answer_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")
    return ""


async def run(request, *, tools=None, model=None):
    from langchain.agents import create_agent
    from langchain.agents.middleware import wrap_tool_call
    from langchain_core.messages import HumanMessage, messages_from_dict, messages_to_dict
    from langchain_openai import ChatOpenAI

    from agent.chat_agent import _filter_agent_tools_for_runtime_mode, get_tools
    from agent.chat_agent_utils import _tool_output_indicates_failure
    from agent.graph.helpers.tool_io import _normalize_output_paths, _sanitize_tool_invoke_input
    from tools.runtime_tool_policy import assert_tool_allowed

    original = tools if tools is not None else _filter_agent_tools_for_runtime_mode(get_tools())[0]
    mapping = {"mcp__venusfactory__" + t.name: t for t in original}
    if len(mapping) != len(original):
        raise ValueError("Duplicate scientific tool identity")
    advertised = [t.model_copy(update={"name": name}) for name, t in mapping.items()]
    emit({"kind": "catalog", "tools": sorted(mapping)})
    if request.get("handshake"):
        if (await asyncio.to_thread(sys.stdin.readline)).strip() != "ready":
            raise RuntimeError("Host did not admit the scientific catalog")
    calls = 0

    @wrap_tool_call
    async def observed_tool(call, handler):
        nonlocal calls
        tool_call = call.tool_call
        name, cid = tool_call["name"], tool_call["id"]
        if name not in mapping or calls >= request["budget"]["tool_calls"]:
            raise RuntimeError("Scientific tool admission or budget exceeded")
        calls += 1
        tool = mapping[name]
        assert_tool_allowed(tool.name)
        args = _sanitize_tool_invoke_input(tool.name, tool, tool_call["args"], request["workspace"], {})
        args = _normalize_output_paths(tool.name, tool, args, request["workspace"])
        emit({"kind": "tool_call_start", "tool_call_id": cid, "tool_name": name, "args": args})
        try:
            result = await handler(call.override(tool_call={**tool_call, "args": args}))
            content = result.content
            failed, _ = _tool_output_indicates_failure(content)
            emit(
                {
                    "kind": "tool_result",
                    "tool_call_id": cid,
                    "is_error": failed or getattr(result, "status", "") == "error",
                    "output": content,
                }
            )
            return result
        except Exception as exc:
            emit(
                {
                    "kind": "tool_result",
                    "tool_call_id": cid,
                    "is_error": True,
                    "output": {"error_type": type(exc).__name__},
                }
            )
            raise

    llm = model or ChatOpenAI(
        model=request["model"],
        base_url=request["base_url"],
        api_key=request["token"],
        max_tokens=request["budget"]["output_tokens"],
        temperature=request["budget"]["temperature"],
        timeout=request["budget"]["request_seconds"] + 30,
        max_retries=0,
        model_kwargs={"parallel_tool_calls": False},
    )
    prompt = (
        "You are VenusFactory's conversational protein-science agent. Answer ordinary questions directly. "
        "Freely choose the actual registered scientific tools when computation or evidence is needed. "
        "Ask the user for missing input when necessary; do not invent sequences, files, results or tool names. "
        "Use the registered skill-reading tool when its guidance helps; a skill is not mandatory for every turn. "
        "You are not the four-role Scientist goal workflow. Stop when the current request is answered or needs user input. "
        "Use the supplied session workspace for generated files. Distinguish predicted properties from measured biology. "
        + ("Respond in Chinese. " if request.get("lang") == "zh" else "Respond in English. ")
        + "Session workspace: "
        + request["workspace"]
    )
    messages = messages_from_dict(request.get("messages", [])) + [HumanMessage(content=request["text"])]
    agent = create_agent(llm, tools=advertised, middleware=[observed_tool], system_prompt=prompt)
    final = None
    async for mode, data in agent.astream(
        {"messages": messages}, stream_mode=["messages", "values"], config={"recursion_limit": 80}
    ):
        if mode == "values":
            final = data["messages"]
        else:
            message, metadata = data
            if metadata.get("langgraph_node") != "model":
                continue
            content = message.content
            if isinstance(content, list):
                content = "".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")
            if isinstance(content, str) and content:
                emit({"kind": "text", "text": content})
    if not final or final[-1].type != "ai" or not answer_text(final[-1].content).strip() or final[-1].tool_calls:
        raise ValueError("Agent did not finish with an answer or clarification")
    emit(
        {
            "kind": "completed",
            "messages": messages_to_dict(final),
            "text": answer_text(final[-1].content),
            "tool_calls": calls,
        }
    )


def main():
    request = json.loads(sys.stdin.readline())
    with contextlib.redirect_stdout(sys.stderr):
        try:
            asyncio.run(run(request))
        except Exception as exc:
            emit({"kind": "error", "error_type": type(exc).__name__})
            raise SystemExit(1) from None


if __name__ == "__main__":
    main()
