"""Adaptive mode using the installed LangChain/LangGraph agent implementation.

No custom LLM/tool loop: create_agent owns execution. The scientific worker,
model client, operational contracts and evidence are shared with Scientist.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from contextlib import AsyncExitStack
from importlib.metadata import version

from fastmcp import Client
from fastmcp.client.transports import StdioTransport
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from .langgraph_tools import build_tools, mcp_command
from .scientist_runtime import ModelAudit
from .storage import atomic_json


async def run_adaptive(
    *,
    workspace,
    capability,
    python_root,
    output,
    prompt,
    model,
    base_url,
    api_key,
    allowed_tools,
    timeout_s=900,
    max_tool_calls=12,
    max_model_calls=24,
    phase="develop",
    request_timeout_s=330,
    max_output_tokens=12288,
):
    if phase not in {"develop", "reuse", "probe", "chat"}:
        raise ValueError("Unsupported lifecycle phase")
    started = time.monotonic()
    deadline = started + timeout_s
    audit = ModelAudit(output / "model_calls.jsonl", deadline, max_calls=max_model_calls)
    llm = ChatOpenAI(
        model=model,
        base_url=base_url,
        api_key=api_key,
        max_tokens=max_output_tokens,
        temperature=0.1,
        max_retries=0,
        timeout=request_timeout_s,
        callbacks=[audit],
    )
    result = {
        "status": "running",
        "mode": "science_agent",
        "engine": "langgraph",
        "phase": phase,
        "model": model,
        "session_id": "adaptive_" + uuid.uuid4().hex,
        "tool_calls": 0,
        "tool_errors": 0,
        "human_interventions": 0,
        "packages": {n: version(n) for n in ["langchain", "langgraph", "langchain-openai"]},
    }
    transport = StdioTransport(
        command="bwrap",
        keep_alive=False,
        args=mcp_command(workspace, capability, python_root),
        cwd=str(workspace),
        log_file=output / "mcp.log",
    )

    def event(value):
        with (output / "events.jsonl").open("a") as f:
            f.write(json.dumps(value, ensure_ascii=False) + "\n")

    # Empty trace is valid evidence of a conversational turn with no tool use.
    (output / "events.jsonl").touch(exist_ok=False)
    try:
        async with asyncio.timeout(timeout_s), AsyncExitStack() as stack:
            client = await stack.enter_async_context(Client(transport)) if allowed_tools else None
            tools = (
                await build_tools(
                    client,
                    allowed_tools=allowed_tools,
                    workspace=workspace,
                    capability=capability,
                    deadline=deadline,
                    max_tool_calls=max_tool_calls,
                    result=result,
                    event=event,
                    namespaced=True,
                    validate_results=phase not in {"probe", "chat"},
                )
                if client
                else []
            )
            graph = create_agent(
                llm,
                tools=tools,
                system_prompt=(
                    "You are an interactive protein-research assistant. Answer the current question or carry out the requested task. "
                    "A direct answer or a clarification question is valid. Choose scientific operations "
                    "adaptively using only the supplied tools. Empty discovery or experience is a valid cold start. "
                    "Use measured feedback; distinguish a submitted candidate from independently accepted evidence. "
                    "When tools are used, report actual operations, artifacts, results, failures and limitations. "
                    "Do not invent observations or claim scientific novelty merely from generating code."
                ),
            )
            state = await graph.ainvoke(
                {"messages": [HumanMessage(content=prompt)]}, config={"recursion_limit": 2 * max_tool_calls + 4}
            )
            messages = state.get("messages", [])
            final = messages[-1] if messages else None
            final_ok = bool(final and final.type == "ai" and final.content and not getattr(final, "tool_calls", None))
            if final_ok:
                atomic_json(output / "final_report.json", {"content": final.content})
            expected = {"develop": "candidate.json", "reuse": "predictions.json", "probe": "probe.json", "chat": None}[
                phase
            ]
            result["final_report_written"] = final_ok
            result["artifact_required"] = expected is not None
            result["output_artifact_written"] = bool(expected and (workspace / expected).is_file())
            result["status"] = (
                "completed" if final_ok and (expected is None or result["output_artifact_written"]) else "incomplete"
            )
    except TimeoutError:
        result["status"] = "timeout"
    except Exception as exc:
        result.update(status="runtime_error", error_type=type(exc).__name__)
    finally:
        result.update(
            wall_seconds=round(time.monotonic() - started, 3), llm_calls=audit.calls, usage_records=audit.usage
        )
        atomic_json(output / "run.json", result)
    return result
