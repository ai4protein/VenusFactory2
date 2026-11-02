"""The same bounded, isolated MCP interface for both LangGraph modes."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from functools import partial
from pathlib import Path

from langchain_core.tools import StructuredTool
from pydantic import Field, create_model

from .reproduction_tool_contracts import SUPPORTED, validate_reproduction_result
from .tool_contracts import validate_generated_result


def mcp_command(workspace, capability, python_root):
    cmd = ["--unshare-all", "--die-with-parent", "--new-session", "--cap-drop", "ALL"]
    for name in ["/usr", "/bin", "/lib", "/lib64"]:
        if Path(name).exists():
            cmd += ["--ro-bind", name, name]
    cmd += [
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--tmpfs",
        "/tmp",
        "--tmpfs",
        "/home",
        "--dir",
        "/opt",
        "--ro-bind",
        str(python_root.resolve()),
        "/opt/venus-runtime",
        "--bind",
        str(workspace.resolve()),
        "/workspace",
        "--ro-bind",
        str(capability.resolve()),
        "/capabilities",
        "--chdir",
        "/workspace",
        "--clearenv",
        "--setenv",
        "HOME",
        "/tmp",
        "--setenv",
        "PATH",
        "/opt/venus-runtime/bin:/usr/bin:/bin",
        "--setenv",
        "VENUS_RSI_WORKSPACE",
        "/workspace",
        "--setenv",
        "OPENBLAS_NUM_THREADS",
        "1",
        "--setenv",
        "OMP_NUM_THREADS",
        "1",
        "--",
        "/opt/venus-runtime/bin/python",
        "-u",
        "/capabilities/generated_tools.py",
    ]
    return cmd


def argument_model(name, schema):
    """Typed tool arguments let LangChain remove its ambient context fields."""
    fields = {}
    for key, definition in schema.get("properties", {}).items():
        if definition.get("type") != "string":
            raise ValueError("generated-model tool expects a string argument schema")
        default = ... if key in schema.get("required", []) else definition.get("default", "")
        fields[key] = (str, Field(default=default, description=definition.get("description", "")))
    return create_model(name + "Arguments", **fields)


async def build_tools(
    client,
    *,
    allowed_tools,
    workspace,
    capability,
    deadline,
    max_tool_calls,
    result,
    event,
    namespaced=False,
    validate_results=True,
):
    schemas = {t.name: t for t in await client.list_tools()}
    if not set(allowed_tools) <= set(schemas):
        raise ValueError("Required research tools missing")
    lock = asyncio.Lock()
    loop = asyncio.get_running_loop()

    def wrapper(name):
        validator = (
            partial(validate_generated_result, name, workspace=workspace, capability=capability)
            if validate_results
            else partial(validate_reproduction_result, name, workspace=workspace, capability=capability)
            if name in SUPPORTED
            else None
        )

        async def call(**kwargs):
            async with lock:
                if result["tool_calls"] >= max_tool_calls:
                    raise RuntimeError("Tool budget exhausted")
                result["tool_calls"] += 1
                cid = uuid.uuid4().hex
                event(
                    {
                        "kind": "tool_call_start",
                        "tool_call_id": cid,
                        "tool_name": "mcp__venusfactory__" + name,
                        "args": kwargs,
                    }
                )
                try:
                    answer = await client.call_tool(name, kwargs, timeout=max(1, deadline - time.monotonic()))
                    if getattr(answer, "is_error", False):
                        raise ValueError("MCP operation reported failure")
                    content = "\n".join(x.text for x in answer.content if hasattr(x, "text"))
                    decoded = json.loads(content)
                    if not isinstance(decoded, dict):
                        raise ValueError("Scientific tool must return an object")
                    if validator:
                        validator(decoded)
                    # Emit exactly one terminal event, after parsing and validation.
                    event({"kind": "tool_result", "tool_call_id": cid, "is_error": False, "output": content})
                    return json.dumps({"success": True, **decoded}, allow_nan=False)
                except Exception as exc:
                    result["tool_errors"] += 1
                    event(
                        {
                            "kind": "tool_result",
                            "tool_call_id": cid,
                            "is_error": True,
                            "output": type(exc).__name__ + ": " + str(exc)[:3000],
                        }
                    )
                    raise

        def sync_call(**kwargs):
            return asyncio.run_coroutine_threadsafe(call(**kwargs), loop).result(
                timeout=max(1, deadline - time.monotonic())
            )

        metadata = {
            "deferred_arguments": name
            in {
                "evaluate_model_code",
                "evaluate_library_model_code",
                "submit_model_code",
                "predict_with_generated_model",
                "submit_reproduction",
                "inspect_reproduction_job",
                "submit_evidence_review",
                "evaluate_paper_step",
                "submit_paper_step",
                "execute_registered_paper_step",
            }
        }
        if validator:
            metadata["result_validator"] = validator
        return StructuredTool(
            name=("mcp__venusfactory__" if namespaced else "") + name,
            description=schemas[name].description or name,
            args_schema=argument_model(name, schemas[name].inputSchema),
            func=sync_call,
            coroutine=call,
            metadata=metadata,
        )

    return [wrapper(name) for name in sorted(allowed_tools)]
