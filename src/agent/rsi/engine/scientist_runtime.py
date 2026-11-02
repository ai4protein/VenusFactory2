"""Research adapter for the actual four-role product graph.

The same immutable MCP source bundle is used by both modes. This adapter never
calls initialize_session_state(), which would resolve a product-default model.
It binds every chain explicitly to the supplied research model.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
import uuid

from fastmcp import Client
from fastmcp.client.transports import StdioTransport
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from .langgraph_tools import argument_model as argument_model  # compatibility re-export
from .langgraph_tools import build_tools, mcp_command
from .scientific_episode import artifact_written, completion_feedback, episode_contract, successful_tools
from .storage import atomic_json


class ModelAudit(BaseCallbackHandler):
    raise_error = True

    def __init__(self, path, deadline, max_calls=24):
        self.path = path
        self.deadline = deadline
        self.max_calls = max_calls
        self.calls = 0
        self.lock = threading.Lock()
        self.usage = []

    def on_chat_model_start(self, serialized, messages, **kwargs):
        with self.lock:
            if time.monotonic() > self.deadline or self.calls >= self.max_calls:
                raise RuntimeError("research model budget exhausted")
            self.calls += 1
            with self.path.open("a") as f:
                f.write(
                    json.dumps(
                        {
                            "event": "llm_start",
                            "call": self.calls,
                            "run_id": str(kwargs.get("run_id")),
                            "messages": [[m.type for m in group] for group in messages],
                        }
                    )
                    + "\n"
                )

    def on_llm_end(self, response, **kwargs):
        usage = (response.llm_output or {}).get("token_usage")
        with self.lock:
            if usage:
                self.usage.append(usage)
            with self.path.open("a") as f:
                summaries = []
                for group in getattr(response, "generations", []):
                    for generation in group:
                        message = getattr(generation, "message", None)
                        reason = (getattr(generation, "generation_info", None) or {}).get("finish_reason")
                        if reason is None:
                            reason = (getattr(message, "response_metadata", None) or {}).get("finish_reason")
                        summaries.append(
                            {
                                "finish_reason": reason
                                if isinstance(reason, str)
                                and reason in {"stop", "length", "tool_calls", "function_call", "content_filter"}
                                else "unknown",
                                "content_characters": len(str(getattr(message, "content", ""))),
                                "valid_tool_calls": len(getattr(message, "tool_calls", []) or []),
                                "invalid_tool_calls": len(getattr(message, "invalid_tool_calls", []) or []),
                            }
                        )
                f.write(
                    json.dumps(
                        {
                            "event": "llm_end",
                            "run_id": str(kwargs.get("run_id")),
                            "usage": usage,
                            "generations": summaries,
                        }
                    )
                    + "\n"
                )


async def run_scientist(
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
    timeout_s=420,
    max_tool_calls=12,
    research_policy="role_research",
    phase="develop",
    request_timeout_s=90,
    max_output_tokens=4096,
    max_model_calls=24,
):
    from agent.chat_agent import ProteinContextManager, _build_chains_for_llm
    from agent.graph.compile import create_agent_graph

    if research_policy not in {"role_research", "task_context_only"}:
        raise ValueError("unsupported research scope")
    contract = episode_contract(capability, phase, allowed_tools)
    expected_artifact = contract.artifact_file
    if contract.required_tools:
        prompt += (
            "\n\nHost-authored completion obligations: every tool in "
            + json.dumps(list(contract.required_tools))
            + " must return a successful result before this episode ends. Preserve the requested order; "
            "submission requires a subsequent job-status inspection. A saved artifact alone is insufficient."
        )
    if phase == "chat" and ((workspace / expected_artifact).exists() or (workspace / expected_artifact).is_symlink()):
        raise ValueError("Scientific episode requires a fresh output artifact")
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
        "mode": "science_expert",
        "phase": phase,
        "episode_kind": contract.kind,
        "context_tool": contract.context_tool,
        "expected_artifact": expected_artifact,
        "engine": "graph",
        "model": model,
        "tool_calls": 0,
        "tool_errors": 0,
        "session_id": "scientist_" + uuid.uuid4().hex,
        "nodes": [],
        "scripted_feedback": [],
        "human_interventions": 0,
        "research_policy": research_policy,
        "report_policy": "evidence_summary",
        "comparison_status": "bounded scientific episode; whole-goal outcome is assessed by the outer controller",
        "scientific_workflow_required": True,
    }
    transport = StdioTransport(
        command="bwrap",
        keep_alive=False,
        args=mcp_command(workspace, capability, python_root),
        cwd=str(workspace),
        log_file=output / "mcp.log",
    )
    events = output / "events.jsonl"
    observed_events = []
    completion_repairs = []

    def event(value):
        observed_events.append(value)
        with events.open("a") as f:
            f.write(json.dumps(value, ensure_ascii=False) + "\n")

    state = {}
    try:
        async with asyncio.timeout(timeout_s), Client(transport) as client:
            tools = await build_tools(
                client,
                allowed_tools=allowed_tools,
                workspace=workspace,
                capability=capability,
                deadline=deadline,
                max_tool_calls=max_tool_calls,
                result=result,
                event=event,
                validate_results=phase != "chat",
            )
            pi_tools = [
                t
                for t in tools
                if t.name
                in {
                    contract.context_tool,
                    "discover_generated_models",
                    "read_model_experience",
                    "read_model_library_experience",
                }
            ]
            chains = _build_chains_for_llm(llm, tools, pi_tools)
            chains.update(
                all_tools=tools,
                llm=llm,
                disabled_tool_names=[],
                skills_metadata="No standalone skills are available in this bounded task. Use only the supplied domain tools; do not add unsupported plotting or code-execution steps.",
                available_tools_list=", ".join(t.name for t in tools),
                session_id=result["session_id"],
                agent_session_dir=str(workspace),
                dialogue_memory=[],
                tool_cache={},
            )
            # Trusted adapter setting, never inferred from a user/model string.
            chains["scientific_workflow_required"] = True
            chains["report_policy"] = "evidence_summary"
            chains["finalizer"] = (
                ChatPromptTemplate.from_messages(
                    [
                        (
                            "system",
                            "You are the Scientific Critic. Audit the actual execution evidence in at most 350 words. Report the task, operations, artifacts, measured feedback, failures and limits. Do not invent citations, biological background, independent validation or success. Short mutable-site strings represent selected positions within proteins, not whole proteins. Distinguish candidate submission from controller acceptance. This is an evidence report, not a manuscript.",
                        ),
                        ("human", "{full_run_record}"),
                    ]
                )
                | llm
                | StrOutputParser()
            )
            inspected_task = None

            async def restricted_research(query):
                nonlocal inspected_task
                if inspected_task is None:
                    tool = next(t for t in tools if t.name == contract.context_tool)
                    inspected_task = await tool.ainvoke({})
                # A data card is explicitly not a retrieved paper or a new fact.
                return (
                    [
                        {
                            "source": "permitted_task_metadata",
                            "title": "Host-bound scientific task context (not a literature search)",
                            "content": inspected_task,
                        }
                    ],
                    [],
                )

            chains["research_search_runner"] = restricted_research

            async def materialize_arguments(ctx):
                payload = {
                    "task": prompt,
                    "step": ctx.task_desc,
                    "tool": ctx.tool_name,
                    "provided_arguments": ctx.merged_tool_input,
                    "previous_tool_results": ctx.protein_ctx.get_tool_records(limit=12),
                }
                author = llm.with_structured_output(ctx.tool.args_schema, method="function_calling", include_raw=True)
                messages = [
                    SystemMessage(
                        content="You are the Machine Learning Specialist. Generate complete executable arguments for this scientific tool from the task and actual prior results. For reproduction tasks, preserve the exact admitted plugin and parameters, use a stable idempotency key, copy the returned job identity, and copy reviewed claims with every field and type. Never treat queued submission as completed computation. For source_code, author full deterministic fit/predict Python, with encoding, fitting and JSON state. Use no placeholders. For source_sha256 or model_id, copy an actual successfully tested hash or discovered model identity from prior tool feedback; never invent it. Independent assessment labels are unavailable."
                    ),
                    HumanMessage(content=json.dumps(payload, default=str)),
                ]
                for attempt in range(2):
                    response = await author.ainvoke(messages)
                    value = response.get("parsed")
                    raw = response.get("raw")
                    with (output / "argument_generation.jsonl").open("a") as f:
                        f.write(
                            json.dumps(
                                {
                                    "tool": ctx.tool_name,
                                    "attempt": attempt + 1,
                                    "parsed": value is not None,
                                    "finish_reason": getattr(raw, "response_metadata", {}).get("finish_reason"),
                                    "response_content": getattr(raw, "content", ""),
                                    "parsing_error": str(response.get("parsing_error") or "")[:1000],
                                }
                            )
                            + "\n"
                        )
                    if value is not None:
                        return value.model_dump()
                    messages.append(
                        HumanMessage(
                            content="Your last response did not contain valid structured arguments. Return the required function call with all required fields, using actual prior results for identities, parameters or evidence. Include full source_code only when required by this tool schema. This is the final format-repair attempt; do not give prose instead."
                        )
                    )
                raise ValueError("model returned no valid structured tool arguments after two recorded attempts")

            chains["materialize_tool_arguments"] = materialize_arguments
            state = {
                "messages": [HumanMessage(content=prompt)],
                "protein_context": ProteinContextManager(),
                "session_id": result["session_id"],
                "agent_session_dir": str(workspace),
                "history": [],
                "conversation_log": [],
                "dialogue_memory": [],
                "tool_executions": [],
                "tool_cache": {},
                "pi_report": "",
                "pi_suggest_steps": "",
                "plan": [],
                "current_step_index": 0,
                "step_results": {},
                "waiting_for": None,
                "auto_execute": True,
                "review_sub_reports": False,
                "full_manuscript": False,
                "ui_lang": "en",
                "user_lang": "en",
                "execution_failed": False,
            }
            graph = create_agent_graph()
            config = {"configurable": {"chains": chains}, "recursion_limit": 60}
            for _resume in range(8):
                async for update in graph.astream(state, config=config, stream_mode="updates"):
                    for node, delta in update.items():
                        if isinstance(delta, dict):
                            state.update(delta)
                        result["nodes"].append(node)
                        atomic_json(output / "graph_history.json", state.get("history", []))
                        atomic_json(
                            output / "graph_progress.json",
                            {
                                "nodes": result["nodes"],
                                "status": state.get("status"),
                                "waiting_for": state.get("waiting_for"),
                            },
                        )
                waiting = state.get("waiting_for")
                if waiting == "clarification":
                    feedback = (
                        "The full computational task and data access are already supplied. Skip literature subreports and proceed to computational planning; preserve PI analysis and use task inspection during execution."
                        if research_policy == "task_context_only"
                        else "Use only the supplied task description and tools. No additional information is available. Continue research using task inspection; do not browse external literature."
                    )
                    state["messages"] = state["messages"] + [
                        HumanMessage(content=prompt + "\n\nPredeclared clarification policy:\n" + feedback)
                    ]
                    state["waiting_for"] = (
                        "skip_to_plan" if research_policy == "task_context_only" else "clarification_answered"
                    )
                elif (
                    waiting in {"plan_confirmation", "plan_confirm", "plan"}
                    or state.get("status") == "waiting_for_plan_confirmation"
                ):
                    feedback = "Execute the proposed plan within the fixed research budget."
                    state["waiting_for"] = "plan_confirmed"
                elif state.get("status") == "waiting_for_step_review":
                    feedback = "Continue within the original task and budget."
                    state["waiting_for"] = "step_continue"
                elif (
                    state.get("status") == "waiting_for_iteration"
                    and not state.get("execution_failed")
                    and len(completion_repairs) < contract.completion_repair_limit
                    and result["tool_calls"] < max_tool_calls
                    and audit.calls < max_model_calls
                    and time.monotonic() < deadline
                    and (feedback := completion_feedback(contract, observed_events)) is not None
                ):
                    completion_repairs.append(
                        {"text": feedback, "after_tool_calls": result["tool_calls"], "after_model_calls": audit.calls}
                    )
                    atomic_json(output / "completion_repairs.json", completion_repairs)
                    state["messages"] = state["messages"] + [HumanMessage(content=prompt + "\n\n" + feedback)]
                    state["waiting_for"] = "skip_to_plan"
                    state["finalizer_succeeded"] = False
                else:
                    break
                result["scripted_feedback"].append({"waiting_for": waiting, "text": feedback})
            result["four_role_path_completed"] = {
                "research_plan_node",
                "plan_node",
                "execute_node",
                "finalize_node",
            }.issubset(result["nodes"])
            result["status"] = (
                "completed"
                if artifact_written(workspace, contract)
                and not (set(contract.required_tools) - successful_tools(observed_events))
                and result["four_role_path_completed"]
                and state.get("finalizer_succeeded")
                else "incomplete"
            )
    except TimeoutError:
        result["status"] = "timeout"
    except Exception as exc:
        result.update(status="runtime_error", error_type=type(exc).__name__, error=str(exc)[:1000])
    finally:
        # Preserve completed role evidence even when a later role exceeds budget.
        result["completion_contract"] = {
            "required_tools": list(contract.required_tools),
            "missing_tools": sorted(set(contract.required_tools) - successful_tools(observed_events)),
            "repair_limit": contract.completion_repair_limit,
            "repairs": completion_repairs,
        }
        result["graph_status"] = state.get("status")
        result["role_ids"] = sorted({x.get("role_id") for x in state.get("history", []) if x.get("role_id")})
        result["candidate_written"] = (workspace / "candidate.json").exists()
        result["output_artifact_written"] = artifact_written(workspace, contract)
        result["finalizer_succeeded"] = state.get("finalizer_succeeded", False)
        result["four_role_path_completed"] = {
            "research_plan_node",
            "plan_node",
            "execute_node",
            "finalize_node",
        }.issubset(result["nodes"])
        atomic_json(output / "graph_history.json", state.get("history", []))
        result.update(
            wall_seconds=round(time.monotonic() - started, 3), llm_calls=audit.calls, usage_records=audit.usage
        )
        atomic_json(output / "run.json", result)
    return result
