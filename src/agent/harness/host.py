"""Own the model gateway, immutable run binding and adapter lifecycle."""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

from .contracts import digest, file_hash
from .events import audit_requests, audit_trace
from .registry import builtin_registry


@dataclass
class RuntimeContext:
    profile: object
    task: object
    relay: object
    runtimes: dict = field(repr=False)


def write_json(path, value):
    import json

    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    tmp.replace(path)


async def execute(profile, task, runtimes, *, registry=None, credential=None):
    from agent.rsi.engine.model_gateway import GatewayPolicy

    from .gateway import EvidenceGateway
    from .native_expert_team import scientific_tool_exposure

    adapter, preflight = (registry or builtin_registry()).resolve(profile, runtimes)
    if profile.model.protocol == "responses":
        from .responses_gateway import TRANSPORT, ResponsesGateway, ResponsesPolicy

        policy = ResponsesPolicy.from_profile(profile, task.tools)
        gateway_type, transport_name = ResponsesGateway, TRANSPORT
    else:
        if profile.budget.temperature is None or profile.budget.reasoning_effort is not None:
            raise ValueError("Chat Completions needs its declared temperature/thinking controls")
        policy = GatewayPolicy(
            model=profile.model.model,
            mode=scientific_tool_exposure(profile.mode, profile.harness),
            scientific_tools=task.tools,
            max_calls=profile.budget.model_calls,
            temperature=profile.budget.temperature,
            max_tokens=profile.budget.output_tokens,
            thinking=profile.budget.thinking,
        )
        gateway_type, transport_name = EvidenceGateway, "buffered_complete_response_v1"
    if profile.harness in {"codex", "expert-codex"} and profile.model.protocol == "chat_completions":
        from .responses_bridge import ResponsesChatBridge, TRANSPORT, validate_profile
        validate_profile(profile)
        gateway_type, transport_name = ResponsesChatBridge, TRANSPORT
    if profile.harness in {"claude-code", "expert-claude-code"}:
        from .claude_adapter import validate_profile
        from .messages_bridge import MessagesBridge

        validate_profile(profile)
        gateway_type = MessagesBridge
        transport_name = "shared_chat_messages_free_bridge.v1"
    key = credential if credential is not None else os.environ.get(profile.model.credential_env)
    if not key:
        raise ValueError("Required model credential reference is unresolved")
    task.output.mkdir(parents=True, exist_ok=False)
    before = task.binding()
    from agent.rsi.layout import source_relative

    lock = {
        "schema": 1,
        "profile": profile.public(),
        "preflight": preflight,
        "task": before,
        "adapter_source": {p.name: file_hash(p) for p in sorted([*Path(__file__).parent.glob("*.py"), *Path(__file__).parent.glob("*.mjs")])},
        "transport": transport_name,
        "phase_deadline": "includes adapter startup; cleanup time reported in host_wall_seconds",
        "supporting_source": {
            name: file_hash(Path(__file__).resolve().parents[3] / source_relative(Path(name).name) if name.startswith("rsi/") else Path(__file__).parent.parent / name)
            for name in [
                "rsi/model_gateway.py",
                "rsi/live_model_gateway.py",
                "rsi/model_access.py",
                "rsi/openrouter_policy.py",
                "rsi/runtime.py",
                "rsi/sandbox.py",
                "rsi/scientist_runtime.py",
                "rsi/scientific_episode.py",
                "rsi/adaptive_runtime.py",
                "rsi/langgraph_tools.py",
                "rsi/tool_contracts.py",
                "rsi/model_policy.py",
                "kimi_client.py",
            ]
        },
        "model_identity": "configured provider alias; immutable upstream snapshot unverified",
        "gateway_tool_exposure_mode": scientific_tool_exposure(profile.mode, profile.harness),
        "tool_budget_boundary": "runtime event observation; in-flight calls may race cancellation",
        "network_policy": "runtime shares network; tool workers and generated-code workers are isolated separately",
    }
    lock["sha256"] = digest(lock)
    write_json(task.output / "profile.lock.json", lock)
    started = time.monotonic()
    cancelled = False
    try:
        async with gateway_type(
            policy=policy,
            upstream_url=profile.model.base_url,
            api_key=key,
            output=task.output / "model_requests.jsonl",
            timeout_s=profile.budget.request_seconds,
            request_recovery=profile.request_recovery,
        ) as relay:
            async with asyncio.timeout(profile.budget.wall_seconds):
                result = await adapter.run(RuntimeContext(profile, task, relay, runtimes))
    except asyncio.CancelledError:
        # Persist observed usage after adapter/gateway cleanup, then preserve
        # cancellation semantics for ordinary callers.
        cancelled = True
        result = {"status": "interrupted", "error_type": "CancelledError", "phase": task.phase}
    except TimeoutError:
        result = {"status": "timeout", "error_type": "TimeoutError"}
    except Exception as exc:
        # Provider SDK exceptions may include authentication headers or response bodies.
        result = {"status": "runtime_error", "error_type": type(exc).__name__}
    result["host_wall_seconds"] = round(time.monotonic() - started, 3)
    after = task.binding()
    result["binding_intact"] = (
        all(after["inputs"].get(k) == v for k, v in before["inputs"].items())
        and before["capabilities"] == after["capabilities"]
    )
    result["audit"] = {
        "events": audit_trace(task.output / "events.jsonl", task.tools, profile.budget.tool_calls),
        "requests": audit_requests(task.output / "model_requests.jsonl"),
    }
    result["audit"]["passed"] = (
        result["audit"]["events"]["passed"] and result["audit"]["requests"]["passed"] and result["binding_intact"]
    )
    result["profile_sha256"] = lock["sha256"]
    result["harness"] = profile.harness
    result["mode"] = profile.mode
    from .native_expert_team import NATIVE_EXPERT_HARNESSES

    if profile.harness in NATIVE_EXPERT_HARNESSES and result["status"] == "completed":
        from .native_expert_audit import audit_native_expert

        try:
            result["audit"]["native_expert"] = audit_native_expert(
                task.output, task.workspace, task.capability, result=result
            )
        except Exception as exc:
            result["audit"]["native_expert"] = {"passed": False, "error_type": type(exc).__name__}
        result["audit"]["passed"] = result["audit"]["passed"] and result["audit"]["native_expert"]["passed"]
    if result["status"] == "completed" and not result["audit"]["passed"]:
        result["status"] = "protocol_failed"
    if not result["binding_intact"]:
        result["status"] = "binding_modified"
    if cancelled:
        result["llm_calls"] = result["audit"]["requests"]["model_calls"]
        result["tool_calls"] = result["audit"]["events"]["tool_calls"]
        result["cancellation_finalized"] = True
    write_json(task.output / "run.json", result)
    if cancelled:
        raise asyncio.CancelledError
    return result
