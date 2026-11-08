"""Standalone SDK worker; launched inside the research filesystem/PID namespace."""

from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

from events import normalize_dsh
from mcp_config import deepseek_mcp_plugin, scientific_server


def main():
    from deepseek_harness import DeepSeekHarness

    request = json.load(sys.stdin)
    root = Path("/evidence")
    # Minimal composition owns its loop. Remove native shell tools instead of wrapping another loop.
    patch = [
        {"id": name, "disabled": True}
        for name in ["persistent-bash", "persistent-pwsh", "terminal-bash", "terminal-pwsh"]
    ]
    server = scientific_server(entry=request["entry"], tools=request["tools"], clean_pythonpath=True)
    patch.append({"insert": [deepseek_mcp_plugin(server, request["timeout"]),
        {"id": "venus-tool-arguments", "name": "/runner/dsh_argument_guard.mjs"}]})
    patch_path = root / "composition.patch.yml"
    patch_path.write_text(json.dumps(patch, indent=2))
    (root / "events.jsonl").touch(exist_ok=False)
    count = 0
    sid = "venus-" + uuid.uuid4().hex
    expected = {"mcp__venusfactory__" + name for name in request["tools"]}

    def observe(notification):
        nonlocal count
        if notification.method != "session.event" or notification.payload.get("sessionId") != sid:
            return
        event = notification.payload.get("event", {})
        with (root / "runtime_events.jsonl").open("a") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
        with (root / "events.jsonl").open("a") as f:
            for row in normalize_dsh(event):
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        if event.get("type") == "tool/call":
            count += 1
            if count > request["tool_calls"]:
                raise RuntimeError("scientific_tool_budget_exceeded")
            data = event.get("data", {})
            name = data.get("name", data.get("toolName"))
            if name not in expected:
                raise RuntimeError("undeclared_scientific_tool")

    result = {"status": "runtime_error", "session_id": sid, "tool_calls": 0}
    try:
        with DeepSeekHarness(
            dsh_home="/runtime_home/dsh",
            profile="sdk-minimal",
            patches=(str(patch_path),),
            cwd="/workspace",
            provider="deepseek-official",
            model=request["model"],
            max_tokens=request["max_tokens"],
            base_url=request["base_url"],
            api_key=request["token"],
            env={"DSH_CONTEXT_WINDOW": str(request["context_tokens"])},
            request_timeout_seconds=request["timeout"],
            initialize_timeout_seconds=30,
        ) as harness:
            answer = harness.run(request["prompt"], session_id=sid, on_notification=observe)
            status = "completed" if answer.finish_reason == "completed" else "runtime_error"
            if (
                request.get("phase") == "chat"
                and status == "completed"
                and not str(answer.final_response or "").strip()
            ):
                status = "incomplete"
            result.update(
                status=status,
                finish_reason=answer.finish_reason,
                text=answer.final_response,
            )
    except Exception as exc:
        result["error_type"] = type(exc).__name__
        # Ephemeral local token may appear in SDK initialization diagnostics.
        print(str(exc).replace(request["token"], "[redacted]")[:4000], file=sys.stderr)
    result["tool_calls"] = count
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
