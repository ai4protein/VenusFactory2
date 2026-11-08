#!/usr/bin/env python3
"""Check configured free model/Harness pairs with the actual Web tool pipeline."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
HARNESSES = ["kimi-code", "deepseek-harness", "codex", "claude-code"]


async def probe(cat, runtimes, harness, model_id, folder):
    from langchain_core.tools import StructuredTool
    from agent.harness.catalog import free_model, resolve_credential
    from agent.harness.failure_summary import public_failure
    from agent.harness.gateway import EvidenceGateway
    from agent.harness.interactions import INTERACTION_NAMES
    from agent.harness.messages_bridge import MessagesBridge
    from agent.harness.responses_gateway import ResponsesGateway, ResponsesPolicy
    from agent.harness.responses_bridge import ResponsesChatBridge
    from agent.harness.tool_disclosure import CONTROLS
    from agent.harness.web_native_worker import run
    from agent.rsi.engine.model_gateway import GatewayPolicy

    profile, option = cat.select({"harness": harness, "model_id": model_id})
    if not free_model(profile.model):
        raise ValueError("Compatibility probes allow only configured free routes")
    profile = replace(profile, budget=replace(
        profile.budget, wall_seconds=210, request_seconds=80, model_calls=6,
        tool_calls=8, output_tokens=min(1024, profile.budget.output_tokens),
    ))
    workspace = folder / "workspace"
    workspace.mkdir(parents=True)
    invocations, events, deltas = [], [], []
    started = time.monotonic()

    async def residue_count(sequence: str):
        """Count residues and cysteines in a supplied amino-acid sequence."""
        invocations.append(sequence)
        result = {"length": len(sequence), "cysteines": sequence.count("C")}
        (workspace / "count.json").write_text(json.dumps(result))
        # Exercise ordinary Markdown results, including a trailing budget note.
        return f"# Result\nResidues: {result['length']}\nCysteines: {result['cysteines']}\n\nFinish with a brief answer."

    tool = StructuredTool.from_function(coroutine=residue_count, metadata={"workspace_relative_paths": True})
    names = CONTROLS | INTERACTION_NAMES
    if profile.model.protocol == "responses":
        policy, gateway = ResponsesPolicy.from_profile(profile, names), ResponsesGateway
    else:
        policy = GatewayPolicy(
            model=profile.model.model, mode="science_agent", scientific_tools=names,
            max_calls=profile.budget.model_calls, max_tokens=profile.budget.output_tokens,
            temperature=profile.budget.temperature, thinking=profile.budget.thinking,
        )
        gateway = (MessagesBridge if harness == "claude-code" else
                   ResponsesChatBridge if harness == "codex" else EvidenceGateway)

    async def on_text(call, text):
        if text:
            deltas.append(round(time.monotonic() - started, 3))

    record = {"harness": harness, "model_id": model_id, "protocol": profile.model.protocol}
    requests = folder / "requests.jsonl"
    try:
        async with gateway(
            policy=policy, upstream_url=profile.model.base_url,
            api_key=resolve_credential(profile.model, option.credential_toml),
            output=requests, timeout_s=profile.budget.request_seconds, on_text=on_text,
        ) as relay:
            request = {
                "profile": profile.public(), "runtimes": runtimes,
                "workspace": str(workspace), "control": str(folder), "session_id": folder.name,
                "text": "Use residue_count for ACDEFGHIKC. Search and inspect the tool first, "
                        "call it exactly once, then briefly report its residue and cysteine counts.",
                "lang": "en", "messages": [], "base_url": relay.base_url,
                "token": relay.token, "audit_required": True,
            }
            async with asyncio.timeout(profile.budget.wall_seconds + 5):
                await run(request, tools=[tool], event_sink=events.append)
        completed = next((e for e in reversed(events) if e.get("kind") == "completed"), None)
        passed = bool(completed and invocations == ["ACDEFGHIKC"] and deltas)
        record.update(status="passed" if passed else "conformance_failed", passed=passed)
    except Exception as exc:
        record.update(status="failed", passed=False, error_type=type(exc).__name__,
                      failure=public_failure(requests))
    record.update(seconds=round(time.monotonic() - started, 3), tool_executions=len(invocations),
                  stream_chunks=len(deltas), first_text_seconds=deltas[0] if deltas else None)
    (folder / "summary.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


async def main(args):
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    from backend.web_v2.agent_execution import catalog, runtime_paths
    from agent.harness.catalog import free_model

    cat, runtimes = catalog(), runtime_paths()
    free_ids = [m.id for m in cat.models.values() if free_model(m.binding)]
    model_ids = args.models or free_ids
    if not set(model_ids) <= set(free_ids):
        raise ValueError("Choose enabled free model IDs from the configured catalog")
    public = cat.public(runtimes)
    pairs = [p for p in public["combinations"] if p["harness"] in args.harnesses and p["model_id"] in model_ids]
    args.output.mkdir(parents=True, exist_ok=False, mode=0o700)
    report = {"readiness": public["readiness"], "live": args.execute, "pairs": pairs, "runs": []}
    target = args.output / "report.json"

    def save():
        target.write_text(json.dumps(report, indent=2) + "\n")

    save()
    for pair in pairs:
        if not args.execute or not pair["ready"]:
            print(json.dumps(pair), flush=True)
            continue
        name = pair["harness"] + "--" + pair["model_id"]
        print("Checking " + name, flush=True)
        result = await probe(cat, runtimes, pair["harness"], pair["model_id"], args.output / name)
        report["runs"].append(result)
        save()
        print(json.dumps(result), flush=True)
    return int(args.execute and (not report["runs"] or any(not row["passed"] for row in report["runs"])))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", help="Configured free model IDs")
    parser.add_argument("--harnesses", nargs="+", choices=HARNESSES, default=HARNESSES)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true", help="Send synthetic probes to configured free APIs")
    args = parser.parse_args()
    if args.execute and not args.models:
        parser.error("Live probes require explicit --models to bound API requests")
    args.output = args.output.resolve()
    os.umask(0o077)
    sys.exit(asyncio.run(main(args)))
