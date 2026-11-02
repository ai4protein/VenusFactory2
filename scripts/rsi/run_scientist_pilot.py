"""Exercise the genuine four-role graph using a completed Agent run's inputs.

This is an integration pilot. Full scientific comparisons additionally require
matched model parameters, repeated paired campaigns and audited native-tool
visibility differences; this script makes no mode-superiority claim.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import tomllib
from pathlib import Path

from run_generated_lifecycle import policy_for
from run_protein_lifecycle import ROOT, audit_events, prepare_case

from agent.rsi.engine.model_lifecycle import ModelLifecycle
from agent.rsi.engine.model_policy import read_model_configuration
from agent.rsi.engine.protein_model import digest
from agent.rsi.engine.scientist_runtime import run_scientist
from agent.rsi.engine.storage import atomic_json, sha256_file


async def main(args):
    if not 120 <= args.timeout_s <= 1800:
        raise ValueError("timeout must be between 120 and 1800 seconds")
    args.output.mkdir(parents=True, exist_ok=False)
    workspace = args.output / "workspace"
    workspace.mkdir()
    development = args.reference / "develop"
    binding = json.loads((development / "binding.json").read_text())
    capability = development / "bundles/versions" / binding["capability_version"]
    for name in ["training.json"]:
        shutil.copy2(development / "workspace" / name, workspace / name)
    prompt = (development / "submitted_prompt.txt").read_text()
    (args.output / "submitted_prompt.txt").write_text(prompt)
    config = json.loads(args.config.read_text())
    _, identity = read_model_configuration(Path(config["kimi_config"]), required=config["required_model"])
    if identity != binding["model"]:
        raise ValueError("model identity differs from reference run")
    private = tomllib.loads(Path(config["kimi_config"]).read_text())
    provider = private["providers"][identity["provider"]]
    protocol = {
        "mode": "science_expert",
        "reference_agent_run": str(args.reference),
        "model": identity,
        "task_prompt_sha256": sha256_file(args.output / "submitted_prompt.txt"),
        "capability_version": binding["capability_version"],
        "allowed_scientific_tools": binding["allowed_mcp_tools"],
        "timeout_s": args.timeout_s,
        "max_tool_calls": 12,
        "max_model_calls": 24,
        "max_output_tokens": 4096,
        "temperature": 0.1,
        "researcher_feedback": "predefined continuation only; no new biological information",
        "research_policy": args.research_policy,
        "strict_fair_comparison": False,
        "unresolved": [
            "native extra tools visible only in Agent runtime",
            "Agent sampling temperature not verified",
            "pilot has one lineage; no uncertainty estimate",
        ],
    }
    atomic_json(args.output / "protocol.json", protocol)
    (args.output / "implementation.py").write_bytes(Path(__file__).read_bytes())
    (args.output / "scientist_runtime.py").write_bytes((ROOT / "src/agent/rsi/engine/scientist_runtime.py").read_bytes())
    result = await run_scientist(
        workspace=workspace,
        capability=capability,
        python_root=Path(config["python_root"]),
        output=args.output,
        prompt=prompt,
        model=identity["model"],
        base_url=provider["base_url"],
        api_key=provider["api_key"],
        allowed_tools=binding["allowed_mcp_tools"],
        research_policy=args.research_policy,
        timeout_s=args.timeout_s,
    )
    audit = audit_events(args.output, "develop", result, {"submit_model_code"}, {"evaluate_model_code"})
    audit["inputs_unchanged"] = sha256_file(workspace / "training.json") == sha256_file(
        development / "workspace/training.json"
    )
    audit["passed"] = audit["passed"] and audit["inputs_unchanged"]
    atomic_json(args.output / "audit.json", audit)
    report = {"run": result, "audit": audit, "strict_fair_comparison": False}
    if audit["passed"]:
        scope, train, val, future, source, split = prepare_case(
            "alde_parpgb", ROOT / "research_runs/rsi/20261011_paper_case_audit"
        )
        if digest(train) != digest(json.loads((workspace / "training.json").read_text())):
            raise ValueError("training differs")
        lc = ModelLifecycle(args.output / "controller", policy_for(scope, train, val))
        mid = lc.submit(workspace / "candidate.json", session_id=result["session_id"], mode="science_expert")
        report["assessment"] = lc.assess(mid, training=train, validation=val)
    atomic_json(args.output / "report.json", report)
    print(
        json.dumps(
            {
                "status": result["status"],
                "tool_calls": result["tool_calls"],
                "llm_calls": result["llm_calls"],
                "role_ids": result.get("role_ids"),
                "error": result.get("error"),
                "audit": audit,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--reference", type=Path, default=ROOT / "research_runs/rsi/20261011_generated_alde_to_mcba_glm51_v1"
    )
    p.add_argument("--config", type=Path, default=ROOT / "configs/rsi/generated_models_glm51.json")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--research-policy", choices=["role_research", "task_context_only"], default="role_research")
    p.add_argument("--timeout-s", type=int, default=420)
    asyncio.run(main(p.parse_args()))
