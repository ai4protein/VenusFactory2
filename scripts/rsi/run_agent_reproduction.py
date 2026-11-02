"""GLM submits a pinned reproduction job, then independently inspects its result."""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import tomllib
from pathlib import Path

from run_protein_lifecycle import ROOT, audit_events
from run_verified_case_review import review_case

from agent.rsi.campaigns.case_jobs import CaseJobBroker
from agent.rsi.engine.live_model_gateway import ModelGateway
from agent.rsi.engine.model_gateway import GatewayPolicy
from agent.rsi.engine.model_policy import read_model_configuration
from agent.rsi.engine.runtime import run_task
from agent.rsi.engine.sandbox import SandboxKimi
from agent.rsi.engine.storage import atomic_json, sha256_file
from agent.rsi.engine.versions import VersionStore

SYSTEM = "You are VenusFactory's scientific execution agent. Use only the declared reproduction MCP tools. Submit the requested pinned computational workflow within its stated budget and report actual job evidence. Queued is not completed. Controller-verified computational outputs do not establish experimental binding, PET degradation or full-paper replication. Never invent outputs or metrics. Do not ask for human intervention."


async def phase(folder, phase_name, prompt, workspace, capability, config, identity, provider):
    folder.mkdir()
    allowed = {"discover_reproduction_plugins", "inspect_reproduction_job"}
    if phase_name == "launch":
        allowed.add("submit_reproduction")
    required = (
        {"discover_reproduction_plugins", "submit_reproduction"}
        if phase_name == "launch"
        else {"inspect_reproduction_job"}
    )
    policy = GatewayPolicy(
        model=identity["model"],
        mode="science_agent",
        scientific_tools=allowed,
        max_calls=12,
        max_tokens=12288,
        thinking="enabled",
    )
    async with ModelGateway(
        policy=policy,
        upstream_url=provider["base_url"],
        api_key=provider["api_key"],
        output=folder / "model_requests.jsonl",
        timeout_s=180,
    ) as relay:
        sandbox = SandboxKimi(
            workspace=workspace,
            control_dir=folder / "control",
            capability_dir=capability,
            kimi_bin=Path(config["kimi_bin"]),
            config_path=Path(config["kimi_config"]),
            python_root=Path(config["python_root"]),
            expected_model=identity,
            mcp_entry="generated_tools.py",
            relay=relay,
        )
        async with sandbox:
            result = await run_task(
                sandbox.client,
                prompt=prompt,
                system_prompt=SYSTEM,
                model=identity["configured_id"],
                output_dir=folder,
                timeout_s=420,
                max_tool_calls=12,
                allowed_mcp_tools=frozenset(allowed),
                request_gateway=True,
            )
    audit = audit_events(folder, phase_name, result, required)
    atomic_json(folder / "audit.json", audit)
    return {"run": result, "audit": audit}


async def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    config = json.loads(args.config.read_text())
    _, identity = read_model_configuration(Path(config["kimi_config"]), required=config["required_model"])
    private = tomllib.loads(Path(config["kimi_config"]).read_text())
    provider = private["providers"][identity["provider"]]
    parameters = json.loads(args.parameters)
    workspace = args.output / "workspace"
    workspace.mkdir()
    source = args.output / "bundle_source"
    source.mkdir()
    # The existing runtime entry whitelist is retained; the immutable manifest
    # records this explicit entry-name adapter, not a generated-model operation.
    shutil.copy2(ROOT / "src/agent/rsi/campaigns/case_job_tools.py", source / "generated_tools.py")
    atomic_json(
        source / "adapter.json",
        {
            "source": "src/agent/rsi/campaigns/case_job_tools.py",
            "entry_alias": "generated_tools.py",
            "sha256": sha256_file(ROOT / "src/agent/rsi/campaigns/case_job_tools.py"),
        },
    )
    versions = VersionStore(args.output / "bundles")
    manifest = versions.snapshot(source, metadata={"plugin": args.plugin, "model": identity})
    capability = args.output / "bundles/versions" / manifest["version_id"]
    report = {
        "status": "running",
        "plugin": args.plugin,
        "parameters": parameters,
        "model": identity,
        "capability_version": manifest["version_id"],
        "agent_orchestrated": False,
        "maximum_jobs": 1,
        "review_semantics": "host-replayed structured facts; free-form model prose is not validated evidence",
        "job_timeout_s": 1800,
        "launch_and_review_timeout_s": 420,
        "claim_scope": "agent-requested pinned computational workflow plus fresh-session result inspection; not full-paper or wet-lab replication",
        "implementation_sha256": {
            name: sha256_file(ROOT / name)
            for name in [
                "src/agent/rsi/campaigns/case_jobs.py",
                "src/agent/rsi/campaigns/case_job_tools.py",
                "src/agent/rsi/campaigns/case_plugins.py",
                "scripts/rsi/run_agent_reproduction.py",
                "scripts/rsi/run_verified_case_review.py",
                "src/agent/rsi/engine/case_review_tools.py",
                "src/agent/rsi/engine/case_evidence.py",
            ]
        },
    }
    atomic_json(args.output / "report.json", report)
    async with CaseJobBroker(
        root=ROOT,
        workspace=workspace,
        output=args.output / "controller",
        allowed_plugins={args.plugin},
        max_jobs=1,
        timeout_s=1800,
    ) as broker:
        prompt = (
            "Inspect the available reproduction plugin, then submit exactly one job for "
            + args.plugin
            + " with these exact parameters: "
            + json.dumps(parameters)
            + ". Use one stable idempotency key. Inspect its status once and return the actual job identity and current state. The controller will wait without consuming LLM calls and start a fresh result-inspection session when the job finishes."
        )
        report["launch"] = await phase(
            args.output / "launch", "launch", prompt, workspace, capability, config, identity, provider
        )
        atomic_json(args.output / "report.json", report)
        print(
            json.dumps({"phase": "launch", "status": report["launch"]["run"]["status"], "jobs": len(broker.jobs)}),
            flush=True,
        )
        await broker.wait_jobs()
        report["jobs"] = [broker.status(job_id) for job_id in broker.jobs]
        if report["launch"]["audit"]["passed"] and len(broker.jobs) == 1:
            job = report["jobs"][0]
            review_prompt = (
                "Independently inspect the completed or failed job "
                + job["job_id"]
                + " using inspect_reproduction_job. Report only controller-verified evidence, its scientific scope and what has not been reproduced. Do not submit any new job."
            )
            if job["status"] == "completed":
                report["review"] = await review_case(
                    source_report=Path(broker.jobs[job["job_id"]]["output"]) / "report.json",
                    plugin=args.plugin,
                    job_id=job["job_id"],
                    output=args.output / "review",
                    config=config,
                )
                report["agent_orchestrated"] = report["review"]["status"] == "completed"
            else:
                report["review"] = await phase(
                    args.output / "review", "review", review_prompt, workspace, capability, config, identity, provider
                )
                report["agent_orchestrated"] = False
            report["status"] = "completed" if report["agent_orchestrated"] else "review_or_job_failed"
        else:
            report["status"] = "launch_failed"
    versions.verify(manifest["version_id"])
    atomic_json(args.output / "report.json", report)
    print(
        json.dumps(
            {"status": report["status"], "plugin": args.plugin, "agent_orchestrated": report["agent_orchestrated"]}
        ),
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plugin", choices=["rfantibody.v2", "turbopetase.v1"], required=True)
    p.add_argument("--parameters", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--config", type=Path, default=ROOT / "configs/rsi/generated_models_glm51.json")
    asyncio.run(main(p.parse_args()))
