"""New GLM session reviews existing computational evidence, with host replay."""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import tomllib
from pathlib import Path

from run_protein_lifecycle import ROOT, audit_events

from agent.rsi.engine.case_evidence import canonical_claims, render_evidence_report, validate_review
from agent.rsi.engine.live_model_gateway import ModelGateway
from agent.rsi.engine.model_gateway import GatewayPolicy
from agent.rsi.engine.model_policy import read_model_configuration
from agent.rsi.engine.runtime import run_task
from agent.rsi.engine.sandbox import SandboxKimi
from agent.rsi.engine.storage import atomic_json, sha256_file
from agent.rsi.engine.versions import VersionStore


from agent.rsi.layout import source_relative

async def review_case(*, source_report, plugin, job_id, output, config):
    output.mkdir(parents=True, exist_ok=False)
    _, identity = read_model_configuration(Path(config["kimi_config"]), required=config["required_model"])
    private = tomllib.loads(Path(config["kimi_config"]).read_text())
    provider = private["providers"][identity["provider"]]
    source_sha = sha256_file(source_report)
    expected = canonical_claims(plugin, json.loads(source_report.read_text()))
    source = output / "bundle_source"
    (source / "agent/rsi/engine").mkdir(parents=True)
    for name in ["case_evidence.py", "protein_model.py"]:
        shutil.copy2(ROOT / source_relative(name), source / "agent/rsi/engine" / name)
    for path in [source / "agent/__init__.py", source / "agent/rsi/__init__.py", source / "agent/rsi/engine/__init__.py"]:
        path.write_text("")
    shutil.copy2(ROOT / "src/agent/rsi/engine/case_review_tools.py", source / "generated_tools.py")
    atomic_json(
        source / "evidence.json",
        {
            "job_id": job_id,
            "plugin": plugin,
            "report_sha256": source_sha,
            "facts": expected,
            "review_policy": "Submit source-bound structured facts. Free-form narrative is unverified and will not become the scientific report.",
        },
    )
    versions = VersionStore(output / "bundles")
    manifest = versions.snapshot(source, metadata={"model": identity, "plugin": plugin, "job_id": job_id})
    capability = output / "bundles/versions" / manifest["version_id"]
    workspace = output / "workspace"
    workspace.mkdir()
    tools = frozenset({"inspect_case_evidence", "submit_evidence_review"})
    policy = GatewayPolicy(
        model=identity["model"],
        mode="science_agent",
        scientific_tools=tools,
        max_calls=8,
        max_tokens=12288,
        thinking="enabled",
    )
    async with ModelGateway(
        policy=policy,
        upstream_url=provider["base_url"],
        api_key=provider["api_key"],
        output=output / "model_requests.jsonl",
        timeout_s=180,
    ) as relay:
        sandbox = SandboxKimi(
            workspace=workspace,
            control_dir=output / "control",
            capability_dir=capability,
            kimi_bin=Path(config["kimi_bin"]),
            config_path=Path(config["kimi_config"]),
            python_root=Path(config["python_root"]),
            expected_model=identity,
            mcp_entry="generated_tools.py",
            relay=relay,
        )
        async with sandbox:
            run = await run_task(
                sandbox.client,
                prompt="Inspect the pinned case evidence, then submit the exact source-bound facts as claims_json via submit_evidence_review. Preserve types and every field. Do not infer that experimental measurements came from inference, conflate table duplicates with token indexing, or equate design count with filter passes. You may correct a rejected submission once. End by reporting only the accepted artifact path.",
                system_prompt="You are the scientific evidence reviewer. Use only the declared case evidence tools. Every factual claim must match the controller-derived report; no additional biological interpretation is requested.",
                model=identity["configured_id"],
                output_dir=output,
                timeout_s=420,
                max_tool_calls=6,
                allowed_mcp_tools=tools,
                request_gateway=True,
            )
    audit = audit_events(output, "review", run, set(tools), {"submit_evidence_review"})
    audit["source_unchanged"] = sha256_file(source_report) == source_sha
    result = {
        "run": run,
        "audit": audit,
        "source_report": str(source_report),
        "source_sha256": source_sha,
        "plugin": plugin,
        "job_id": job_id,
        "status": "review_failed",
    }
    if audit["passed"] and audit["source_unchanged"] and (workspace / "review.json").exists():
        claims = json.loads((workspace / "review.json").read_text())["claims"]
        result["verified"] = validate_review(claims, canonical_claims(plugin, json.loads(source_report.read_text())))
        (output / "verified_evidence.md").write_text(render_evidence_report(job_id, claims))
        result["status"] = "completed"
    versions.verify(manifest["version_id"])
    atomic_json(output / "report.json", result)
    print(json.dumps({"job_id": job_id, "review_status": result["status"]}), flush=True)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-report", type=Path, required=True)
    p.add_argument("--plugin", choices=["turbopetase.v1", "rfantibody.v2"], required=True)
    p.add_argument("--job-id", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--config", type=Path, default=ROOT / "configs/rsi/generated_models_glm51.json")
    args = p.parse_args()
    asyncio.run(
        review_case(
            source_report=args.source_report,
            plugin=args.plugin,
            job_id=args.job_id,
            output=args.output,
            config=json.loads(args.config.read_text()),
        )
    )
