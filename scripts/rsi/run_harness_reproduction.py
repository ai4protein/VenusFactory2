"""Either scientific mode requests an admitted job, then reviews source-bound facts.

A shared execution host enforces one model binding and cumulative inference
budgets. Scientific jobs wait without spending LLM requests; their separate
compute deadline and end-to-end time remain explicit. Existing-evidence review
never relabels old computations as agent-generated experiments.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import time
from dataclasses import replace
from pathlib import Path

from run_harness_probe import ROOT, credential_from_toml, default_runtimes
from run_protein_lifecycle import audit_events

from agent.harness import Profile, Task
from agent.harness.contracts import digest, file_hash
from agent.harness.host import execute, write_json
from agent.rsi.engine.case_evidence import canonical_claims, render_evidence_report, validate_review
from agent.rsi.campaigns.case_jobs import CaseJobBroker
from agent.rsi.campaigns.case_plugins import PLUGINS, antibody_allocation, invocation, maximum_case_job_seconds
from agent.rsi.engine.source_snapshot import research_implementation_paths
from agent.rsi.engine.versions import VersionStore


def read(path):
    return json.loads(path.read_text())


def bundle(folder, files, metadata, facts=None, episode=None, recipe=None):
    source = folder / "source"
    source.mkdir(parents=True)
    for src, dst in files:
        target = source / dst
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / src, target)
    if facts is not None:
        (source / "agent/__init__.py").write_text("")
        (source / "agent/rsi/__init__.py").write_text("")
        write_json(source / "evidence.json", facts)
    if episode is not None:
        write_json(source / "scientific_task.json", {"schema": 1, "kind": "reproduction", **episode})
    if recipe is not None:
        write_json(source / "registered_recipe.json", recipe)
    versions = VersionStore(folder / "bundles")
    manifest = versions.snapshot(source, metadata=metadata)
    return versions, manifest["version_id"], folder / "bundles/versions" / manifest["version_id"]


def archive_implementation(paths, output):
    # Frozen source files retain read-only permissions after copy2. Deduplicate
    # before copying so overlapping source groups cannot reopen those files.
    paths = list(dict.fromkeys(paths))
    hashes = {str(p.relative_to(ROOT)): file_hash(p) for p in paths}
    for p in paths:
        relative = p.relative_to(ROOT)
        dest = output / "implementation" / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest)
        if file_hash(dest) != hashes[str(relative)]:
            raise ValueError("Implementation changed while being archived")
    return hashes


def submission_receipt_matches(workspace, jobs):
    """Check the tool-written receipt against the actual host allocation."""
    path = workspace / "case_submission.json"
    if len(jobs) != 1 or path.is_symlink() or not path.is_file():
        return False
    job = next(iter(jobs.values()))
    try:
        received = read(path)
        expected = {
            "schema": 1,
            "job_id": job["job_id"],
            "plugin_id": job["plugin_id"],
            "parameters": job["parameters"],
            "submission_only": True,
        }
        return digest(received) == digest(expected)
    except (OSError, ValueError):
        return False


async def main(args, *, profile=None, credential=None, goal_context="", registered_recipe=None):
    output = args.output.resolve()
    output.relative_to((ROOT / "research_runs/rsi").resolve())
    profile = profile or Profile.from_dict(read(args.profile))
    if profile.mode not in {"science_agent", "science_expert"}:
        raise ValueError("Unsupported scientific mode")
    parameters = json.loads(args.parameters)
    invocation(
        args.plugin,
        parameters,
        root=ROOT,
        output=output / "preflight",
        controller_python=Path(default_runtimes()["python_bin"]),
    )
    maximum_backbones = getattr(args, "maximum_backbones", 8)
    maximum_sequences = getattr(args, "maximum_sequences_per_backbone", 4)
    backbones, sequences = antibody_allocation(args.plugin, parameters)
    if args.plugin.startswith("rfantibody") and (backbones > maximum_backbones or sequences > maximum_sequences):
        raise ValueError("Declared antibody allocation exceeds predeclared job limits")
    key = credential if credential is not None else credential_from_toml(args.credential_toml, profile.model)
    if registered_recipe is not None and (
        registered_recipe.get("plugin") != args.plugin
        or digest(registered_recipe.get("parameters")) != digest(parameters)
    ):
        raise ValueError("Registered recipe differs from the requested allocation")
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    paths = research_implementation_paths(ROOT)
    hashes = archive_implementation(paths, output)
    protocol = {
        "profile": profile.public(),
        "plugin": args.plugin,
        "parameters": parameters,
        "maximum_jobs": 1,
        "goal_context_sha256": digest(goal_context),
        "registered_recipe_sha256": digest(registered_recipe) if registered_recipe is not None else None,
        "job_timeout_s": args.job_timeout,
        "design_limits": {"maximum_backbones": maximum_backbones, "maximum_sequences_per_backbone": maximum_sequences},
        "implementation": hashes,
        "interaction": "two four-role scientific episodes around a host-owned job"
        if profile.mode == "science_expert"
        else "two conversational turns around a host-owned job",
        "persistence": "job records persist; stopping this supervisor cancels its active computations",
        "claim_scope": "task submission, computational execution or explicitly labelled reuse, and source-bound evidence review; no comparative mode or biological superiority claim",
    }
    write_json(output / "protocol.json", protocol)
    report = {
        "status": "running",
        "protocol_sha256": digest(protocol),
        "agent_requested": False,
        "end_to_end_verified": False,
        "new_computation": False,
    }
    write_json(output / "report.json", report)
    workspace = output / "workspace"
    workspace.mkdir()
    versions, version, cap = bundle(
        output / "launch_capability",
        [("src/agent/rsi/campaigns/case_job_tools.py", "generated_tools.py")],
        {"plugin": args.plugin},
        episode={
            "context_tool": "inspect_registered_reproduction" if registered_recipe else "discover_reproduction_plugins",
            "artifact_file": "case_submission.json",
            "schema": 2,
            "required_tools": ["discover_reproduction_plugins", "submit_reproduction", "inspect_reproduction_job"]
            + (["inspect_registered_reproduction"] if registered_recipe else []),
            "completion_repair_limit": 1,
        },
        recipe=registered_recipe,
    )
    required_launch = {"discover_reproduction_plugins", "submit_reproduction", "inspect_reproduction_job"}
    if registered_recipe is not None:
        required_launch.add("inspect_registered_reproduction")
    tools = frozenset(required_launch)
    async with CaseJobBroker(
        root=ROOT,
        workspace=workspace,
        output=output / "controller",
        allowed_plugins={args.plugin},
        max_jobs=1,
        timeout_s=args.job_timeout,
        expected_parameters={args.plugin: parameters},
        maximum_backbones=maximum_backbones,
        maximum_sequences_per_backbone=maximum_sequences,
    ) as broker:
        prompt = (
            "Discover the admitted reproduction plugin, then submit exactly one job for "
            + args.plugin
            + " with exact parameters "
            + json.dumps(parameters)
            + ". Use a stable idempotency key. Inspect the returned job once and report its actual ID and state. Queued/running is not completed. The host will wait and return final evidence in another conversation turn; do not poll repeatedly."
        )
        prompt = goal_context + "\n\n" + prompt
        if registered_recipe is not None:
            prompt += " First inspect_registered_reproduction and reuse its exact plugin and parameters. This fresh computation tests reuse of an existing recipe, not algorithm invention or independent biological replication."
        task = Task(
            workspace,
            cap,
            output / "launch",
            prompt,
            "Use only supplied reproduction tools. Report actual evidence and preserve computational versus experimental scope.",
            tools,
            phase="chat",
        )
        result = await execute(profile, task, default_runtimes(), credential=key)
        audit = audit_events(
            task.output,
            "launch",
            result,
            required_launch,
        )
        report["launch"] = {"run": result, "scientific_audit": audit}
        versions.verify(version)
        report["submission_receipt_verified"] = submission_receipt_matches(workspace, broker.jobs)
        report["agent_requested"] = (
            report["submission_receipt_verified"]
            and result["status"] == "completed"
            and result["audit"]["passed"]
            and audit["passed"]
            and len(broker.jobs) == 1
        )
        write_json(output / "report.json", report)
        print(
            json.dumps(
                {
                    "phase": "launch",
                    "status": result["status"],
                    "agent_requested": report["agent_requested"],
                    "jobs": len(broker.jobs),
                }
            ),
            flush=True,
        )
        # Retain launched compute even if the final launch turn fails its audit.
        await broker.wait_jobs()
        report["jobs"] = [broker.status(j) for j in broker.jobs]
        if report["agent_requested"] and len(broker.jobs) == 1 and report["jobs"][0]["status"] == "completed":
            jid = report["jobs"][0]["job_id"]
            source_report = Path(broker.jobs[jid]["output"]) / "report.json"
            source_sha = file_hash(source_report)
            raw = read(source_report)
            facts = canonical_claims(args.plugin, raw)
            remaining = {
                "model_calls": profile.budget.model_calls - result["audit"]["requests"]["model_calls"],
                "tool_calls": profile.budget.tool_calls - result["audit"]["events"]["tool_calls"],
                "wall_seconds": profile.budget.wall_seconds - result["host_wall_seconds"],
            }
            if any(n <= 0 for n in remaining.values()):
                report["status"] = "inference_budget_exhausted"
            else:
                review_profile = replace(profile, budget=replace(profile.budget, **remaining))
                rv, vid, rcap = bundle(
                    output / "review_capability",
                    [
                        ("src/agent/rsi/engine/case_review_tools.py", "generated_tools.py"),
                        ("src/agent/rsi/engine/case_evidence.py", "agent/rsi/engine/case_evidence.py"),
                        ("src/agent/rsi/engine/protein_model.py", "agent/rsi/engine/protein_model.py"),
                        ("src/agent/rsi/cases/mcba_evidence.py", "agent/rsi/cases/mcba_evidence.py"),
                        ("src/agent/rsi/cases/rfantibody_evidence.py", "agent/rsi/cases/rfantibody_evidence.py"),
                        ("src/agent/rsi/engine/storage.py", "agent/rsi/engine/storage.py"),
                    ],
                    {"plugin": args.plugin, "source_report_sha256": source_sha},
                    {"job_id": jid, "plugin": args.plugin, "report_sha256": source_sha, "facts": facts},
                    episode={
                        "context_tool": "inspect_case_evidence",
                        "artifact_file": "review.json",
                        "schema": 2,
                        "required_tools": ["inspect_case_evidence", "submit_evidence_review"],
                        "completion_repair_limit": 1,
                    },
                )
                work = output / "review_workspace"
                work.mkdir()
                request = Task(
                    work,
                    rcap,
                    output / "review",
                    goal_context
                    + "\n\nInspect the completed case evidence and submit exactly the source-bound facts via submit_evidence_review. Retain every field and type. Do not add interpretation or claim independent replicates from reused source seeds. Correct a rejected submission at most once and report the accepted file.",
                    "Review scientific evidence using only the two supplied tools. Do not relabel old computation as newly generated.",
                    frozenset({"inspect_case_evidence", "submit_evidence_review"}),
                    phase="chat",
                )
                run = await execute(review_profile, request, default_runtimes(), credential=key)
                review_audit = audit_events(
                    request.output, "review", run, set(request.tools), {"submit_evidence_review"}
                )
                report["review"] = {"run": run, "scientific_audit": review_audit, "source_report_sha256": source_sha}
                rv.verify(vid)
                if (
                    run["status"] == "completed"
                    and run["audit"]["passed"]
                    and review_audit["passed"]
                    and file_hash(source_report) == source_sha
                ):
                    claims = read(work / "review.json")["claims"]
                    report["review"]["verified"] = validate_review(claims, facts)
                    (output / "verified_evidence.md").write_text(render_evidence_report(jid, claims))
                    report.update(
                        status="completed",
                        end_to_end_verified=True,
                        new_computation=raw.get("execution_origin") != "verified_existing",
                    )
                else:
                    report["status"] = "review_failed"
        else:
            report["status"] = "launch_or_job_failed"
    report["end_to_end_wall_seconds"] = round(time.monotonic() - started, 3)
    report["implementation_unchanged"] = all(file_hash(ROOT / n) == sha for n, sha in hashes.items())
    if not report["implementation_unchanged"]:
        report.update(status="implementation_changed", end_to_end_verified=False)
    write_json(output / "report.json", report)
    print(
        json.dumps(
            {
                k: report[k]
                for k in [
                    "status",
                    "agent_requested",
                    "end_to_end_verified",
                    "new_computation",
                    "end_to_end_wall_seconds",
                ]
            }
        ),
        flush=True,
    )
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--profile", type=Path, required=True)
    p.add_argument("--credential-toml", type=Path, required=True)
    p.add_argument("--plugin", choices=sorted(PLUGINS), required=True)
    p.add_argument("--parameters", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--job-timeout", type=int, default=10800)
    p.add_argument("--maximum-backbones", type=int, default=8)
    p.add_argument("--maximum-sequences-per-backbone", type=int, default=4)
    args = p.parse_args()
    deadline_limit = maximum_case_job_seconds(args.plugin)
    if not 1 <= args.job_timeout <= deadline_limit:
        p.error(f"job deadline must be 1..{deadline_limit} seconds for this plugin")
    if not 1 <= args.maximum_backbones <= 1000 or not 1 <= args.maximum_sequences_per_backbone <= 20:
        p.error("invalid predeclared design limits")
    asyncio.run(main(args))
