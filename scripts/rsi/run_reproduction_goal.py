"""Persistent expert reproduction, verified recipe registration and fresh reuse.

A recipe packages an admitted workflow, not an invented model. Every successful
reuse performs new computation; it is not an independent biological replicate.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import time
from pathlib import Path
from types import SimpleNamespace

from reproduction_continuation import POLICY as REVIEW_CONTINUATION
from reproduction_continuation import (
    capture_pending,
    continue_review,
    pending_evidence,
    policy_enabled,
    verify_continuation,
)
from run_harness_probe import ROOT, credential_from_toml, default_runtimes
from run_harness_reproduction import archive_implementation, submission_receipt_matches
from run_harness_reproduction import main as reproduce
from run_protein_lifecycle import audit_events

from agent.harness import Profile
from agent.harness.contracts import digest, file_hash
from agent.harness.host import write_json
from agent.rsi.case_evidence import canonical_claims, validate_review
from agent.rsi.case_jobs import verify_job_report
from agent.rsi.case_plugins import PLUGINS, invocation
from agent.rsi.provider_access import access_wait
from agent.rsi.reproduction_goal_contract import CASES, CRITERIA, allocation
from agent.rsi.source_snapshot import research_implementation_paths
from agent.rsi.versions import VersionStore
from agent.science_goal import GoalSpec, GoalStore, run_goal


def read(path):
    return json.loads(path.read_text())


def safe_file(root, name):
    path = root / name
    path.resolve().relative_to(root.resolve())
    if any(part.is_symlink() for part in [path, *path.parents] if part != root.parent):
        raise ValueError("Symlinked goal evidence is not admitted")
    if not path.is_file():
        raise ValueError("Required goal evidence is missing")
    return path


def verify_run(folder, declaration, *, registered_recipe=None, replay_artifacts=True):
    report = read(safe_file(folder, "report.json"))
    protocol = read(safe_file(folder, "protocol.json"))
    if protocol.get("kind") == "interrupted_reproduction_review_recovery":
        return verify_continuation(folder, declaration, registered_recipe, replay_artifacts=replay_artifacts)
    if report.get("status") != "completed" or any(
        report.get(k) is not True
        for k in [
            "end_to_end_verified",
            "agent_requested",
            "new_computation",
            "implementation_unchanged",
            "submission_receipt_verified",
        ]
    ):
        raise ValueError("Reproduction did not pass all host gates")
    if report.get("protocol_sha256") != digest(protocol):
        raise ValueError("Reproduction protocol changed")
    if protocol["plugin"] != declaration["plugin"] or digest(protocol["parameters"]) != digest(
        declaration["parameters"]
    ):
        raise ValueError("Reproduction allocation changed")
    if protocol.get("registered_recipe_sha256") != (digest(registered_recipe) if registered_recipe else None):
        raise ValueError("Reproduction recipe binding changed")
    jobs = report.get("jobs", [])
    if len(jobs) != 1 or jobs[0].get("status") != "completed":
        raise ValueError("Exactly one verified job is required")
    job = jobs[0]
    jid = job["job_id"]
    if not isinstance(jid, str) or not re.fullmatch(r"case_[0-9a-f]{20}", jid):
        raise ValueError("Invalid job identity")
    if job["plugin_id"] != declaration["plugin"] or digest(job["parameters"]) != digest(declaration["parameters"]):
        raise ValueError("Host job differs from the admitted recipe")
    domain = safe_file(folder, "controller/" + jid + "/report.json")
    if file_hash(domain) != job["report_sha256"] or file_hash(domain) != report["review"]["source_report_sha256"]:
        raise ValueError("Domain report changed")
    raw = read(domain)
    if replay_artifacts:
        verify_job_report(ROOT, {**job, "output": str(domain.parent)}, raw)
    facts = canonical_claims(declaration["plugin"], raw)
    review_path = safe_file(folder, "review_workspace/review.json")
    validate_review(read(review_path)["claims"], facts)
    if not submission_receipt_matches(folder / "workspace", {jid: job}):
        raise ValueError("Submitted job receipt is not authentic")
    required = {"discover_reproduction_plugins", "submit_reproduction", "inspect_reproduction_job"}
    if registered_recipe is not None:
        required.add("inspect_registered_reproduction")
    sessions = []
    files = [
        "report.json",
        "protocol.json",
        "workspace/case_submission.json",
        "review_workspace/review.json",
        str(domain.relative_to(folder)),
    ]
    for phase, tools in [("launch", required), ("review", {"inspect_case_evidence", "submit_evidence_review"})]:
        run = report[phase]["run"]
        if run.get("status") != "completed" or run.get("audit", {}).get("passed") is not True:
            raise ValueError("Execution did not pass the shared harness audit")
        if digest(read(safe_file(folder, phase + "/run.json"))) != digest(run):
            raise ValueError("Execution record changed")
        if not audit_events(folder / phase, phase, run, tools, {"submit_evidence_review"})["passed"]:
            raise ValueError("Required scientific tool chain is missing")
        sid = run.get("session_id")
        if not isinstance(sid, str) or not sid or sid in sessions:
            raise ValueError("Distinct execution sessions are required")
        sessions.append(sid)
        files += [
            phase + "/" + name for name in ["run.json", "events.jsonl", "profile.lock.json", "model_requests.jsonl"]
        ]
    return {
        "files": {name: file_hash(safe_file(folder, name)) for name in files},
        "sessions": sessions,
        "job_id": jid,
        "facts_sha256": digest(facts),
        "new_computation": True,
        "full_paper_reproduced": False,
    }


class ReproductionGoal:
    def __init__(self, store, credential, *, case=None):
        self.store, self.credential = store, credential
        self.protocol = read(store.root / "protocol.json")
        self.review_continuation = policy_enabled(self.protocol, store.read())
        self.paper = None
        if case is None:
            self.declaration = self.protocol["allocation"]
            if self.declaration != allocation(self.protocol["case"]):
                raise ValueError("Goal allocation changed")
        else:
            from agent.rsi.paper_campaign import admitted_campaign

            papers = admitted_campaign(self.protocol, store.read())
            self.paper = next((p for p in papers if p["case"] == case), None)
            if self.paper is None:
                raise ValueError("Paper scope was not admitted to this campaign")
            self.declaration = self.paper["allocation"]
        for name, sha in self.protocol["implementation"].items():
            if file_hash(ROOT / name) != sha:
                raise ValueError("Goal implementation changed; explicit migration is required")
        self.versions = VersionStore(store.root / "recipes")

    def registered(self, state):
        version = state["progress"].get("recipe_version")
        if not version:
            return None
        self.versions.verify(version)
        recipe = read(self.versions.root / "versions" / version / "recipe.json")
        if recipe["plugin"] != self.declaration["plugin"] or digest(recipe["parameters"]) != digest(
            self.declaration["parameters"]
        ):
            raise ValueError("Registered allocation changed")
        if self.paper is not None and recipe.get("paper") != self.paper:
            raise ValueError("Registered paper scope changed")
        origin_goal, original = self.store.root, recipe
        if "library_origin" in recipe:
            from paper_library import source_recipe

            origin_goal, original = source_recipe(self.store.root, self.protocol, self.paper, recipe)
        folder = safe_file(origin_goal, original["source_run"] + "/report.json").parent
        if digest(verify_run(folder, self.declaration)) != digest(original["proof"]):
            raise ValueError("Registered scientific evidence changed")
        return recipe

    def evidence(self, path):
        return {"path": str(path.relative_to(self.store.root)), "sha256": file_hash(path)}

    def evaluate(self, state):
        checks = {key: {"passed": False, "evidence": []} for key in CRITERIA}
        recipe = self.registered(state)
        held = state["progress"].get("review_reconciliation")
        if held:
            if not self.review_continuation or set(held) != {"path", "sha256"}:
                raise ValueError("Invalid held review record")
            if file_hash(safe_file(self.store.root, held["path"])) != held["sha256"]:
                raise ValueError("Held computation record changed")
        pending = state["progress"].get("pending_review")
        if pending:
            if not self.review_continuation:
                raise ValueError("Pending review was not admitted")
            pending_evidence(self.store.root, pending, self.declaration, recipe)
        if recipe is not None:
            path = self.versions.root / "versions" / state["progress"]["recipe_version"] / "recipe.json"
            for key in ["verified_reproduction", "registered_recipe"]:
                checks[key] = {"passed": True, "evidence": [self.evidence(path)]}
            record = state["progress"].get("reuse_receipt")
            if record:
                path = safe_file(self.store.root, record)
                receipt = read(path)
                if receipt["recipe_version"] != state["progress"]["recipe_version"]:
                    raise ValueError("Reuse receipt names another recipe")
                folder = safe_file(self.store.root, receipt["source_run"] + "/report.json").parent
                proof = verify_run(folder, self.declaration, registered_recipe=recipe)
                if digest(proof) != digest(receipt["proof"]) or set(proof["sessions"]) & set(
                    recipe["proof"]["sessions"]
                ):
                    raise ValueError("Reuse is not independently verified in fresh sessions")
                if "library_origin" not in recipe and receipt["source_run"] == recipe["source_run"]:
                    raise ValueError("Reuse must execute a new computation")
                checks["fresh_session_reuse"] = {"passed": True, "evidence": [self.evidence(path)]}
        return {"criteria": checks}

    async def cycle(self, index, profile, state, context, *, prepared=False):
        started = time.monotonic()
        folder = self.store.root / "episodes" / f"cycle_{index:03d}"
        if prepared:
            if self.paper is None or read(safe_file(folder, "paper.json")) != self.paper:
                raise ValueError("Prepared episode does not match admitted paper")
        else:
            folder.mkdir(parents=True, exist_ok=False)
        recipe = self.registered(state)
        phase = "reuse" if recipe else "reproduce"
        output = folder / "reproduction"
        if state["progress"].get("review_reconciliation"):
            self.evaluate(state)
            outcome = {
                "phase": phase,
                "status": "failed",
                "needs_input": True,
                "reason": "completed_job_review_not_recoverable",
                "summary": "Completed computation retained; review provenance requires explicit reconciliation. No job was resubmitted.",
                "progress": dict(state["progress"]),
                "usage": {"model_calls": 0, "tool_calls": 0, "wall_seconds": time.monotonic() - started},
            }
            write_json(folder / "outcome.json", outcome)
            return outcome
        args = SimpleNamespace(
            output=output,
            plugin=self.declaration["plugin"],
            parameters=json.dumps(self.declaration["parameters"]),
            job_timeout=self.declaration["job_timeout"],
            maximum_backbones=64,
            maximum_sequences_per_backbone=4,
        )
        pending = state["progress"].get("pending_review")
        if pending:
            if not self.review_continuation:
                raise ValueError("Pending review was not admitted")
            report = await continue_review(
                self.store.root, output, pending, self.declaration, recipe, profile, self.credential, context
            )
        else:
            report = await reproduce(
                args, profile=profile, credential=self.credential, goal_context=context, registered_recipe=recipe
            )
        usage = {"model_calls": 0, "tool_calls": 0, "wall_seconds": 0.0}
        for key in ["launch", "review"]:
            if key in report:
                audit = report[key]["run"]["audit"]
                usage["model_calls"] += audit["requests"]["model_calls"]
                usage["tool_calls"] += audit["events"]["tool_calls"]
        progress = dict(state["progress"])
        outcome = {
            "phase": phase,
            "status": "failed",
            "summary": "Reproduction or evidence review did not pass; failed attempts remain available",
            "progress": progress,
            "usage": usage,
            "run_record": str((output / "report.json").relative_to(self.store.root)),
        }
        if report.get("needs_input"):
            outcome.update(needs_input=True, summary=report["summary"], reason=report["status"])
            if not (output / "report.json").is_file():
                outcome.pop("run_record", None)
        if pending:
            outcome["review_continuation"] = True
        elif self.review_continuation and report.get("status") in {"review_failed", "inference_budget_exhausted"}:
            try:
                progress["pending_review"] = capture_pending(self.store.root, folder, output, self.declaration, recipe)
            except (ValueError, OSError, KeyError) as exc:
                # A missing/unpaired trace is not proof of zero tool execution.
                # Retain the completed job and the full cycle reservation; a
                # restart must not silently submit the expensive job again.
                progress["review_reconciliation"] = self.evidence(output / "report.json")
                outcome.pop("usage", None)
                outcome.update(
                    needs_input=True,
                    reason="completed_job_review_not_recoverable",
                    continuation_error_type=type(exc).__name__,
                    summary="Completed computation retained; review provenance could not be verified. Explicit reconciliation is required; no automatic resubmission.",
                )
        if report.get("end_to_end_verified") is True or report.get("recovered_review_verified") is True:
            proof = verify_run(output, self.declaration, registered_recipe=recipe)
            progress.pop("pending_review", None)
            source_run = str(output.relative_to(self.store.root))
            if recipe is None:
                source = folder / "recipe_source"
                source.mkdir()
                value = {
                    "schema": 1,
                    "kind": "verified_reproduction_recipe",
                    "plugin": self.declaration["plugin"],
                    "parameters": self.declaration["parameters"],
                    "scope": PLUGINS[self.declaration["plugin"]].scope,
                    "source_run": source_run,
                    "proof": proof,
                    "novel_model_claim": False,
                }
                if self.paper is not None:
                    value["paper"] = self.paper
                write_json(source / "recipe.json", value)
                manifest = self.versions.snapshot(source, metadata={"kind": value["kind"]})
                # GoalStore publishes the version with its cycle checkpoint.
                # Unselected packages from interrupted attempts remain auditable.
                progress["recipe_version"] = manifest["version_id"]
                outcome.update(
                    status="completed",
                    summary="Computation and expert evidence review verified; recipe registered, fresh-session reuse still required",
                )
            else:
                if set(proof["sessions"]) & set(recipe["proof"]["sessions"]):
                    raise ValueError("Fresh reuse session identity required")
                path = folder / "reuse_receipt.json"
                write_json(
                    path,
                    {
                        "recipe_version": progress["recipe_version"],
                        "source_run": source_run,
                        "proof": proof,
                        "new_algorithm_claim": False,
                        "independent_biological_replicate": False,
                    },
                )
                progress["reuse_receipt"] = str(path.relative_to(self.store.root))
                outcome.update(
                    status="completed",
                    summary="Registered recipe retrieved and executed with reviewed evidence in fresh sessions",
                )
        if outcome["status"] != "completed" and not progress.get("review_reconciliation"):
            for stage in ("review", "launch"):
                waiting = access_wait(output / stage, self.store.root)
                if waiting:
                    outcome.update(waiting)
                    break
        usage["wall_seconds"] = time.monotonic() - started
        write_json(folder / "outcome.json", outcome)
        return outcome


def prepare(args):
    profile = Profile.from_dict(read(args.profile))
    declaration = allocation(args.case)
    invocation(
        declaration["plugin"],
        declaration["parameters"],
        root=ROOT,
        output=args.output / "preflight",
        controller_python=Path(default_runtimes()["python_bin"]),
    )
    goal = GoalSpec(
        args.objective,
        CRITERIA,
        max_cycles=args.max_cycles,
        model_calls=args.model_calls,
        tool_calls=args.tool_calls,
        wall_seconds=args.wall_seconds,
    )
    store = GoalStore.create(
        args.output,
        goal,
        profile,
        condition=getattr(args, "condition", "science_goal"),
        admission={"review_continuation": REVIEW_CONTINUATION},
    )
    files = research_implementation_paths(ROOT)
    hashes = archive_implementation(files, store.root)
    write_json(
        store.root / "protocol.json",
        {
            "case": args.case,
            "review_continuation": REVIEW_CONTINUATION,
            "allocation": declaration,
            "implementation": hashes,
            "cycle_wall_seconds": declaration["job_timeout"] + profile.budget.wall_seconds,
            "scope": "Expert computational reproduction, source-bound review and workflow reuse; not autonomous model invention or complete publication reproduction",
        },
    )
    return store


async def main(args):
    store = prepare(args) if args.action == "create" else GoalStore(args.output)
    if args.action == "pause":
        state = store.pause()
    elif args.action in {"create", "status"}:
        state = store.read()
    else:
        if args.action == "resume":
            store.resume()
        profile = Profile.from_dict(store.read()["definition"]["profile"])
        key = (
            credential_from_toml(args.credential_toml, profile.model)
            if args.credential_toml
            else os.environ.get(profile.model.credential_env)
        )
        if not key:
            raise ValueError("Goal model credential is unresolved")
        plugin = ReproductionGoal(store, key)
        state = await run_goal(
            store, plugin.cycle, plugin.evaluate, cycle_wall_seconds=plugin.protocol["cycle_wall_seconds"]
        )
    print(
        json.dumps(
            {
                "status": state["status"],
                "episodes": len(state["cycles"]),
                "usage": state["usage"],
                "criteria": state["assessment"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["create", "run", "pause", "resume", "status"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--objective")
    parser.add_argument("--case", choices=sorted(CASES), default="phl7_wt_hrex_cuda")
    parser.add_argument("--max-cycles", type=int, default=4)
    parser.add_argument("--model-calls", type=int, default=96)
    parser.add_argument("--tool-calls", type=int, default=48)
    parser.add_argument("--wall-seconds", type=float, default=7200)
    parser.add_argument("--credential-toml", type=Path)
    asyncio.run(main(parser.parse_args()))
