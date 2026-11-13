#!/usr/bin/env python3
"""Explicit Scientist runtime acceptance matrix; dry-run never resolves credentials.

This is separate from the scientific comparison in run_harness_matrix.py.
A successful runtime does not establish scientific utility or native compaction.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import asyncio
import json
import os
from pathlib import Path
import re
import sys
import sqlite3
import tempfile
import hashlib

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

os.environ["PYTHON_DOTENV_DISABLED"] = "1"

from agent.harness import Profile, builtin_registry
from agent.harness.catalog import ExecutionCatalog
from agent.harness.contracts import digest, file_hash
from agent.harness.host import write_json
from agent.harness.installation import default_runtimes

SMOKE_ID = "synthetic-token-counts.v1"
SMOKE_OBJECTIVE = (
    "Perform a non-research software integration smoke check. Read the supplied "
    "synthetic tokens, count each distinct token, and save a short analysis report "
    "with the observed counts and input source. Do not train models, select a "
    "scientific topic, claim scientific utility, or publish a reusable capability."
)
SMOKE_INPUT = {"tokens": ["alpha", "beta", "alpha", "gamma", "beta", "alpha"]}
LIMIT_KEYS = {"max_cycles", "model_calls", "tool_calls", "wall_seconds"}


def zero_price(row):
    price = row.get("pricing")
    return (isinstance(price, dict)
            and set(price) == {"input_per_million", "output_per_million"}
            and all(type(v) in (int, float) and v == 0 for v in price.values()))


@contextmanager
def readonly_policy():
    """Use the frozen-policy interface without SQLite write side effects."""
    from mcp_tools.tools.capability_policy import CapabilityStore, mode
    from mcp_tools.tools.capability_snapshot import document, read
    store = CapabilityStore()
    if store.snapshot is not None:
        rows = read(store.snapshot)
    elif store.path.exists():
        with sqlite3.connect(store.path.resolve().as_uri() + "?mode=ro", uri=True) as conn:
            conn.row_factory = sqlite3.Row
            rows = {row["key"]: dict(row) for row in conn.execute(
                "SELECT key, enabled, revision, selector FROM switches")}
    else:
        rows = {}
    previous = os.environ.get("VENUS_CAPABILITY_POLICY_SNAPSHOT")
    with tempfile.TemporaryDirectory(prefix="venus-matrix-policy-") as temp:
        path = Path(temp) / "policy.json"
        path.write_text(json.dumps(document(rows, mode())))
        path.chmod(0o600)
        os.environ["VENUS_CAPABILITY_POLICY_SNAPSHOT"] = str(path)
        try:
            yield
        finally:
            if previous is None:
                os.environ.pop("VENUS_CAPABILITY_POLICY_SNAPSHOT", None)
            else:
                os.environ["VENUS_CAPABILITY_POLICY_SNAPSHOT"] = previous


def build_plan(catalog_path, base_profile, runtimes, runtime_root):
    with readonly_policy():
        return _build_plan(catalog_path, base_profile, runtimes, runtime_root)


def _build_plan(catalog_path, base_profile, runtimes, runtime_root):
    """Inventory every configured enabled free model x configured Scientist harness.

    Publication and runtime gates are observed without resolving credentials.
    Static admission is never called live availability.
    """
    catalog_path = Path(catalog_path)
    raw = json.loads(catalog_path.read_text())
    catalog = ExecutionCatalog(
        ROOT, base_profile, path=catalog_path, free_only=True, include_unlisted=True,
    )
    harnesses = raw.get("harnesses", {}).get("science_expert")
    if not isinstance(harnesses, list) or not harnesses or len(set(harnesses)) != len(harnesses):
        raise ValueError("Explicit unique science_expert harnesses are required")
    registry = builtin_registry()
    models = [row for row in raw["models"] if row.get("enabled", True) is True and zero_price(row)]
    if not models:
        raise ValueError("No enabled zero-price routes")
    cells = []
    for row in models:
        for harness in harnesses:
            cell = {
                "id": f"cell_{len(cells):03d}", "model_id": row["id"], "harness": harness,
                "platform": {"status": "unverified"},
                "research": {"status": "not_started"},
                "availability": {"static": "configured_zero_price", "effective": "not_probed",
                                 "credential": "not_checked"},
                "compatibility": {"status": "unverified", "reasons": []},
                "compression": {"status": "not_observed", "evidence": []},
                "execution": "not_started",
            }
            try:
                profile, _ = catalog.select({"harness": harness, "model_id": row["id"]})
                check = registry.preflight(profile, runtimes, include_runtime_digest=False)
                cell["profile"] = profile.public()
                cell["compatibility"] = {
                    "status": "statically_admitted" if check["ready"] else "blocked",
                    "reasons": check["reasons"],
                }
                cell["availability"]["configuration"] = "admitted" if check["ready"] else "blocked"
            except ValueError:
                # Do not copy configuration exceptions: they may contain private paths.
                cell["compatibility"] = {"status": "blocked", "reasons": ["execution_policy_incompatible"]}
                cell["availability"]["configuration"] = "blocked"
            cells.append(cell)
    return {
        "schema": 1, "kind": "scientist_runtime_acceptance", "status": "dry_run",
        "catalog_sha256": file_hash(catalog_path),
        "runtime_root": str(Path(runtime_root).resolve()),
        "runtimes": {k: str(v) for k, v in runtimes.items()},
        "coverage": {"models": len(models), "harnesses": len(harnesses), "cells": len(cells)},
        "claim": "Static inventory; credentials and live provider availability have not been checked.",
        "cells": cells,
    }


def load_case(path):
    """Require a predeclared smoke fixture or an explicitly approved topic."""
    path = Path(path).resolve()
    value = json.loads(path.read_text())
    allowed = {"schema", "id", "kind", "objective", "limits", "inputs", "approval"}
    if not isinstance(value, dict) or set(value) - allowed or value.get("schema") != 1:
        raise ValueError("Invalid case contract")
    if not isinstance(value.get("id"), str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", value["id"]):
        raise ValueError("Explicit case identity required")
    limits = value.get("limits")
    if (not isinstance(limits, dict) or set(limits) != LIMIT_KEYS
            or any(type(v) is not int or v <= 0 for v in limits.values())):
        raise ValueError("Declare positive finite integer goal budgets")
    if value.get("kind") == "non_research_smoke":
        if value["id"] != SMOKE_ID or value.get("objective", SMOKE_OBJECTIVE) != SMOKE_OBJECTIVE:
            raise ValueError("Only the predeclared synthetic-token smoke case is supported")
        if value.get("inputs") or value.get("approval"):
            raise ValueError("Smoke case inputs and objective are fixed")
        value["objective"] = SMOKE_OBJECTIVE
    elif value.get("kind") == "approved_topic":
        approval = value.get("approval")
        if (not isinstance(approval, dict) or set(approval) != {"status", "reference"}
                or approval.get("status") != "approved"
                or not isinstance(approval.get("reference"), str) or not approval["reference"].strip()):
            raise ValueError("An explicit topic approval reference is required")
        if not isinstance(value.get("objective"), str) or not value["objective"].strip():
            raise ValueError("An approved objective is required; automatic topic selection is forbidden")
    else:
        raise ValueError("Select non_research_smoke or approved_topic explicitly")
    inputs = []
    for row in value.get("inputs", []):
        if not isinstance(row, dict) or set(row) != {"path", "sha256"}:
            raise ValueError("Each supplied input requires path and SHA256")
        relative = Path(row["path"])
        if relative.is_absolute() or ".." in relative.parts or any(part.startswith(".") for part in relative.parts):
            raise ValueError("Inputs must be ordinary files within the case directory")
        resolved = (path.parent / relative).resolve()
        if not resolved.is_relative_to(path.parent) or not resolved.is_file() or file_hash(resolved) != row["sha256"]:
            raise ValueError("Input is missing, out of scope, or changed")
        inputs.append(resolved)
    return value, inputs


def native_type_hash_matches(row, normalized_path):
    """Bind each normalized observation to retained native bytes, not model text."""
    reference = row.get("native_event_path")
    expected = row.get("native_event_sha256")
    if reference not in {"runtime_events.jsonl", "native_events.jsonl"}:
        return False
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
        return False
    path = normalized_path.parent / reference
    if path.is_symlink() or not path.is_file():
        return False
    for line in path.read_text().splitlines():
        try:
            raw = json.loads(line)
        except ValueError:
            continue
        encoded = json.dumps(raw, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
        if hashlib.sha256(encoded).hexdigest() != expected or not isinstance(raw, dict):
            continue
        # The hash must identify the admitted native type, not an arbitrary model message.
        native_type = row.get("native_type")
        if raw.get("type") == "compaction/end":
            failed = bool((raw.get("data") or {}).get("error"))
            return native_type == "compaction/end" and (
                row.get("phase") == "completed" and not failed
                or row.get("phase") in {"failed", "blocked", "cancelled"} and failed)
        if raw.get("type") == "system" and raw.get("subtype") == "compact_boundary":
            return row.get("phase") == "completed" and native_type in {
                "system/compact_boundary", "system.compact_boundary", "compact_boundary"}
        if row.get("phase") not in {"completed", "failed", "blocked", "cancelled"}:
            # Non-terminal notices remain observations, never completion evidence.
            return isinstance(native_type, str) and (
                native_type == raw.get("type")
                or native_type.startswith(str(raw.get("type")) + "."))
    return False


def compression_evidence(goal_root, *, native_contexts=()):
    """Keep recovered failures, and require a raw boundary plus same-session tools."""
    observations = []
    stream_states = []
    for path in sorted((Path(goal_root) / "episodes").rglob("events.jsonl")):
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        terminals = []
        for index, row in enumerate(rows):
            if row.get("kind") != "context_compaction":
                continue
            sid = row.get("session_id")
            verified = native_type_hash_matches(row, path)
            evidence = {"path": str(path.relative_to(goal_root)), "line": index + 1,
                        "phase": row.get("phase"), "session_id": sid, "raw_type_hash_checked": verified,
                        "native_event_sha256": row.get("native_event_sha256")}
            completed = verified and row.get("phase") == "completed" and isinstance(sid, str) and bool(sid)
            starts = set()
            continued = False
            for event in rows[index + 1:]:
                if not completed or event.get("session_id") != sid:
                    continue
                call_id = event.get("tool_call_id")
                if event.get("kind") == "tool_call_start" and isinstance(call_id, str) and call_id:
                    starts.add(call_id)
                if event.get("kind") == "tool_result" and call_id in starts and event.get("is_error") is False:
                    continued = True
            evidence["continued"] = continued
            observations.append(evidence)
            if row.get("phase") in {"completed", "failed", "blocked", "cancelled"}:
                terminals.append(evidence)
        if terminals:
            last = terminals[-1]
            stream_states.append(
                "observed_continued" if last["continued"]
                else "observed_failed" if last["raw_type_hash_checked"] and last["phase"] in {"failed", "blocked", "cancelled"}
                else "continuation_unverified")
    status = "not_observed"
    if observations:
        status = "continuation_unverified"
        if "observed_failed" in stream_states:
            status = "observed_failed"
        elif stream_states and all(state == "observed_continued" for state in stream_states):
            status = "observed_continued"
    failures = sum(row["raw_type_hash_checked"] and row["phase"] in {"failed", "blocked", "cancelled"}
                   for row in observations)
    candidate_status = status
    # Hash/type matches and normalized tools alone cannot bind raw sessions,
    # notifications, terminal result or the host request interval. Only a verified
    # persisted host binding may supply that context; never manufacture one here.
    status = "continuation_unverified" if observations else "not_observed"
    verified_roots, witnesses, problems = set(), [], []
    if native_contexts:
        from agent.harness.native_context_recovery import verify_native_continuation
        for native in native_contexts:
            try:
                native_root = Path(native["root"]).resolve()
                native_root.relative_to(Path(goal_root).resolve() / "episodes")
                verified = verify_native_continuation(native)
                completed = [witness for witness in verified if witness["boundary_phase"] == "completed"]
                if completed:
                    verified_roots.add(native_root)
                    witnesses.extend(completed)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                problems.append(type(exc).__name__)
    required_roots = {(Path(goal_root) / row["path"]).parent.resolve() for row in observations}
    if required_roots and required_roots <= verified_roots:
        status = "observed_continued"
    return {"status": status, "evidence": observations, "verified_witnesses": witnesses,
            "verification_problems": problems, "observed_failure_count": failures,
            "recovery_candidate": candidate_status == "observed_continued" and bool(failures),
            "recovered": status == "observed_continued" and bool(failures)}

def load_cycle_native_contexts(folder, state):
    """Read only the two host-owned Web phase roots; never derive call intervals."""
    folder = Path(folder)
    paths = [folder / "native/run", folder / "execution_retry/native/run"]
    if not any((root / "native_binding.json").exists() for root in paths):
        return []
    from agent.harness.native_expert_team import NATIVE_EXPERT_HARNESSES
    from agent.harness.native_context_recovery import native_context_for_run
    goal_harness = state.get("definition", {}).get("profile", {}).get("harness")
    expected_harness = NATIVE_EXPERT_HARNESSES.get(goal_harness)
    if expected_harness is None:
        raise ValueError("missing_goal_harness_binding")
    contexts = []
    last_after = None
    for root in paths:
        binding_path = root / "native_binding.json"
        if not binding_path.exists():
            continue
        for path in (binding_path, root / "native_result.json"):
            path.resolve().relative_to(folder.resolve())
            if any(part.is_symlink() for part in (path, *path.parents) if part != folder.parent):
                raise ValueError("symlinked_native_binding")
        binding = json.loads(binding_path.read_text())
        result = json.loads((root / "native_result.json").read_text())
        tools = binding.get("tools")
        before, after = binding.get("calls_before"), binding.get("calls_after")
        if (binding.get("schema") != "web_native_phase.v1"
                or binding.get("harness") != expected_harness
                or binding.get("native_result_sha256") != digest(result)
                or binding.get("session_id") != result.get("session_id")
                or not isinstance(tools, list) or any(not isinstance(tool, str) for tool in tools)
                or len(set(tools)) != len(tools) or tools != sorted(tools)
                or binding.get("tools_sha256") != digest(tools)
                or type(binding.get("tool_limit")) is not int or binding["tool_limit"] <= 0):
            raise ValueError("native_phase_binding_changed")
        # Ordinary Agent and unstarted native results may retain unknown counters.
        # Those records are evidence of an attempt, not authority to forgive calls.
        if not isinstance(binding.get("session_id"), str) or not binding["session_id"] or before is None or after is None:
            continue
        if (type(before) is not int or type(after) is not int or not 0 <= before < after
                or last_after is not None and before < last_after):
            raise ValueError("invalid_native_phase_interval")
        native = native_context_for_run(root, result, expected_harness,
                                       frozenset(tools), binding["tool_limit"], after)
        native["calls_before"] = before
        contexts.append(native)
        last_after = after
    return contexts


def classify(goal_root, state, kind):
    from agent.harness.events import audit_requests
    from agent.harness.model_availability import model_service_blocker
    from agent.harness.failure_summary import public_failure
    root = Path(goal_root)
    cycles = state.get("cycles", [])
    research = {"status": "accepted" if state.get("status") == "completed" else "incomplete",
                "goal_status": state.get("status")}
    if state.get("progress", {}).get("sc_revision"):
        research["status"] = "rejected"
    if kind == "non_research_smoke":
        research["status"] = "not_applicable_smoke"
    platform = {"status": "unverified"}
    availability = {"static": "configured_zero_price", "effective": "unverified", "credential": "resolved_by_host"}
    evidence = []
    requests_ok = []
    blockers = []
    runtime_errors = []
    native_contexts = []
    binding_problems = []
    external_diagnostics = []
    external_codes = {"context_limit", "model_access_401", "model_access_403",
                      "upstream_rate_limited", "upstream_timeout", "model_unavailable",
                      "model_response_empty", "model_response_refused", "model_response_invalid"}
    completed_native = False
    for folder in sorted((root / "episodes").glob("cycle_*")):
        gateway = folder / "model_requests.jsonl"
        gateway_code = None
        try:
            phase_contexts = load_cycle_native_contexts(folder, state)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            phase_contexts = []
            binding_problems.append({"path": str(folder.relative_to(root)), "error_type": type(exc).__name__})
        native_contexts.extend(phase_contexts)
        if gateway.exists():
            blocker = model_service_blocker(gateway)
            if blocker:
                blockers.append(blocker)
            requests_ok.append(audit_requests(gateway, native_context=phase_contexts or None)["passed"])
            gateway_code = public_failure(gateway, fallback="unverified")["code"]
            if gateway_code in external_codes:
                external_diagnostics.append(gateway_code)
            evidence.append(str(gateway.relative_to(root)))
        failure = folder / "failure.json"
        if failure.exists():
            record = json.loads(failure.read_text())
            # Generic research validation errors do not prove platform failure.
            if not record.get("code") and record.get("message") in {"runtime_error", "protocol_failed", "Native turn failed"}:
                record["code"] = gateway_code if gateway_code in external_codes else record["message"]
            if record.get("code") in external_codes:
                external_diagnostics.append(record["code"])
            if record.get("error_type") in {"ModuleNotFoundError", "ImportError"} or record.get("code") in {
                    "runtime_error", "protocol_failed", "model_protocol_incompatible"}:
                runtime_errors.append({k: record[k] for k in ("error_type", "code") if k in record})
        events = folder / "events.json"
        if events.exists():
            for event in json.loads(events.read_text()):
                if event.get("kind") == "completed" and event.get("event_audit", {}).get("passed") is True:
                    completed_native = True
    if blockers:
        availability.update(effective=blockers[-1]["code"], http_status=blockers[-1]["http_status"])
    elif external_diagnostics:
        availability["effective"] = external_diagnostics[-1]
    elif requests_ok and all(requests_ok):
        availability["effective"] = "observed_available"
    if runtime_errors:
        platform.update(status="error", errors=runtime_errors)
    elif (cycles and requests_ok and all(requests_ok) and completed_native and not external_diagnostics and not binding_problems
          and all(c.get("status") in {"completed", "needs_revision"} for c in cycles)):
        platform["status"] = "pass"
    platform["evidence"] = evidence
    if binding_problems:
        platform["native_binding_problems"] = binding_problems
    return {"platform": platform, "research": research, "availability": availability,
            "compression": compression_evidence(root, native_contexts=native_contexts)}


async def execute_cell(cell, folder, case, inputs, catalog, runtimes):
    """Credential resolution is confined to the existing host resolver."""
    from agent.harness.autonomous_research import AutonomousResearch, create
    from agent.harness.catalog import resolve_credential
    from agent.harness.research_cli_inputs import prepare_inputs, stage_inputs
    profile, option = catalog.select({"harness": cell["harness"], "model_id": cell["model_id"]})
    if profile.public() != cell["profile"]:
        raise ValueError("Frozen profile changed")
    # Never resolve credentials during inventory, and never persist their values.
    try:
        credential = resolve_credential(profile.model, option.credential_toml)
    except (ValueError, OSError, KeyError):
        cell["availability"].update(effective="credential_unavailable", credential="unavailable")
        cell["execution"] = "blocked"
        return
    if case["kind"] == "approved_topic":
        for path, expected in zip(inputs, case.get("inputs", []), strict=True):
            if file_hash(path) != expected["sha256"]:
                raise ValueError("Approved input changed after allocation")
    store = create(folder / "goal", profile, case["objective"], case["limits"])
    if case["kind"] == "non_research_smoke":
        fixture = folder / "smoke_input.json"
        write_json(fixture, SMOKE_INPUT)
        inputs = [fixture]
    with prepare_inputs([str(path) for path in inputs]) as prepared:
        if prepared:
            stage_inputs(store, prepared)
    runner = AutonomousResearch(store, credential, runtimes, **({"tools": []} if case["kind"] == "non_research_smoke" else {}))
    state = await runner.run()
    cell.update(classify(store.root, state, case["kind"]))
    cell["execution"] = "finished"


async def _run(args):
    if args.execute and not args.case:
        raise ValueError("--execute requires an explicitly predeclared --case")
    case, inputs = load_case(args.case) if args.case else (None, [])
    profile = Profile.from_dict(json.loads(args.profile.read_text()))
    runtimes = default_runtimes(args.runtime_root)
    plan = build_plan(args.catalog, profile, runtimes, args.runtime_root)
    if case:
        plan["case"] = case
        plan["case_sha256"] = digest(case)
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "allocation.json", plan)
    write_json(args.output / "report.json", plan)
    for cell in plan["cells"]:
        folder = args.output / cell["id"]
        folder.mkdir()
        write_json(folder / "report.json", cell)
    if not args.execute:
        return plan
    # Do not load dotenv or mutate deployment configuration.
    os.environ["PYTHON_DOTENV_DISABLED"] = "1"
    os.environ["VENUS_EXECUTION_CATALOG"] = str(args.catalog.resolve())
    catalog = ExecutionCatalog(ROOT, profile, path=args.catalog, free_only=True, include_unlisted=True)
    plan["status"] = "running"
    for cell in plan["cells"]:
        folder = args.output / cell["id"]
        if cell["compatibility"]["status"] == "blocked":
            cell["execution"] = "blocked"
        else:
            cell["execution"] = "running"
            write_json(args.output / "report.json", plan)
            try:
                if file_hash(args.catalog) != plan["catalog_sha256"]:
                    raise ValueError("Catalog changed after allocation")
                await execute_cell(cell, folder, case, inputs, catalog, runtimes)
            except Exception as exc:
                cell["execution"] = "failed"
                cell["platform"] = {"status": "error", "error_type": type(exc).__name__}
        write_json(folder / "report.json", cell)
        write_json(args.output / "report.json", plan)
    plan["status"] = "finished"
    plan["claim"] = "All allocated cells retained; platform, research and compaction outcomes are independent."
    write_json(args.output / "report.json", plan)
    return plan


async def run(args):
    previous = os.environ.get("VENUS_EXECUTION_CATALOG")
    try:
        with readonly_policy():
            return await _run(args)
    finally:
        if previous is None:
            os.environ.pop("VENUS_EXECUTION_CATALOG", None)
        else:
            os.environ["VENUS_EXECUTION_CATALOG"] = previous


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=ROOT / "configs/harness/free_execution_catalog.json")
    parser.add_argument("--profile", type=Path, default=ROOT / "configs/harness/free_science_goal.json")
    parser.add_argument("--runtime-root", type=Path, default=ROOT, help="Installation root; may differ from the source worktree")
    parser.add_argument("--case", type=Path, help="Explicit reviewed case contract; required with --execute")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    result = asyncio.run(run(parser.parse_args(argv)))
    print(json.dumps({"status": result["status"], **result["coverage"]}))


if __name__ == "__main__":
    main()
